from __future__ import annotations

import pytest

from vibe.core.graph.cache import (
    CACHE_VERSION,
    CacheStore,
    open_shared_cache,
    shared_cache_path,
)
from vibe.core.graph.executor import execute
from vibe.core.graph.fingerprint import fingerprint_node
from vibe.core.graph.model import Graph, Node
from vibe.core.paths import VIBE_HOME


def test_put_get_roundtrip_and_dedup(cache: CacheStore) -> None:
    assert cache.get("fp1") is None
    assert cache.put("fp1", b'{"n": 1}', op_type="tst_const") is True
    assert cache.get("fp1") == b'{"n": 1}'
    # Second put with the same key is a silent INSERT OR IGNORE no-op.
    assert cache.put("fp1", b'{"n": 999}', op_type="tst_const") is False
    assert cache.get("fp1") == b'{"n": 1}'
    assert cache.row_count() == 1


def test_wal_mode_enabled(cache: CacheStore) -> None:
    mode = cache._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_busy_timeout_set(cache: CacheStore) -> None:
    # A concurrent session may hold the write lock; we wait rather than raise SQLITE_BUSY.
    assert cache._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_clear_empties_store(cache: CacheStore) -> None:
    cache.put("fp1", b"{}", op_type="tst_const")
    cache.put("fp2", b"{}", op_type="tst_const")
    assert cache.clear() == 2  # returns rows removed (the purge path)
    assert cache.row_count() == 0
    assert cache.get("fp1") is None


def test_shared_cache_path_is_versioned_under_vibe_home() -> None:
    # The shared store lives under VIBE_HOME (not a session dir) and carries CACHE_VERSION,
    # so an operator-semantics bump starts a fresh file. VIBE_HOME is a per-test tmp here.
    path = shared_cache_path()
    assert path.name == f"graph-cache-v{CACHE_VERSION}.sqlite"
    assert path.parent == VIBE_HOME.path


def test_open_shared_cache_persists_across_opens() -> None:
    # Two independent opens of the shared store (i.e. two sessions) see the same rows —
    # this is what makes a result computed in one session a cache hit in the next.
    with open_shared_cache() as first:
        first.put("shared_fp", b'{"n": 7}', op_type="tst_const")
    with open_shared_cache() as second:
        assert second.get("shared_fp") == b'{"n": 7}'


def test_evict_to_drops_oldest_first(cache: CacheStore) -> None:
    # created_at drives LRU order; each payload is 100 bytes.
    cache.put("old", b"x" * 100, op_type="tst_const", created_at=1.0)
    cache.put("mid", b"x" * 100, op_type="tst_const", created_at=2.0)
    cache.put("new", b"x" * 100, op_type="tst_const", created_at=3.0)
    assert cache.total_bytes() == 300

    removed = cache.evict_to(250)  # room for ~2 → evict the single oldest
    assert removed == 1
    assert cache.get("old") is None
    assert cache.get("mid") is not None and cache.get("new") is not None
    assert cache.total_bytes() <= 250


def test_evict_to_is_noop_under_budget(cache: CacheStore) -> None:
    cache.put("a", b"x" * 10, op_type="tst_const")
    assert cache.evict_to(1_000) == 0  # cheap SUM, nothing removed
    assert cache.row_count() == 1


def _chain(x: int, y: int) -> Graph:
    g = Graph()
    g.add(Node(id="a", op="tst_const", params={"value": x}))
    g.add(Node(id="b", op="tst_const", params={"value": y}))
    g.add(Node(id="sum", op="tst_add", inputs={"a": "a", "b": "b"}))
    return g


@pytest.mark.asyncio
async def test_cross_run_dedup(cache: CacheStore) -> None:
    """Two structurally identical graphs share cache rows: rows == distinct fingerprints."""
    g1, g2 = _chain(3, 4), _chain(3, 4)
    await execute(g1, cache)
    await execute(g2, cache)

    distinct = {fingerprint_node(g1, nid) for nid in g1.nodes}
    assert cache.row_count() == len(distinct) == 3
