from __future__ import annotations

import pytest

from vibe.core.graph.cache import CacheStore
from vibe.core.graph.executor import execute
from vibe.core.graph.fingerprint import fingerprint_node
from vibe.core.graph.model import Graph, Node


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
