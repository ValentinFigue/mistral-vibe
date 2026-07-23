"""Content-addressed result cache, backed by a single SQLite file.

Results are keyed by their node fingerprint (the recipe hash). This one store serves
several jobs at once:

* **intra-run incrementality** — a re-run re-pays only for nodes whose fingerprint changed;
* **crash-resume** — a killed run finds every completed node already present;
* **cross-run dedup** — an identical fingerprint is a silent ``INSERT OR IGNORE`` no-op;
* **cross-session reuse** — the shared store (:func:`open_shared_cache`) lives under
  ``VIBE_HOME`` (not the session dir), so an operation computed in one session is an instant
  cache hit in the next. It is used by both graph-authoring agents (``graph`` and ``analyst``).

Because the cache keys on the recipe (not the output), an entry is **trusted blindly**. Two
consequences follow from making it cross-session:

* **Operator drift.** The fingerprint omits operator *code* version, so a changed operator
  could return a stale cached result. Guard it at the storage layer: :data:`CACHE_VERSION` is
  baked into the shared filename (``graph-cache-v{N}.sqlite``); bump it on any operator-semantics
  change for a clean, total invalidation (the old file becomes garbage-collectable).
* **Trust / privacy.** Cross-*user* sharing stays out of scope (the store is single-user, local).
  Payloads (possibly PII from ``read_csv``) now persist across sessions and projects, so
  :meth:`CacheStore.clear` (and deleting ``~/.vibe/graph-cache-*.sqlite``) is the purge path.

The store is append-only by default (``byte_size``/``created_at`` are carried for LRU/size GC).
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import time

from vibe.core.logger import logger

# Bump when an operator's *semantics* change, to invalidate every cross-session cached result
# (the recipe fingerprint does not capture operator code version). Baked into the shared filename.
# v2: `describe` output gained median/p25/p75 columns.
# v3: `describe`/`correlation` label column renamed `column` → `field` (reserved-word fix).
CACHE_VERSION = 3

# Soft byte budget for the shared store. Enforced once per authoring run (not per put), so old
# results are trimmed as new analyses accumulate across sessions. A cap, not a hard limit.
SHARED_CACHE_MAX_BYTES = 512 * 1024 * 1024

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    fingerprint TEXT PRIMARY KEY,
    op_type     TEXT NOT NULL,
    payload     BLOB NOT NULL,
    byte_size   INTEGER NOT NULL,
    created_at  REAL NOT NULL
)
"""


class CacheStore:
    """A SQLite-backed content-addressed store of serialized node results."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # WAL: durable-after-commit resumability + non-blocking readers.
        # M1 uses a single connection (cooperative asyncio); real thread/process
        # parallelism in M2 will need per-thread connections + SQLITE_BUSY retry.
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        # Shared store: a concurrent session may hold the write lock. Wait instead of raising
        # SQLITE_BUSY — writes are a single idempotent INSERT OR IGNORE, so a short wait suffices.
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get(self, fingerprint: str) -> bytes | None:
        """Return the stored payload bytes for a fingerprint, or ``None`` on a miss."""
        row = self._conn.execute(
            "SELECT payload FROM results WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        return row[0] if row is not None else None

    def put(
        self,
        fingerprint: str,
        payload: bytes,
        *,
        op_type: str,
        created_at: float | None = None,
    ) -> bool:
        """Store ``payload`` if absent. Returns ``True`` if inserted, ``False`` if deduped."""
        created = time.time() if created_at is None else created_at
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO results (fingerprint, op_type, payload, byte_size, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (fingerprint, op_type, payload, len(payload), created),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def row_count(self) -> int:
        """Number of distinct results currently stored."""
        return self._conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]

    def clear(self) -> int:
        """Drop every stored result. Returns the number of rows removed.

        The purge path for the shared cross-session store: recovers from a poisoned entry (a
        non-pure operator can persist a wrong result globally) and removes payload residue.
        """
        removed = self.row_count()
        self._conn.execute("DELETE FROM results")
        self._conn.commit()
        return removed

    def total_bytes(self) -> int:
        """Total payload bytes currently stored (sum of ``byte_size``)."""
        return self._conn.execute("SELECT COALESCE(SUM(byte_size), 0) FROM results").fetchone()[0]

    def evict_to(self, max_bytes: int) -> int:
        """Evict oldest results (by ``created_at``) until the total payload is ≤ ``max_bytes``.

        Returns the number of rows removed. Correctness is never at stake — an evicted entry is
        at worst a recompute; a running graph's within-run dependencies are held in the executor's
        in-memory ``results``, never re-read from here. Cheap when already under budget (one SUM).
        """
        total = self.total_bytes()
        if total <= max_bytes:
            return 0
        victims: list[str] = []
        for fp, size in self._conn.execute(
            "SELECT fingerprint, byte_size FROM results ORDER BY created_at ASC"
        ).fetchall():
            if total <= max_bytes:
                break
            victims.append(fp)
            total -= size
        if victims:
            self._conn.executemany(
                "DELETE FROM results WHERE fingerprint = ?", [(fp,) for fp in victims]
            )
            self._conn.commit()
            logger.info(
                "graph cache: evicted %d oldest result(s) to stay under %d bytes",
                len(victims),
                max_bytes,
            )
        return len(victims)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> CacheStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def shared_cache_path() -> Path:
    """Path of the shared, cross-session result store under ``VIBE_HOME``.

    Versioned by :data:`CACHE_VERSION` so an operator-semantics change starts a fresh file.
    Imported lazily to avoid a ``paths → graph`` import cycle (paths must not import graph).
    """
    from vibe.core.paths import VIBE_HOME

    return VIBE_HOME.path / f"graph-cache-v{CACHE_VERSION}.sqlite"


def open_shared_cache() -> CacheStore:
    """Open the shared cross-session cache (see :func:`shared_cache_path`)."""
    return CacheStore(shared_cache_path())
