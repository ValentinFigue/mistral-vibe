"""Content-addressed result cache, backed by a single SQLite file.

Results are keyed by their node fingerprint (the recipe hash). This one store serves
three jobs at once:

* **intra-run incrementality** — a re-run re-pays only for nodes whose fingerprint changed;
* **crash-resume** — a killed run finds every completed node already present;
* **cross-run dedup** — an identical fingerprint is a silent ``INSERT OR IGNORE`` no-op.

Cross-*user* dedup is deliberately out of scope: the cache keys on the recipe, not the
output, so an entry is trusted blindly — sharing a store across users is a trust boundary
the design has not yet specified. M1's store is single, local, and trusted.

The store is append-only in M1 (no eviction — it grows unbounded); the ``byte_size`` and
``created_at`` columns are carried so LRU/size GC is cheap to add later.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import time

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

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> CacheStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
