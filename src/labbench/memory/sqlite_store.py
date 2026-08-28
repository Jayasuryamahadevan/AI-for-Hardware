"""SQLite-backed memory: the queryable half of the memory story.

Mirrors the ledger's choice of storage for the same reason: SQLite gives
indexed lookup by kind/run/device for free, and needs nothing beyond the
standard library. It is the right default when nobody has said otherwise,
which is why the gateway falls back to it when a lab configures no memory
backend at all (see `gateway.Gateway.__init__`).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .store import MemoryNotFound, MemoryRecord, MemoryStore, score, tokenise

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    title      TEXT NOT NULL,
    content    TEXT NOT NULL,
    tags       TEXT NOT NULL,
    metadata   TEXT NOT NULL,
    actor      TEXT NOT NULL,
    device_id  TEXT,
    run_id     TEXT,
    created    REAL NOT NULL,
    updated    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_memory_kind ON memory(kind);
CREATE INDEX IF NOT EXISTS ix_memory_run  ON memory(run_id);
"""


class SqliteMemory(MemoryStore):
    """One SQLite file holding every note and document.

    Configuration:

        memory:
          - id: notes
            backend: sqlite
            settings:
              path: ./labbench-data/memory.sqlite   # default, if omitted
    """

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)
        path = Path(config.get("path", "./labbench-data/memory.sqlite")).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()

    async def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- writing ------------------------------------------------------------

    async def write(self, record: MemoryRecord) -> MemoryRecord:
        with self._lock:
            self._conn.execute(
                "INSERT INTO memory (id,kind,title,content,tags,metadata,actor,device_id,"
                "run_id,created,updated) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET title=excluded.title, content=excluded.content, "
                "tags=excluded.tags, metadata=excluded.metadata, updated=excluded.updated",
                (
                    record.id, record.kind, record.title, record.content,
                    json.dumps(record.tags), json.dumps(record.metadata), record.actor,
                    record.device_id, record.run_id, record.created, record.updated,
                ),
            )
            self._conn.commit()
        return record

    async def delete(self, memory_id: str) -> None:
        with self._lock:
            cur = self._conn.execute("DELETE FROM memory WHERE id = ?", (memory_id,))
            self._conn.commit()
        if cur.rowcount == 0:
            raise MemoryNotFound(f"no memory {memory_id!r}", memory_id=memory_id)

    # -- reading --------------------------------------------------------------

    def _row(self, row: tuple) -> MemoryRecord:
        (mid, kind, title, content, tags, metadata, actor, device_id,
         run_id, created, updated) = row
        return MemoryRecord(
            id=mid, kind=kind, title=title, content=content,
            tags=json.loads(tags), metadata=json.loads(metadata), actor=actor,
            device_id=device_id, run_id=run_id, created=created, updated=updated,
        )

    async def get(self, memory_id: str) -> MemoryRecord:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memory WHERE id = ?", (memory_id,)
            ).fetchone()
        if row is None:
            raise MemoryNotFound(f"no memory {memory_id!r}", memory_id=memory_id)
        return self._row(row)

    async def list(
        self, *, kind: str | None = None, run_id: str | None = None, limit: int = 50,
    ) -> list[MemoryRecord]:
        clauses, params = [], []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        sql = "SELECT * FROM memory"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row(r) for r in rows]

    async def search(
        self, query: str = "", *, kind: str | None = None, tags: list[str] | None = None,
        run_id: str | None = None, device_id: str | None = None, limit: int = 20,
    ) -> list[MemoryRecord]:
        """Term-overlap ranking over title + content.

        Pulls every candidate row (after the cheap kind/run/device filters,
        which SQLite indexes) and scores in Python. A lab's memory store is
        hundreds of rows, not millions, so this is the honest trade: no
        dependency on a full-text-search extension or a vector database for a
        table that fits in memory ten times over.
        """
        clauses, params = [], []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if device_id is not None:
            clauses.append("device_id = ?")
            params.append(device_id)
        sql = "SELECT * FROM memory"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        records = [self._row(r) for r in rows]
        if tags:
            wanted = set(tags)
            records = [r for r in records if wanted.issubset(r.tags)]

        terms = tokenise(query)
        scored = [(score(terms, f"{r.title}\n{r.content}"), r) for r in records]
        scored = [(s, r) for s, r in scored if s > 0 or not query]
        scored.sort(key=lambda pair: (pair[0], pair[1].updated), reverse=True)
        return [r for _, r in scored[:limit]]
