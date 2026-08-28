"""Append-only, hash-chained provenance ledger.

Every agent-initiated action is recorded before and after execution. The
records form a hash chain (each entry commits to its predecessor's digest), so
any retroactive edit or deletion is detectable by re-walking the chain — the
property GxP audit-trail rules and ALCOA+ ask for, and the property a
reproducibility claim needs regardless of regulation.

Storage is SQLite plus a mirrored JSONL file. SQLite gives queryability; the
JSONL gives a format that outlives this codebase and can be handed to a
collaborator or an auditor with no software at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

GENESIS = "0" * 64

RecordKind = Literal[
    "session_start", "session_end", "device_connect", "device_disconnect",
    "command_request", "command_result", "property_write", "property_read",
    "safety_decision", "approval", "simulation", "event", "artifact",
    "run_start", "run_step", "run_end", "estop", "note", "policy_change",
]


def _canonical(obj: Any) -> str:
    """Deterministic JSON so the digest is stable across processes/versions."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class Record(BaseModel):
    """One ledger entry. ALCOA+: attributable, legible, contemporaneous,
    original, accurate — plus complete, consistent, enduring, available."""

    model_config = ConfigDict(extra="forbid")

    seq: int = 0
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = Field(default_factory=time.time)
    kind: str = "note"
    #: Who caused this. "agent:<model>", "human:<id>", "system".
    actor: str = "system"
    session_id: str = ""
    run_id: str | None = None
    job_id: str | None = None
    device_id: str | None = None
    feature: str | None = None
    command: str | None = None
    #: Why — free text the agent supplies; required for hazardous actions.
    reason: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    prev_hash: str = GENESIS
    hash: str = ""

    def compute_hash(self) -> str:
        body = self.model_dump(exclude={"hash"})
        return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    id         TEXT NOT NULL UNIQUE,
    timestamp  REAL NOT NULL,
    kind       TEXT NOT NULL,
    actor      TEXT NOT NULL,
    session_id TEXT NOT NULL,
    run_id     TEXT,
    job_id     TEXT,
    device_id  TEXT,
    feature    TEXT,
    command    TEXT,
    reason     TEXT,
    payload    TEXT NOT NULL,
    prev_hash  TEXT NOT NULL,
    hash       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ledger_run    ON ledger(run_id);
CREATE INDEX IF NOT EXISTS ix_ledger_device ON ledger(device_id);
CREATE INDEX IF NOT EXISTS ix_ledger_kind   ON ledger(kind);
CREATE INDEX IF NOT EXISTS ix_ledger_time   ON ledger(timestamp);
"""


class Ledger:
    """Thread-safe hash-chained store.

    Writes are serialised under one lock: the chain is only meaningful if
    `prev_hash` is read and the successor written atomically.
    """

    def __init__(self, path: str | os.PathLike[str], *, session_id: str = "") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.path.with_suffix(".jsonl")
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        # WAL keeps the ledger readable by a monitoring process mid-experiment.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- writing ----------------------------------------------------------

    def head(self) -> tuple[int, str]:
        row = self._conn.execute(
            "SELECT seq, hash FROM ledger ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return (row[0], row[1]) if row else (0, GENESIS)

    def append(self, record: Record) -> Record:
        with self._lock:
            seq, prev = self.head()
            record.seq = seq + 1
            record.prev_hash = prev
            record.session_id = record.session_id or self.session_id
            record.hash = record.compute_hash()
            self._conn.execute(
                "INSERT INTO ledger (seq,id,timestamp,kind,actor,session_id,run_id,"
                "job_id,device_id,feature,command,reason,payload,prev_hash,hash) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.seq, record.id, record.timestamp, record.kind, record.actor,
                    record.session_id, record.run_id, record.job_id, record.device_id,
                    record.feature, record.command, record.reason,
                    _canonical(record.payload), record.prev_hash, record.hash,
                ),
            )
            self._conn.commit()
            with self.jsonl.open("a", encoding="utf-8") as fh:
                fh.write(_canonical(record.model_dump()) + "\n")
            return record

    def log(self, kind: str, **fields: Any) -> Record:
        payload = fields.pop("payload", {}) or {}
        return self.append(Record(kind=kind, payload=payload, **fields))

    # -- reading ----------------------------------------------------------

    def _row_to_record(self, row: sqlite3.Row | tuple) -> Record:
        (seq, rid, ts, kind, actor, sess, run, job, dev, feat, cmd,
         reason, payload, prev_hash, h) = row
        return Record(
            seq=seq, id=rid, timestamp=ts, kind=kind, actor=actor, session_id=sess,
            run_id=run, job_id=job, device_id=dev, feature=feat, command=cmd,
            reason=reason or "", payload=json.loads(payload),
            prev_hash=prev_hash, hash=h,
        )

    def query(
        self, *, run_id: str | None = None, device_id: str | None = None,
        kind: str | None = None, since: float | None = None, limit: int = 200,
    ) -> list[Record]:
        clauses, params = [], []
        for col, val in (("run_id", run_id), ("device_id", device_id), ("kind", kind)):
            if val is not None:
                clauses.append(f"{col} = ?")
                params.append(val)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        sql = "SELECT * FROM ledger"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY seq DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in reversed(rows)]

    def iter_all(self) -> Iterator[Record]:
        for row in self._conn.execute("SELECT * FROM ledger ORDER BY seq ASC"):
            yield self._row_to_record(row)

    # -- integrity --------------------------------------------------------

    def verify(self) -> dict[str, Any]:
        """Re-walk the chain. Returns the first break, if any.

        This is the operation an auditor (or a reviewer asking "did anything
        touch this dataset after the fact?") actually runs.
        """
        prev = GENESIS
        count = 0
        for rec in self.iter_all():
            count += 1
            if rec.prev_hash != prev:
                return {
                    "valid": False, "records": count, "broken_at": rec.seq,
                    "reason": "prev_hash does not match predecessor",
                    "expected": prev, "found": rec.prev_hash,
                }
            recomputed = rec.compute_hash()
            if recomputed != rec.hash:
                return {
                    "valid": False, "records": count, "broken_at": rec.seq,
                    "reason": "record content does not match its digest",
                    "expected": recomputed, "found": rec.hash,
                }
            prev = rec.hash
        return {"valid": True, "records": count, "head": prev}
