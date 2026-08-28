"""The memory contract: durable notes and documents an agent can search.

`ledger.note` (see `core.provenance`) answers "what happened and when" — it is
append-only and tied to a moment, which is exactly right for an audit trail and
exactly wrong for a knowledge base. An agent that spends three days on one
cell line accumulates things worth *keeping and finding again*: which fields
of a plate were unusable, what focus offset this objective needs, the SOP a
human pasted in on day one. None of that is a record of an action; all of it
should still be there, and be searchable, next week.

That is what `MemoryStore` is for. Like `Device`, it is a small ABC so a new
backend is a new module, not a new agent-facing surface: `bridge/toolset.py`
calls `write`/`search`/`get`/`list`/`delete` and does not know or care whether
the words end up in SQLite or on disk as Markdown.

Two backends ship with the package (`sqlite_store.SqliteMemory`,
`filesystem.FilesystemDocs`), matching the same "queryable store plus a plain
format the auditor's own tools can read" split the provenance ledger makes.
"""

from __future__ import annotations

import abc
import re
import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import LabBenchError


class MemoryNotFound(LabBenchError):
    code = "memory_not_found"


class MemoryRecord(BaseModel):
    """One durable, searchable note or document.

    `kind` is free text rather than an enum on purpose: a lab invents its own
    taxonomy ("sop", "observation", "calibration") faster than a package
    maintainer can keep an enum in sync with it.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    kind: str = "note"
    title: str = ""
    content: str = ""
    tags: list[str] = Field(default_factory=list)
    #: Free-form, e.g. {"objective": "40x"}. Not searched; carried for the agent.
    metadata: dict[str, Any] = Field(default_factory=dict)
    actor: str = "agent"
    device_id: str | None = None
    run_id: str | None = None
    created: float = Field(default_factory=time.time)
    updated: float = Field(default_factory=time.time)

    def summary(self) -> dict[str, Any]:
        """Compact form for search results — a title and an excerpt, not the whole body."""
        excerpt = self.content if len(self.content) <= 240 else self.content[:237] + "..."
        return {
            "id": self.id, "kind": self.kind, "title": self.title, "excerpt": excerpt,
            "tags": self.tags, "actor": self.actor, "device_id": self.device_id,
            "run_id": self.run_id, "created": self.created, "updated": self.updated,
        }


_WORD = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    """Lowercase word split shared by every backend's relevance scoring.

    Deliberately not a real IR stack (no stemming, no IDF): the package has no
    dependency on one, per the project's "three dependencies, at runtime" rule,
    and a lab's memory store holds hundreds of notes, not millions — a plain
    term-overlap score is legible and good enough at that scale.
    """
    return _WORD.findall(text.lower())


def score(query_terms: list[str], text: str) -> float:
    """Fraction of query terms present in `text`, weighted by repetition.

    0 when the text matches nothing, so a caller can treat "no matching terms"
    the same as "not returned" without a separate relevance cutoff.
    """
    if not query_terms:
        return 1.0
    doc_terms = tokenise(text)
    if not doc_terms:
        return 0.0
    doc_counts: dict[str, int] = {}
    for t in doc_terms:
        doc_counts[t] = doc_counts.get(t, 0) + 1
    hits = sum(min(doc_counts.get(q, 0), 3) for q in query_terms)  # cap: one very
    return hits / (len(query_terms) * 3)                            # repeated word can't win alone


class MemoryStore(abc.ABC):
    """Base class every memory backend implements.

    Four operations, matching what an agent actually needs: write it down,
    find it again, read one back in full, and (rarely) remove it. There is no
    `update`: a memory is corrected by writing a new one and letting search
    surface the latest, which keeps the store append-friendly like the ledger
    it sits next to, without pretending to be one.
    """

    #: Optional import name whose absence means this backend cannot run.
    requires_package: str | None = None

    def __init__(self, **config: Any) -> None:
        self.config = config

    @abc.abstractmethod
    async def write(self, record: MemoryRecord) -> MemoryRecord: ...

    @abc.abstractmethod
    async def get(self, memory_id: str) -> MemoryRecord: ...

    @abc.abstractmethod
    async def search(
        self, query: str = "", *, kind: str | None = None, tags: list[str] | None = None,
        run_id: str | None = None, device_id: str | None = None, limit: int = 20,
    ) -> list[MemoryRecord]: ...

    @abc.abstractmethod
    async def list(
        self, *, kind: str | None = None, run_id: str | None = None, limit: int = 50,
    ) -> list[MemoryRecord]: ...

    @abc.abstractmethod
    async def delete(self, memory_id: str) -> None: ...

    async def close(self) -> None:  # pragma: no cover - most backends need nothing
        return None
