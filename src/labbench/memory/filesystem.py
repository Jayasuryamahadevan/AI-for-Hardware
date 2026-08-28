"""Filesystem-backed memory: the plain-text half of the memory story.

Each record is one Markdown file with a YAML front-matter block, the format
`ledger.py` already uses for its own reasoning: "for the auditor who has none
of this software." A human can open the directory in any editor, read what an
agent learned, correct it, or drop in an SOP by hand — and the agent's own
`memory.search` sees the edit on the next call, because the file *is* the
record, not a cache of it.

Chosen over SQLite when a lab wants its memory under version control:
`git diff` on a directory of Markdown files is legible in a way a diff of a
SQLite file never is.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from .store import MemoryNotFound, MemoryRecord, MemoryStore, score, tokenise

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "note"


class FilesystemDocs(MemoryStore):
    """One Markdown file per record under `settings.path`.

    Configuration:

        memory:
          - id: docs
            backend: filesystem
            settings:
              path: ./labbench-data/memory   # default, if omitted
    """

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)
        self.root = Path(config.get("path", "./labbench-data/memory")).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, record: MemoryRecord) -> Path:
        return self.root / f"{record.id}_{_slug(record.title or record.kind)}.md"

    def _find_path(self, memory_id: str) -> Path | None:
        matches = list(self.root.glob(f"{memory_id}_*.md")) or list(
            self.root.glob(f"{memory_id}.md")
        )
        return matches[0] if matches else None

    # -- (de)serialisation ----------------------------------------------------

    @staticmethod
    def _render(record: MemoryRecord) -> str:
        front = {
            "id": record.id, "kind": record.kind, "title": record.title,
            "tags": record.tags, "metadata": record.metadata, "actor": record.actor,
            "device_id": record.device_id, "run_id": record.run_id,
            "created": record.created, "updated": record.updated,
        }
        return f"---\n{yaml.safe_dump(front, sort_keys=False)}---\n\n{record.content}\n"

    @staticmethod
    def _parse(text: str, *, fallback_id: str) -> MemoryRecord:
        match = _FRONT_MATTER.match(text)
        if not match:
            # A file dropped in by hand with no front matter is still a
            # legitimate memory: the whole point of this backend is that a
            # human can add to it without learning the schema first.
            return MemoryRecord(id=fallback_id, title=fallback_id, content=text.strip())
        front = yaml.safe_load(match.group(1)) or {}
        body = match.group(2).strip()
        return MemoryRecord(
            id=front.get("id", fallback_id), kind=front.get("kind", "note"),
            title=front.get("title", ""), content=body, tags=front.get("tags") or [],
            metadata=front.get("metadata") or {}, actor=front.get("actor", "agent"),
            device_id=front.get("device_id"), run_id=front.get("run_id"),
            created=front.get("created", time.time()), updated=front.get("updated", time.time()),
        )

    # -- MemoryStore ------------------------------------------------------

    async def write(self, record: MemoryRecord) -> MemoryRecord:
        old = self._find_path(record.id)
        if old is not None and old != self._path_for(record):
            old.unlink(missing_ok=True)
        self._path_for(record).write_text(self._render(record), encoding="utf-8")
        return record

    async def get(self, memory_id: str) -> MemoryRecord:
        path = self._find_path(memory_id)
        if path is None:
            raise MemoryNotFound(f"no memory {memory_id!r}", memory_id=memory_id)
        return self._parse(path.read_text(encoding="utf-8"), fallback_id=memory_id)

    async def delete(self, memory_id: str) -> None:
        path = self._find_path(memory_id)
        if path is None:
            raise MemoryNotFound(f"no memory {memory_id!r}", memory_id=memory_id)
        path.unlink()

    def _all(self) -> list[MemoryRecord]:
        records = []
        for path in sorted(self.root.glob("*.md")):
            fallback = path.stem.split("_", 1)[0] or f"mem_{uuid.uuid4().hex[:12]}"
            try:
                records.append(self._parse(path.read_text(encoding="utf-8"), fallback_id=fallback))
            except (yaml.YAMLError, OSError):
                continue  # a file a human is mid-edit must not break search for everyone else
        return records

    async def list(
        self, *, kind: str | None = None, run_id: str | None = None, limit: int = 50,
    ) -> list[MemoryRecord]:
        records = self._all()
        if kind is not None:
            records = [r for r in records if r.kind == kind]
        if run_id is not None:
            records = [r for r in records if r.run_id == run_id]
        records.sort(key=lambda r: r.updated, reverse=True)
        return records[:limit]

    async def search(
        self, query: str = "", *, kind: str | None = None, tags: list[str] | None = None,
        run_id: str | None = None, device_id: str | None = None, limit: int = 20,
    ) -> list[MemoryRecord]:
        records = self._all()
        if kind is not None:
            records = [r for r in records if r.kind == kind]
        if run_id is not None:
            records = [r for r in records if r.run_id == run_id]
        if device_id is not None:
            records = [r for r in records if r.device_id == device_id]
        if tags:
            wanted = set(tags)
            records = [r for r in records if wanted.issubset(r.tags)]

        terms = tokenise(query)
        scored = [(score(terms, f"{r.title}\n{r.content}"), r) for r in records]
        scored = [(s, r) for s, r in scored if s > 0 or not query]
        scored.sort(key=lambda pair: (pair[0], pair[1].updated), reverse=True)
        return [r for _, r in scored[:limit]]
