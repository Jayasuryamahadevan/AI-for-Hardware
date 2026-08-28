"""Durable notes and documents an agent can search.

Plugin shape mirrors `core.registry.DriverRegistry` deliberately: a backend is
discovered through the ``labbench.memory`` entry-point group, so a third party
ships one as an ordinary pip package with no edit to LabBench, exactly like a
driver. The logic is kept local to this package rather than shared with
`DriverRegistry` because `core/` must stay importable with nothing installed
at all (see the package-level docstring in `labbench/__init__.py`) and must
not, therefore, know that `memory/` exists.

A lab that configures no memory backend still gets one: `MemoryManager`
defaults to a single `SqliteMemory` under the lab's data directory, because a
durable place to write things down is infrastructure an agent should never
have to ask an operator to set up first.
"""

from __future__ import annotations

import importlib.metadata as md
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import DriverUnavailable
from .store import MemoryNotFound, MemoryRecord, MemoryStore, score, tokenise

ENTRY_POINT_GROUP = "labbench.memory"

__all__ = [
    "MemoryBackendRegistry",
    "MemoryConfig",
    "MemoryManager",
    "MemoryNotFound",
    "MemoryRecord",
    "MemoryStore",
    "score",
    "tokenise",
]


class MemoryConfig(BaseModel):
    """One backend in a lab configuration's `memory:` block."""

    model_config = ConfigDict(extra="forbid")

    id: str = "default"
    backend: str = "sqlite"
    settings: dict[str, Any] = Field(default_factory=dict)


class MemoryBackendRegistry:
    """Maps backend names to `MemoryStore` subclasses."""

    def __init__(self) -> None:
        self._backends: dict[str, type[MemoryStore]] = {}
        self._failed: dict[str, str] = {}
        self._loaded = False

    def discover(self, *, force: bool = False) -> dict[str, type[MemoryStore]]:
        if self._loaded and not force:
            return self._backends
        for ep in md.entry_points(group=ENTRY_POINT_GROUP):
            try:
                cls = ep.load()
            except Exception as exc:  # noqa: BLE001 - optional dependency missing, usually
                self._failed[ep.name] = f"{type(exc).__name__}: {exc}"
                continue
            if not (isinstance(cls, type) and issubclass(cls, MemoryStore)):
                self._failed[ep.name] = "entry point is not a MemoryStore subclass"
                continue
            self._backends[ep.name] = cls
        self._loaded = True
        return self._backends

    def get(self, name: str) -> type[MemoryStore]:
        self.discover()
        if name in self._backends:
            return self._backends[name]
        if name in self._failed:
            raise DriverUnavailable(
                f"memory backend {name!r} is installed but failed to load: "
                f"{self._failed[name]}",
                driver=name, cause=self._failed[name],
            )
        raise DriverUnavailable(
            f"unknown memory backend {name!r}; available: {sorted(self._backends)}",
            driver=name, available=sorted(self._backends), unavailable=self._failed,
        )

    def catalog(self) -> dict[str, Any]:
        self.discover()
        return {"available": sorted(self._backends), "unavailable": dict(self._failed)}


class MemoryManager:
    """Owns every configured memory store and answers to the first by default.

    Most labs need exactly one memory space, so every method takes an
    optional `store` id and falls back to whichever was configured first (or
    the implicit default). A lab that genuinely wants two -- a shared
    filesystem SOP library plus a private per-project SQLite scratchpad --
    names them and an agent addresses either by id, the same shape
    `device.invoke` uses for instruments.
    """

    def __init__(
        self, configs: list[MemoryConfig] | None = None, *, data_dir: Path | str = ".",
        registry: MemoryBackendRegistry | None = None,
    ) -> None:
        self.registry = registry or MemoryBackendRegistry()
        self._stores: dict[str, MemoryStore] = {}
        self._default_id: str | None = None
        data_dir = Path(data_dir)
        configs = configs or [MemoryConfig(id="default", backend="sqlite",
                                            settings={"path": str(data_dir / "memory.sqlite")})]
        for cfg in configs:
            cls = self.registry.get(cfg.backend)
            self._stores[cfg.id] = cls(**cfg.settings)
            if self._default_id is None:
                self._default_id = cfg.id

    def store(self, name: str | None = None) -> MemoryStore:
        key = name or self._default_id
        if key not in self._stores:
            raise DriverUnavailable(
                f"no memory store {name!r} configured; available: {sorted(self._stores)}",
                available=sorted(self._stores),
            )
        return self._stores[key]

    def ids(self) -> list[str]:
        return sorted(self._stores)

    async def close(self) -> None:
        for store in self._stores.values():
            await store.close()
