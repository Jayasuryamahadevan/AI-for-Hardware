"""Memory backends: SqliteMemory, FilesystemDocs, and the manager over both."""

from __future__ import annotations

import pytest

from labbench.memory import MemoryBackendRegistry, MemoryConfig, MemoryManager
from labbench.memory.filesystem import FilesystemDocs
from labbench.memory.sqlite_store import SqliteMemory
from labbench.memory.store import MemoryNotFound, MemoryRecord, score, tokenise


def backend_params():
    return [SqliteMemory, FilesystemDocs]


@pytest.fixture(params=backend_params(), ids=["sqlite", "filesystem"])
def store(request, tmp_path):
    if request.param is SqliteMemory:
        return request.param(path=str(tmp_path / "memory.sqlite"))
    return request.param(path=str(tmp_path / "memory"))


class TestBothBackends:
    """Every backend must satisfy the same contract; parametrised over both."""

    async def test_write_then_get(self, store):
        record = await store.write(MemoryRecord(title="Focus offset", content="Add 3um for 40x."))
        fetched = await store.get(record.id)
        assert fetched.content == "Add 3um for 40x."

    async def test_get_unknown_raises(self, store):
        with pytest.raises(MemoryNotFound):
            await store.get("mem_does_not_exist")

    async def test_delete(self, store):
        record = await store.write(MemoryRecord(content="temporary"))
        await store.delete(record.id)
        with pytest.raises(MemoryNotFound):
            await store.get(record.id)

    async def test_delete_unknown_raises(self, store):
        with pytest.raises(MemoryNotFound):
            await store.delete("mem_nope")

    async def test_search_ranks_by_term_overlap(self, store):
        await store.write(MemoryRecord(title="A", content="the objective needs cleaning weekly"))
        await store.write(MemoryRecord(title="B", content="unrelated buffer preparation notes"))
        results = await store.search("objective cleaning")
        assert results[0].title == "A"

    async def test_search_empty_query_returns_everything(self, store):
        await store.write(MemoryRecord(title="A"))
        await store.write(MemoryRecord(title="B"))
        results = await store.search("")
        assert {r.title for r in results} == {"A", "B"}

    async def test_search_filters_by_kind(self, store):
        await store.write(MemoryRecord(title="sop", kind="sop", content="how to clean the stage"))
        await store.write(MemoryRecord(title="note", kind="note", content="how to clean the stage"))
        results = await store.search("clean", kind="sop")
        assert [r.title for r in results] == ["sop"]

    async def test_search_filters_by_tags(self, store):
        await store.write(MemoryRecord(title="tagged", tags=["urgent", "scope1"]))
        await store.write(MemoryRecord(title="untagged"))
        results = await store.search("", tags=["urgent"])
        assert [r.title for r in results] == ["tagged"]

    async def test_search_filters_by_run_id(self, store):
        await store.write(MemoryRecord(title="run-a", run_id="run_a"))
        await store.write(MemoryRecord(title="run-b", run_id="run_b"))
        results = await store.search("", run_id="run_a")
        assert [r.title for r in results] == ["run-a"]

    async def test_list_orders_newest_first(self, store):

        first = await store.write(MemoryRecord(title="old"))
        first.updated -= 10  # force a clear ordering without a real sleep
        await store.write(first)
        await store.write(MemoryRecord(title="new"))
        results = await store.list()
        assert results[0].title == "new"

    async def test_write_is_idempotent_by_id(self, store):
        record = await store.write(MemoryRecord(title="v1", content="first"))
        record.content = "second"
        await store.write(record)
        fetched = await store.get(record.id)
        assert fetched.content == "second"
        assert len(await store.list()) == 1


class TestFilesystemSpecific:
    async def test_a_hand_written_markdown_file_is_readable(self, tmp_path):
        docs = FilesystemDocs(path=str(tmp_path))
        (tmp_path / "hand_written.md").write_text("# just some notes\nno front matter here")
        records = await docs.list()
        assert len(records) == 1
        assert "no front matter" in records[0].content


class TestScoring:
    def test_tokenise_lowercases_and_splits(self):
        assert tokenise("Objective 40X!") == ["objective", "40x"]

    def test_score_zero_when_nothing_matches(self):
        assert score(["banana"], "completely different text") == 0.0

    def test_score_positive_when_something_matches(self):
        assert score(["objective"], "the objective is dirty") > 0.0


class TestMemoryManager:
    async def test_defaults_to_one_sqlite_store(self, tmp_path):
        manager = MemoryManager(data_dir=tmp_path)
        assert manager.ids() == ["default"]
        await manager.store().write(MemoryRecord(title="x"))
        assert (tmp_path / "memory.sqlite").exists()
        await manager.close()

    async def test_multiple_named_stores(self, tmp_path):
        manager = MemoryManager(
            [
                MemoryConfig(id="notes", backend="sqlite", settings={"path": str(tmp_path / "a.sqlite")}),
                MemoryConfig(id="docs", backend="filesystem", settings={"path": str(tmp_path / "docs")}),
            ],
            data_dir=tmp_path,
        )
        await manager.store("notes").write(MemoryRecord(title="in sqlite"))
        await manager.store("docs").write(MemoryRecord(title="in filesystem"))
        assert set(manager.ids()) == {"notes", "docs"}
        await manager.close()

    def test_unknown_store_raises(self, tmp_path):
        from labbench.core.errors import DriverUnavailable

        manager = MemoryManager(data_dir=tmp_path)
        with pytest.raises(DriverUnavailable):
            manager.store("nope")


class TestBackendRegistry:
    def test_discovers_both_shipped_backends(self):
        registry = MemoryBackendRegistry()
        catalog = registry.catalog()
        assert {"sqlite", "filesystem"}.issubset(set(catalog["available"]))
