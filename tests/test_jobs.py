"""JobManager: submit, progress, cancel, wait, retention."""

from __future__ import annotations

import asyncio

import pytest

from labbench.core.errors import Cancelled, ValidationError
from labbench.core.jobs import JobManager, JobStatus


def _ctx_factory(job_id, cancel, report):
    from labbench.core.device import ExecutionContext

    return ExecutionContext(job_id=job_id, cancel_event=cancel).with_progress(report)


@pytest.fixture
def manager():
    return JobManager()


class TestSubmit:
    async def test_success_path(self, manager):
        async def work(ctx):
            await ctx.progress(0.5, "halfway")
            return {"answer": 42}

        job = manager.submit(work, label="test", context_factory=_ctx_factory)
        finished = await manager.wait(job.id, timeout=2)
        assert finished.status is JobStatus.SUCCEEDED
        assert finished.result == {"answer": 42}
        assert finished.progress == 1.0

    async def test_failure_path_carries_structured_error(self, manager):
        async def work(ctx):
            raise ValidationError("bad args", parameter="x")

        job = manager.submit(work, label="test", context_factory=_ctx_factory)
        finished = await manager.wait(job.id, timeout=2)
        assert finished.status is JobStatus.FAILED
        assert finished.error["error"] == "validation_error"
        assert finished.error["detail"]["parameter"] == "x"

    async def test_unexpected_exception_marks_state_uncertain(self, manager):
        async def work(ctx):
            raise RuntimeError("driver bug")

        job = manager.submit(work, label="test", context_factory=_ctx_factory)
        finished = await manager.wait(job.id, timeout=2)
        assert finished.status is JobStatus.FAILED
        assert finished.error["state_uncertain"] is True
        assert finished.error["recovery"] == "human_required"

    async def test_artifacts_are_pulled_out_of_the_result(self, manager):
        async def work(ctx):
            return {"value": 1, "artifacts": [{"uri": "file:///a.png"}]}

        job = manager.submit(work, label="test", context_factory=_ctx_factory)
        finished = await manager.wait(job.id, timeout=2)
        assert finished.result == {"value": 1}
        assert len(finished.artifacts) == 1
        assert finished.artifacts[0].uri == "file:///a.png"


class TestCancellation:
    async def test_cooperative_cancel(self, manager):
        started = asyncio.Event()

        async def work(ctx):
            started.set()
            while not ctx.cancelled:
                await asyncio.sleep(0.01)
            raise Cancelled("stopped")

        job = manager.submit(work, label="test", context_factory=_ctx_factory)
        await started.wait()
        await manager.cancel(job.id, reason="test")
        finished = await manager.wait(job.id, timeout=2)
        assert finished.status is JobStatus.CANCELLED

    async def test_kill_force_cancels_an_uncooperative_job(self, manager):
        started = asyncio.Event()

        async def work(ctx):
            started.set()
            await asyncio.sleep(30)  # ignores cancel_event on purpose

        job = manager.submit(work, label="test", context_factory=_ctx_factory)
        await started.wait()
        await manager.kill(job.id)
        finished = await manager.wait(job.id, timeout=2)
        assert finished.status is JobStatus.CANCELLED

    async def test_cancel_after_terminal_is_a_no_op(self, manager):
        async def work(ctx):
            return {}

        job = manager.submit(work, label="test", context_factory=_ctx_factory)
        await manager.wait(job.id, timeout=2)
        result = await manager.cancel(job.id)
        assert result.status is JobStatus.SUCCEEDED


class TestQuerying:
    async def test_get_unknown_job_raises(self, manager):
        from labbench.core.errors import JobNotFound

        with pytest.raises(JobNotFound):
            manager.get("job_does_not_exist")

    async def test_list_filters_by_status(self, manager):
        async def ok(ctx):
            return {}

        async def bad(ctx):
            raise ValidationError("no")

        j1 = manager.submit(ok, label="a", context_factory=_ctx_factory)
        j2 = manager.submit(bad, label="b", context_factory=_ctx_factory)
        await manager.wait(j1.id, timeout=2)
        await manager.wait(j2.id, timeout=2)
        succeeded = manager.list(status=JobStatus.SUCCEEDED)
        assert {j.id for j in succeeded} == {j1.id}

    async def test_shutdown_cancels_running_jobs(self, manager):
        async def work(ctx):
            while not ctx.cancelled:
                await asyncio.sleep(0.01)
            raise Cancelled("stopped")

        job = manager.submit(work, label="test", context_factory=_ctx_factory)
        await asyncio.sleep(0.05)
        await manager.shutdown()
        assert manager.get(job.id).status is JobStatus.CANCELLED
