"""Long-running operation manager.

The hard mismatch between MCP and instruments is duration. A tool call is a
request/response; a tile scan is forty minutes. Blocking the call loses the
work to the first timeout and gives the agent no way to watch, steer or stop it.

So every observable command returns a *handle* immediately, and the agent polls
or subscribes. This is the same submit→poll→cancel shape Lightfall uses for
synchrotron scans and that the forthcoming MCP Tasks extension formalises
(`tasks/get`, `tasks/cancel`); when the SDK ships Tasks, this manager becomes
its backing store rather than being replaced.
"""

from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .errors import Cancelled, JobNotFound, LabBenchError


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


class ProgressUpdate(BaseModel):
    fraction: float
    message: str = ""
    timestamp: float = Field(default_factory=time.time)


class Artifact(BaseModel):
    """A file the job produced. Kept out of the tool result on purpose.

    Returning a 40 MB image inline would blow the context window; the agent
    gets a URI plus enough metadata to decide whether it wants the bytes.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    uri: str
    kind: str = "image"          # image | table | trace | log | report
    mime_type: str = "application/octet-stream"
    bytes: int | None = None
    shape: list[int] | None = None
    dtype: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created: float = Field(default_factory=time.time)


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    device_id: str | None = None
    feature: str | None = None
    command: str | None = None
    run_id: str | None = None
    actor: str = "agent"
    status: JobStatus = JobStatus.PENDING
    created: float = Field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    progress: float = 0.0
    message: str = ""
    #: Bounded history so a long job cannot grow without limit.
    history: list[ProgressUpdate] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    estimated_duration_s: float | None = None

    @property
    def elapsed_s(self) -> float:
        if self.started is None:
            return 0.0
        return (self.finished or time.time()) - self.started

    def eta_s(self) -> float | None:
        if self.status is not JobStatus.RUNNING or self.progress <= 0.01:
            return self.estimated_duration_s
        return self.elapsed_s * (1.0 - self.progress) / self.progress

    def summary(self) -> dict[str, Any]:
        """Compact form for tool results — no history, no full artifact list."""
        out: dict[str, Any] = {
            "job_id": self.id,
            "label": self.label,
            "status": self.status.value,
            "progress": round(self.progress, 4),
            "message": self.message,
            "elapsed_s": round(self.elapsed_s, 2),
        }
        eta = self.eta_s()
        if eta is not None and not self.status.terminal:
            out["eta_s"] = round(eta, 1)
        if self.device_id:
            out["device"] = self.device_id
        if self.artifacts:
            out["artifact_count"] = len(self.artifacts)
        if self.status is JobStatus.SUCCEEDED and self.result is not None:
            out["result"] = self.result
        if self.error:
            out["error"] = self.error
        return out


JobFn = Callable[[Any], Awaitable[Any]]  # receives ExecutionContext


class JobManager:
    """Owns the asyncio tasks behind observable commands."""

    #: Keep completed jobs this long so an agent that reconnects can still read them.
    retention_s: float = 3600.0
    max_history: int = 200

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._cancels: dict[str, asyncio.Event] = {}
        self._watchers: list[Callable[[Job], Awaitable[None] | None]] = []

    def watch(self, fn: Callable[[Job], Awaitable[None] | None]) -> None:
        self._watchers.append(fn)

    async def _notify(self, job: Job) -> None:
        for fn in list(self._watchers):
            res = fn(job)
            if asyncio.iscoroutine(res):
                await res

    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFound(
                f"no job {job_id!r}; it may have expired "
                f"(retention {self.retention_s:.0f}s)",
                job_id=job_id, known=sorted(self._jobs)[-10:],
            )
        return job

    def list(
        self, *, status: JobStatus | None = None, device_id: str | None = None,
        run_id: str | None = None, limit: int = 50,
    ) -> list[Job]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)
        if status is not None:
            jobs = [j for j in jobs if j.status is status]
        if device_id is not None:
            jobs = [j for j in jobs if j.device_id == device_id]
        if run_id is not None:
            jobs = [j for j in jobs if j.run_id == run_id]
        return jobs[:limit]

    def submit(
        self,
        fn: JobFn,
        *,
        label: str,
        context_factory: Callable[[str, asyncio.Event, Callable], Any],
        device_id: str | None = None,
        feature: str | None = None,
        command: str | None = None,
        run_id: str | None = None,
        actor: str = "agent",
        estimated_duration_s: float | None = None,
    ) -> Job:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = Job(
            id=job_id, label=label, device_id=device_id, feature=feature,
            command=command, run_id=run_id, actor=actor,
            estimated_duration_s=estimated_duration_s,
        )
        self._jobs[job_id] = job
        cancel = asyncio.Event()
        self._cancels[job_id] = cancel

        async def report(fraction: float, message: str = "") -> None:
            job.progress = fraction
            if message:
                job.message = message
            job.history.append(ProgressUpdate(fraction=fraction, message=message))
            if len(job.history) > self.max_history:
                # Thin the middle, keep the shape of the curve and both ends.
                job.history = job.history[:1] + job.history[-(self.max_history - 1):]
            await self._notify(job)

        ctx = context_factory(job_id, cancel, report)

        async def runner() -> None:
            job.status = JobStatus.RUNNING
            job.started = time.time()
            await self._notify(job)
            try:
                result = await fn(ctx)
                if isinstance(result, dict):
                    job.artifacts.extend(
                        Artifact(**a) if isinstance(a, dict) else a
                        for a in result.pop("artifacts", []) or []
                    )
                    job.result = result
                else:
                    job.result = {"value": result}
                job.status = JobStatus.SUCCEEDED
                job.progress = 1.0
                job.message = job.message or "completed"
            except (Cancelled, asyncio.CancelledError):
                job.status = JobStatus.CANCELLED
                job.message = "cancelled"
            except LabBenchError as exc:
                job.status = JobStatus.FAILED
                job.error = exc.to_dict()
                job.message = exc.message
            except Exception as exc:  # noqa: BLE001 - surface driver bugs verbatim
                job.status = JobStatus.FAILED
                job.error = {
                    "error": "driver_exception",
                    "message": str(exc),
                    "recovery": "human_required",
                    "state_uncertain": True,
                    "detail": {"traceback": traceback.format_exc(limit=8)},
                }
                job.message = str(exc)
            finally:
                job.finished = time.time()
                self._tasks.pop(job_id, None)
                self._cancels.pop(job_id, None)
                await self._notify(job)
                self._reap()

        self._tasks[job_id] = asyncio.create_task(runner(), name=f"labbench:{label}")
        return job

    async def cancel(self, job_id: str, *, reason: str = "agent requested") -> Job:
        job = self.get(job_id)
        if job.status.terminal:
            return job
        job.message = f"cancelling: {reason}"
        event = self._cancels.get(job_id)
        if event is not None:
            event.set()  # cooperative first: lets the driver park hardware safely
        await self._notify(job)
        return job

    async def kill(self, job_id: str) -> Job:
        """Hard cancel. Only for a driver that ignored cooperative cancellation."""
        job = self.get(job_id)
        task = self._tasks.get(job_id)
        if task is not None:
            task.cancel()
        return job

    async def wait(self, job_id: str, timeout: float | None = None) -> Job:
        task = self._tasks.get(job_id)
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout)
            except TimeoutError:
                pass
        return self.get(job_id)

    async def shutdown(self) -> None:
        for job_id in list(self._cancels):
            await self.cancel(job_id, reason="server shutting down")
        tasks = list(self._tasks.values())
        if tasks:
            await asyncio.wait(tasks, timeout=10)
        for task in tasks:
            if not task.done():
                task.cancel()

    def _reap(self) -> None:
        cutoff = time.time() - self.retention_s
        for jid, job in list(self._jobs.items()):
            if job.status.terminal and (job.finished or 0) < cutoff:
                del self._jobs[jid]
