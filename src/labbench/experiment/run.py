"""Experiment runs: a `Protocol` executed through the `Gateway`, end to end.

A run is not a new path to hardware. Every step still calls
`Gateway.invoke`, which means it still crosses the ledger, the safety kernel
and the approval broker exactly as if an agent had typed the call itself --
`ExperimentManager` is a disciplined caller of that same front door, not a
side door around it.

The one thing a run adds that a bare loop of tool calls could not: when a step
needs a human signature, the *run* parks rather than the coroutine blocking on
it. An agent driving a protocol synchronously would either freeze waiting for
a signature that might not come for hours, or would have to reimplement this
same park-and-resume logic itself. `ExperimentManager` does it once.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import ApprovalRequired, Cancelled, LabBenchError
from .protocol import Protocol, ProtocolStep

log = logging.getLogger("labbench.experiment")

InvokeFn = Callable[..., Awaitable[dict[str, Any]]]


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED)


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    index: int
    device: str
    feature: str
    command: str
    status: str = "pending"
    attempt: int = 0
    started: float | None = None
    finished: float | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    #: Set once a step has asked for a human signature, so a resume can reuse it.
    approval_id: str | None = None


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    protocol_name: str
    protocol_id: str | None = None
    actor: str = "agent"
    status: RunStatus = RunStatus.PENDING
    current_step: int = 0
    variables: dict[str, Any] = Field(default_factory=dict)
    results: list[StepResult] = Field(default_factory=list)
    created: float = Field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    message: str = ""

    @property
    def elapsed_s(self) -> float:
        if self.started is None:
            return 0.0
        return (self.finished or time.time()) - self.started

    def summary(self, *, include_steps: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "run_id": self.id, "protocol": self.protocol_name, "status": self.status.value,
            "current_step": self.current_step, "total_steps": len(self.results),
            "elapsed_s": round(self.elapsed_s, 2), "message": self.message,
        }
        if include_steps:
            out["steps"] = [r.model_dump(mode="json") for r in self.results]
        return out


class ExperimentManager:
    """Owns every run, the way `JobManager` owns every job.

    Kept separate from `JobManager` rather than reusing it: a run's unit of
    work spans many devices and many individual jobs, and an approval parks a
    run in place rather than failing it -- a state `JobManager`'s
    terminal-on-failure model does not have a slot for.
    """

    retention_s: float = 3600.0

    def __init__(
        self, *, invoke: InvokeFn, ledger: Any | None = None,
    ) -> None:
        self._invoke = invoke
        self._ledger = ledger
        self._protocols: dict[str, Protocol] = {}
        self._runs: dict[str, Run] = {}
        self._results_by_label: dict[str, dict[str, dict[str, Any]]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancels: dict[str, asyncio.Event] = {}
        self._watchers: list[Callable[[Run], Awaitable[None] | None]] = []

    def watch(self, fn: Callable[[Run], Awaitable[None] | None]) -> None:
        self._watchers.append(fn)

    async def _notify(self, run: Run) -> None:
        for fn in list(self._watchers):
            res = fn(run)
            if asyncio.iscoroutine(res):
                await res

    def _log(self, kind: str, run: Run, **fields: Any) -> None:
        if self._ledger is not None:
            self._ledger.log(kind, actor=run.actor, run_id=run.id, **fields)

    # -- protocol registry --------------------------------------------------

    def define(self, protocol: Protocol) -> str:
        protocol_id = f"proto_{uuid.uuid4().hex[:12]}"
        self._protocols[protocol_id] = protocol
        return protocol_id

    def get_protocol(self, protocol_id: str) -> Protocol:
        from ..core.errors import ValidationError

        if protocol_id not in self._protocols:
            raise ValidationError(
                f"no protocol {protocol_id!r}; call experiment.define first",
                available=sorted(self._protocols),
            )
        return self._protocols[protocol_id]

    # -- lifecycle ------------------------------------------------------------

    def get(self, run_id: str) -> Run:
        from ..core.errors import JobNotFound

        if run_id not in self._runs:
            raise JobNotFound(f"no run {run_id!r}", run_id=run_id, known=sorted(self._runs)[-10:])
        return self._runs[run_id]

    def list(self, *, status: RunStatus | None = None, limit: int = 50) -> list[Run]:
        runs = sorted(self._runs.values(), key=lambda r: r.created, reverse=True)
        if status is not None:
            runs = [r for r in runs if r.status is status]
        return runs[:limit]

    def start(
        self, protocol: Protocol, *, protocol_id: str | None = None,
        variables: dict[str, Any] | None = None, actor: str = "agent",
    ) -> Run:
        run = Run(
            protocol_name=protocol.name, protocol_id=protocol_id, actor=actor,
            variables={**protocol.variables, **(variables or {})},
            results=[
                StepResult(
                    label=step.resolved_label(i), index=i, device=step.device,
                    feature=step.feature, command=step.command,
                )
                for i, step in enumerate(protocol.steps)
            ],
        )
        self._runs[run.id] = run
        self._results_by_label[run.id] = {}
        self._cancels[run.id] = asyncio.Event()
        self._log("run_start", run, payload={
            "protocol": protocol.name, "steps": len(protocol.steps), "variables": run.variables,
        })
        task = asyncio.create_task(
            self._drive(run, protocol), name=f"labbench:experiment:{run.id}"
        )
        self._tasks[run.id] = task
        return run

    def resume(self, run_id: str, protocol: Protocol) -> Run:
        """Continue a run that parked on `AWAITING_APPROVAL`.

        Call this after the human has answered via `approval.grant` /
        `approval.deny`. The same `Protocol` must be supplied: a run holds no
        reference to it so that a restarted gateway can still report a run's
        history without keeping every protocol it ever executed in memory.
        """
        run = self.get(run_id)
        if run.status is not RunStatus.AWAITING_APPROVAL:
            from ..core.errors import ValidationError

            raise ValidationError(
                f"run {run_id!r} is {run.status.value}, not awaiting approval",
                run_id=run_id, status=run.status.value,
            )
        self._cancels[run_id] = asyncio.Event()
        task = asyncio.create_task(
            self._drive(run, protocol), name=f"labbench:experiment:{run_id}"
        )
        self._tasks[run_id] = task
        return run

    async def cancel(self, run_id: str, *, reason: str = "agent requested") -> Run:
        run = self.get(run_id)
        if run.status.terminal:
            return run
        event = self._cancels.get(run_id)
        if event is not None:
            event.set()  # cooperative: let the in-flight step finish or park
        if run.status in (RunStatus.PENDING, RunStatus.AWAITING_APPROVAL):
            # No task is currently driving this run to observe the event.
            run.status = RunStatus.CANCELLED
            run.message = f"cancelled: {reason}"
            run.finished = time.time()
            self._log("run_end", run, payload={"status": "cancelled", "reason": reason})
            await self._notify(run)
        return run

    # -- execution --------------------------------------------------------

    async def _drive(self, run: Run, protocol: Protocol) -> None:
        run.status = RunStatus.RUNNING
        run.started = run.started or time.time()
        await self._notify(run)
        cancel = self._cancels[run.id]
        results = self._results_by_label[run.id]

        try:
            for index in range(run.current_step, len(protocol.steps)):
                if cancel.is_set():
                    run.status = RunStatus.CANCELLED
                    run.message = "cancelled"
                    break
                run.current_step = index
                outcome = await self._run_step(run, protocol, index, results, cancel)
                if outcome == "parked":
                    return  # AWAITING_APPROVAL; a later resume() re-enters here
                if outcome == "stopped":
                    break
            else:
                run.status = RunStatus.SUCCEEDED
                run.message = "completed"
        except asyncio.CancelledError:
            run.status = RunStatus.CANCELLED
            run.message = "cancelled"
        finally:
            if run.status.terminal:
                run.finished = time.time()
                self._log("run_end", run, payload={
                    "status": run.status.value, "elapsed_s": round(run.elapsed_s, 3),
                })
                self._tasks.pop(run.id, None)
            await self._notify(run)

    async def _run_step(
        self, run: Run, protocol: Protocol, index: int,
        results: dict[str, dict[str, Any]], cancel: asyncio.Event,
    ) -> str:
        """Returns 'continue', 'parked', or 'stopped'.

        Re-entrant: called once when a step first runs and again, for the
        same step, when `resume()` re-drives a run parked on
        `AWAITING_APPROVAL`. Parking is not a failed attempt -- the call was
        never wrong, only unauthorised -- so it must not consume a `repeat`
        slot; `attempt` is only advanced past a genuine `LabBenchError`.
        """
        step = protocol.steps[index]
        sr = run.results[index]
        resuming = sr.status == "awaiting_approval"
        try:
            args = protocol.resolve_args(step, results, run.variables)
        except LabBenchError as exc:
            sr.status, sr.error = "failed", exc.to_dict()
            return await self._settle_failure(run, step, sr, exc.message)

        max_attempts = max(1, step.repeat)
        attempt = sr.attempt if resuming else 1
        while True:
            if cancel.is_set():
                run.status = RunStatus.CANCELLED
                return "stopped"
            sr.attempt = attempt
            sr.status = "running"
            sr.started = sr.started or time.time()
            self._log("run_step", run, device_id=step.device, feature=step.feature,
                      command=step.command, payload={"label": sr.label, "attempt": attempt})
            await self._notify(run)
            try:
                outcome = await self._invoke(
                    step.device, step.feature, step.command, args,
                    actor=run.actor, reason=step.reason or f"{protocol.name}: {sr.label}",
                    run_id=run.id, approval_id=sr.approval_id,
                    wait_for_approval=step.wait_for_approval_s,
                )
            except ApprovalRequired as exc:
                sr.status = "awaiting_approval"
                sr.approval_id = exc.detail.get("approval_id")
                run.status = RunStatus.AWAITING_APPROVAL
                run.message = exc.message
                await self._notify(run)
                return "parked"
            except (Cancelled, asyncio.CancelledError):
                sr.status = "cancelled"
                run.status = RunStatus.CANCELLED
                return "stopped"
            except LabBenchError as exc:
                sr.status, sr.error = "failed", exc.to_dict()
                sr.finished = time.time()
                if attempt < max_attempts:
                    log.info("run %s step %s attempt %d/%d failed, retrying: %s",
                             run.id, sr.label, attempt, max_attempts, exc.message)
                    attempt += 1
                    continue
                return await self._settle_failure(run, step, sr, exc.message)
            else:
                sr.status, sr.result, sr.finished = "succeeded", outcome, time.time()
                results[sr.label] = {"result": outcome}
                await self._notify(run)
                return "continue"

    async def _settle_failure(self, run: Run, step: ProtocolStep, sr: StepResult, message: str) -> str:
        sr.finished = time.time()
        await self._notify(run)
        if step.continue_on_error:
            return "continue"
        run.status = RunStatus.FAILED
        run.message = f"step {sr.label!r} failed: {message}"
        return "stopped"

    async def shutdown(self) -> None:
        for run_id in list(self._cancels):
            await self.cancel(run_id, reason="server shutting down")
        tasks = list(self._tasks.values())
        if tasks:
            await asyncio.wait(tasks, timeout=10)
        for task in tasks:
            if not task.done():
                task.cancel()
