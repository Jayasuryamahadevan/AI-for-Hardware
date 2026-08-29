"""The gateway: where the two ends of the cable meet.

Everything below this module is independent. `core` knows nothing about
transports, `protocol` knows nothing about instruments, `drivers` know nothing
about agents. This is the one place that wires them together, and it owns the
sequence that matters:

    request -> ledger -> safety kernel -> [approval] -> execute -> ledger

That order is the product. An agent cannot reach a driver without passing the
safety kernel, and it cannot pass the safety kernel without both the decision
and its outcome being written to an append-only ledger first. Not because
drivers are polite about it, but because there is no other path.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Self

from .bridge.approval import ApprovalBroker, ApprovalState
from .campaign import CampaignManager, CampaignState
from .core.device import Device, DeviceEvent, ExecutionContext
from .core.errors import (
    ApprovalDenied,
    ApprovalRequired,
    LabBenchError,
    SafetyViolation,
)
from .core.jobs import Job, JobManager
from .core.provenance import Ledger
from .core.registry import DeviceManager, DriverRegistry, LabConfig
from .core.safety import Decision, Effect, SafetyKernel, SafetyPolicy
from .experiment import ExperimentManager
from .memory import MemoryConfig, MemoryManager
from .protocol.router import Router

log = logging.getLogger("labbench.gateway")


class Gateway:
    """One laboratory, exposed to agents."""

    def __init__(
        self,
        config: LabConfig | None = None,
        *,
        registry: DriverRegistry | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        self.config = config or LabConfig()
        self.data_dir = Path(data_dir or self.config.data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = uuid.uuid4().hex[:12]
        self.devices = DeviceManager(registry)
        self.jobs = JobManager()
        self.ledger = Ledger(self.data_dir / "provenance.sqlite", session_id=self.session_id)
        self.safety = SafetyKernel(SafetyPolicy.model_validate(self.config.safety or {}))
        self.approvals = ApprovalBroker(
            broadcast=self._broadcast,
            on_decision=self._record_approval,
        )
        self.memory = MemoryManager(
            [MemoryConfig.model_validate(d) for d in self.config.memory] or None,
            data_dir=self.data_dir,
        )
        self.experiments = ExperimentManager(invoke=self.invoke, ledger=self.ledger)
        self.campaigns = CampaignManager(experiments=self.experiments, ledger=self.ledger)
        self.router = Router()
        self.started = time.time()
        #: Sinks that fan events out to connected transports.
        self._event_sinks: list[Any] = []
        self._closed = False

        self.devices.subscribe_all(self._on_device_event)
        self.jobs.watch(self._on_job_update)
        self.experiments.watch(self._on_experiment_update)
        self.campaigns.watch(self._on_campaign_update)

        from .bridge.toolset import register_tools

        register_tools(self.router, self)

    # -- event fan-out ----------------------------------------------------

    def add_event_sink(self, sink: Any) -> None:
        """Attach a transport's broadcast function. Signature: (topic, payload)."""
        self._event_sinks.append(sink)

    async def _broadcast(self, topic: str, payload: dict[str, Any]) -> None:
        for sink in list(self._event_sinks):
            try:
                result = sink(topic, payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # pragma: no cover - a dead subscriber must not
                log.debug("event sink failed", exc_info=True)  # break the lab

    async def _on_device_event(self, event: DeviceEvent) -> None:
        # Events at warning and above are part of the record, not just the feed.
        if event.severity in ("warning", "error", "critical"):
            self.ledger.log(
                "event", actor="device", device_id=event.device_id,
                feature=event.feature, command=event.name,
                payload={"severity": event.severity, **event.payload},
            )
        await self._broadcast("device.event", event.model_dump(mode="json"))

    async def _on_job_update(self, job: Job) -> None:
        await self._broadcast("job.update", job.summary())
        if job.status.terminal:
            self.ledger.log(
                "run_end" if job.run_id else "command_result",
                actor=job.actor, device_id=job.device_id, feature=job.feature,
                command=job.command, job_id=job.id, run_id=job.run_id,
                payload={
                    "status": job.status.value,
                    "elapsed_s": round(job.elapsed_s, 3),
                    "error": job.error,
                    "artifacts": [a.model_dump(mode="json") for a in job.artifacts],
                },
            )

    async def _on_experiment_update(self, run: Any) -> None:
        # run_start/run_step/run_end already land in the ledger from within
        # ExperimentManager itself, at the moment they happen rather than the
        # moment a watcher gets around to them; this is only the live feed.
        await self._broadcast("experiment.update", run.summary(include_steps=False))

    async def _on_campaign_update(self, state: CampaignState) -> None:
        # Same split as _on_experiment_update: CampaignManager already writes
        # campaign_start/campaign_trial_*/campaign_end to the ledger itself.
        await self._broadcast("campaign.update", state.summary(include_observations=False))

    def _record_approval(self, request: Any) -> None:
        self.ledger.log(
            "approval",
            actor=request.decided_by or "system",
            device_id=request.device, feature=request.feature, command=request.command,
            reason=request.decision_reason, session_id=request.session_id,
            run_id=request.run_id,
            payload={
                "approval_id": request.id, "state": request.state.value,
                "requested_by": request.actor, "intent": request.intent,
                "args": request.args, "digest": request.digest,
            },
        )

    # -- lifecycle --------------------------------------------------------

    async def start(self) -> dict[str, Any]:
        """Build devices from configuration and connect them."""
        self.ledger.log(
            "session_start", actor="system",
            payload={"lab": self.config.name, "devices": [d.id for d in self.config.devices],
                     "autonomy": int(self.safety.policy.autonomy),
                     "hazard_ceiling": self.safety.policy.ceiling().value},
        )
        problems: dict[str, str] = {}
        for device_config in self.config.devices:
            try:
                self.devices.add_from_config(device_config)
            except LabBenchError as exc:
                # A driver that will not load must not stop the rest of the lab
                # from coming up. It is reported, not fatal.
                problems[device_config.id] = exc.message
                log.warning("device %s unavailable: %s", device_config.id, exc.message)
        connected = await self.devices.connect_all()
        for device_id, state in connected.items():
            self.ledger.log("device_connect", actor="system", device_id=device_id,
                            payload={"result": state})
        return {"connected": connected, "unavailable": problems}

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.campaigns.shutdown()
        await self.experiments.shutdown()
        await self.jobs.shutdown()
        await self.devices.disconnect_all()
        await self.memory.close()
        self.ledger.log("session_end", actor="system",
                        payload={"uptime_s": round(time.time() - self.started, 1)})
        self.ledger.close()

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- the execution path -----------------------------------------------

    def device(self, device_id: str) -> Device:
        return self.devices.get(device_id)

    async def authorize(
        self,
        device: Device,
        feature: str,
        command: str,
        args: dict[str, Any],
        *,
        actor: str,
        reason: str = "",
        approval_id: str | None = None,
    ) -> Decision:
        """Run the three gates and record the verdict.

        An approval_id, when supplied, is verified against the exact call being
        made before it is allowed to satisfy the gate.
        """
        approver: str | None = None
        if approval_id:
            granted = self.approvals.verify(
                approval_id, device=device.id, feature=feature, command=command, args=args
            )
            approver = granted.decided_by

        decision = await self.safety.authorize(
            device, feature, command, args,
            actor=actor, approved_by=approver, reason=reason,
        )
        self.ledger.log(
            "safety_decision", actor=actor, device_id=device.id,
            feature=feature, command=command, reason=reason,
            payload=decision.model_dump(mode="json"),
        )
        await self._broadcast("safety.decision", decision.model_dump(mode="json"))
        return decision

    async def invoke(
        self,
        device_id: str,
        feature: str,
        command: str,
        args: dict[str, Any] | None = None,
        *,
        actor: str = "agent",
        reason: str = "",
        approval_id: str | None = None,
        run_id: str | None = None,
        wait_for_approval: float | None = None,
        progress: Any = None,
    ) -> dict[str, Any]:
        """The one path from an agent to an instrument.

        Returns either a result (for a short command) or a job handle (for an
        observable one). Raises `SafetyViolation` when a gate refuses, and
        `ApprovalRequired` carrying an approval id when a human must sign.
        """
        args = dict(args or {})
        device = self.device(device_id)
        _, spec = device.resolve(feature, command)

        self.ledger.log(
            "command_request", actor=actor, device_id=device_id,
            feature=feature, command=command, reason=reason, run_id=run_id,
            payload={"args": args, "hazard": spec.hazard.value,
                     "reversibility": spec.reversibility.value},
        )

        decision = await self.authorize(
            device, feature, command, args,
            actor=actor, reason=reason, approval_id=approval_id,
        )

        if decision.effect is Effect.DENY:
            raise SafetyViolation(
                f"blocked {device_id}.{feature}.{command}: " + "; ".join(decision.reasons),
                decision=decision.model_dump(mode="json"),
            )

        if decision.effect is Effect.REQUIRE_APPROVAL:
            request = await self.approvals.request(
                device=device_id, feature=feature, command=command, args=args,
                prompt=decision.approval_prompt, hazard=spec.hazard.value,
                reasons=decision.reasons, actor=actor, intent=reason,
                session_id=self.session_id, run_id=run_id,
            )
            if wait_for_approval:
                # Blocking on a human is legitimate for a scripted run, and a
                # terrible default for an agent loop, so it must be asked for.
                resolved = await self.approvals.wait(request.id, wait_for_approval)
                if resolved.state is not ApprovalState.GRANTED:
                    raise ApprovalDenied(
                        f"{feature}.{command} was {resolved.state.value}"
                        + (f": {resolved.decision_reason}" if resolved.decision_reason else ""),
                        approval_id=request.id, state=resolved.state.value,
                        decided_by=resolved.decided_by,
                    )
                return await self.invoke(
                    device_id, feature, command, args, actor=actor, reason=reason,
                    approval_id=request.id, run_id=run_id, progress=progress,
                )
            raise ApprovalRequired(
                decision.approval_prompt or f"{command} requires a human signature",
                approval_id=request.id,
                decision=decision.model_dump(mode="json"),
                next_step=(
                    f"a human must call approval.grant with approval_id={request.id!r}, "
                    f"then retry this call passing approval_id={request.id!r}"
                ),
            )

        if spec.observable:
            job = self._submit(device, feature, command, args, actor=actor, run_id=run_id)
            return {"job": job.summary(), "accepted": True,
                    "note": "long-running; poll job.status or subscribe to job.update"}

        context = ExecutionContext(actor=actor, run_id=run_id)
        if progress is not None:
            context = context.with_progress(progress)
        result = await device.invoke(feature, command, args, context)
        self.ledger.log(
            "command_result", actor=actor, device_id=device_id,
            feature=feature, command=command, run_id=run_id,
            payload={"result": _strip_ground_truth(result)},
        )
        return _strip_ground_truth(result)

    def _submit(
        self,
        device: Device,
        feature: str,
        command: str,
        args: dict[str, Any],
        *,
        actor: str,
        run_id: str | None,
    ) -> Job:
        _, spec = device.resolve(feature, command)

        def make_context(job_id: str, cancel: asyncio.Event, report: Any) -> ExecutionContext:
            context = ExecutionContext(
                job_id=job_id, run_id=run_id, actor=actor, cancel_event=cancel
            )
            return context.with_progress(report)

        async def run(context: ExecutionContext) -> Any:
            result = await device.invoke(feature, command, args, context)
            return _strip_ground_truth(result) if isinstance(result, dict) else result

        return self.jobs.submit(
            run,
            label=f"{device.id}.{feature}.{command}",
            context_factory=make_context,
            device_id=device.id, feature=feature, command=command,
            run_id=run_id, actor=actor,
            estimated_duration_s=spec.duration_estimate_s,
        )

    # -- emergency --------------------------------------------------------

    async def estop(self, reason: str, *, actor: str = "agent") -> dict[str, Any]:
        """Stop everything. Never gated, never queued, never refused.

        An e-stop that could be blocked by a policy rule, a rate limit or a
        pending approval would not be an e-stop.
        """
        self.ledger.log("estop", actor=actor, reason=reason, payload={"scope": "all"})
        await self._broadcast("estop", {"reason": reason, "actor": actor})
        jobs = [j.id for j in self.jobs.list() if not j.status.terminal]
        for job_id in jobs:
            await self.jobs.cancel(job_id, reason=f"e-stop: {reason}")
        results = await self.devices.estop_all(reason)
        failures = {k: v for k, v in results.items() if v != "stopped"}
        return {
            "stopped": results,
            "jobs_cancelled": jobs,
            "failures": failures,
            "all_stopped": not failures,
        }

    # -- description ------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        policy = self.safety.policy
        return {
            "lab": self.config.name,
            "description": self.config.description,
            "session_id": self.session_id,
            "uptime_s": round(time.time() - self.started, 1),
            "autonomy": {
                "level": int(policy.autonomy),
                "name": policy.autonomy.name,
                "hazard_ceiling": policy.ceiling().value,
                "approve_irreversible": policy.approve_irreversible,
                "always_approve": ["biological", "radiological"],
            },
            "devices": [
                {
                    "id": device.id,
                    "display_name": device.descriptor.display_name,
                    "kind": device.descriptor.kind,
                    "vendor": device.descriptor.vendor,
                    "model": device.descriptor.model,
                    "state": device.state.value,
                    "simulated": device.descriptor.simulated,
                    "protocol": device.descriptor.protocol,
                    "location": device.descriptor.location,
                    "fault": device.fault,
                    "features": sorted(device.features()),
                }
                for device in self.devices.all().values()
            ],
            "drivers": self.devices.registry.catalog(),
            "jobs_running": len([j for j in self.jobs.list() if not j.status.terminal]),
            "approvals_pending": len(self.approvals.pending()),
            "memory_stores": self.memory.ids(),
            "experiments_running": len(
                [r for r in self.experiments.list() if not r.status.terminal]
            ),
            "campaigns_running": len(
                [c for c in self.campaigns.list() if not c.status.terminal]
            ),
        }


def _strip_ground_truth(result: Any) -> Any:
    """Remove keys a real instrument could not know.

    Simulated drivers report ground truth (the true focal plane, the true
    concentration) so a run can be *scored*. Handing those to the agent would
    let it skip the measurement and read the answer, which would make every
    simulated benchmark meaningless. They are stripped here, at the boundary,
    rather than in each driver -- one place to get right, and the drivers stay
    free to be honest internally.
    """
    if not isinstance(result, dict):
        return result
    return {
        key: _strip_ground_truth(value)
        for key, value in result.items()
        if not key.startswith("truth_")
    }
