"""Replay: reconstruct what a run actually did, from the ledger alone.

Deliberately a *report*, not a re-execution. Automatically replaying a run's
physical actions against a real bench with no fresh authorisation would
re-consume samples and re-run irreversible commands under yesterday's
approval, which is precisely the failure mode `bridge/approval.py`'s digest
binding exists to prevent for a single call -- doing it across an entire
protocol would be the same mistake at larger scale. So `replay_run` answers
"what happened, in order, with what was decided and why" from the
tamper-evident ledger, which is what a reproducibility check or an incident
review actually needs, and `dry_run` separately asks "would this still be
feasible on the bench as it stands today" via simulation, which is safe to
run as often as anyone likes because it never touches hardware.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.provenance import Ledger
from .protocol import Protocol


class ReplayStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq: int
    timestamp: float
    kind: str
    device_id: str | None = None
    feature: str | None = None
    command: str | None = None
    actor: str = ""
    reason: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class ReplaySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    found: bool
    steps: list[ReplayStep] = Field(default_factory=list)
    approvals: list[dict[str, Any]] = Field(default_factory=list)
    denials: list[dict[str, Any]] = Field(default_factory=list)
    started: float | None = None
    finished: float | None = None
    outcome: str = "unknown"


def replay_run(ledger: Ledger, run_id: str) -> ReplaySummary:
    """Rebuild the timeline of one run from the provenance ledger.

    Reads every record tagged with `run_id`, in the order the hash chain
    committed them -- which is also the order they actually happened, since
    the chain cannot be reordered without breaking `ledger.verify()`.
    """
    records = list(reversed(ledger.query(run_id=run_id, limit=100_000)))
    if not records:
        return ReplaySummary(run_id=run_id, found=False, outcome="no such run in this ledger")

    steps = [
        ReplayStep(
            seq=r.seq, timestamp=r.timestamp, kind=r.kind, device_id=r.device_id,
            feature=r.feature, command=r.command, actor=r.actor, reason=r.reason,
            payload=r.payload,
        )
        for r in records
    ]
    approvals = [s.payload | {"seq": s.seq} for s in steps
                 if s.kind == "approval" and s.payload.get("state") == "granted"]
    denials = [s.payload | {"seq": s.seq} for s in steps
               if s.kind == "approval" and s.payload.get("state") in ("denied", "expired")]

    outcome = "unknown"
    for step in reversed(steps):
        if step.kind == "run_end":
            outcome = step.payload.get("status", "unknown")
            break
        if step.kind == "estop":
            outcome = "emergency_stopped"
            break

    return ReplaySummary(
        run_id=run_id, found=True, steps=steps, approvals=approvals, denials=denials,
        started=steps[0].timestamp, finished=steps[-1].timestamp, outcome=outcome,
    )


async def dry_run_protocol(protocol: Protocol, gateway: Any) -> dict[str, Any]:
    """Ask every step's digital twin whether the protocol still looks safe.

    Runs `device.simulate` for each step against the bench's *current* state,
    substituting `${...}` references from the previous *simulated* results
    rather than real ones -- so this can be called before a run exists at all,
    which is the point: it is the protocol-level sibling of
    `Command.simulate`, and it never moves anything.
    """
    results: dict[str, dict[str, Any]] = {}
    report: list[dict[str, Any]] = []
    for index, step in enumerate(protocol.steps):
        label = step.resolved_label(index)
        try:
            args = protocol.resolve_args(step, results, protocol.variables)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            report.append({"step": label, "feasible": False, "error": str(exc)})
            break
        device = gateway.device(step.device)
        sim = await device.simulate(step.feature, step.command, args)
        report.append({
            "step": label, "feasible": sim.feasible, "fidelity": sim.fidelity,
            "violations": sim.violations, "warnings": sim.warnings,
        })
        results[label] = {"result": sim.predicted_state}
        if not sim.feasible:
            break
    return {
        "protocol": protocol.name,
        "feasible": all(r["feasible"] for r in report) and len(report) == len(protocol.steps),
        "steps": report,
    }
