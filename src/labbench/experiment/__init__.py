"""Protocols and runs built on top of devices.

A `Protocol` is a named, checkable sequence of `device.invoke` calls; an
`ExperimentManager` executes one as a `Run`, through the same `Gateway.invoke`
front door an agent would use directly, so a run gets the ledger, the safety
kernel and the approval broker for free rather than as a parallel
implementation to keep in sync. `replay_run` reconstructs a finished run's
timeline from the ledger; `dry_run_protocol` asks every step's digital twin
whether the plan still looks safe, without touching hardware.
"""

from __future__ import annotations

from .protocol import Protocol, ProtocolStep
from .replay import ReplayStep, ReplaySummary, dry_run_protocol, replay_run
from .run import ExperimentManager, Run, RunStatus, StepResult

__all__ = [
    "ExperimentManager",
    "Protocol",
    "ProtocolStep",
    "ReplayStep",
    "ReplaySummary",
    "Run",
    "RunStatus",
    "StepResult",
    "dry_run_protocol",
    "replay_run",
]
