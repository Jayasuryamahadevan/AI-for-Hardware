"""LabBench — a research-grade gateway connecting any AI agent to laboratory hardware.

The package is layered so that each layer is useful without the one above it:

    core/        capability model, safety kernel, jobs, provenance  (no I/O)
    drivers/     one module per southbound protocol
    memory/      durable notes and documents an agent can search
    experiment/  protocols and runs built on top of devices
    protocol/    JSON-RPC 2.0 over HTTP, WebSocket and stdio -- the wire format
    bridge/      the northbound tool surface: schemas per AI dialect, approvals
    gateway.py   assembly: request -> ledger -> safety -> approval -> act
    cli.py       operator entry point

Import cost matters here: `labbench.core` must stay importable on a machine with
no instrument libraries installed at all, because that is the machine an auditor
reads the ledger on.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .core.capability import (
    Command,
    Constraint,
    Event,
    Feature,
    Hazard,
    Parameter,
    Precondition,
    Property,
    Reversibility,
)
from .core.device import (
    Device,
    DeviceDescriptor,
    DeviceEvent,
    DeviceState,
    ExecutionContext,
    SimulationResult,
    TelemetrySample,
)
from .core.errors import LabBenchError, Recovery
from .core.jobs import Artifact, Job, JobManager, JobStatus
from .core.provenance import Ledger, Record
from .core.registry import DeviceConfig, DeviceManager, DriverRegistry, LabConfig
from .core.safety import (
    AutonomyLevel,
    Decision,
    Effect,
    PolicyRule,
    SafetyKernel,
    SafetyPolicy,
)

__all__ = [  # noqa: RUF022 - grouped by topic to mirror the import block above, not sorted
    "__version__",
    # capability
    "Command", "Constraint", "Event", "Feature", "Hazard", "Parameter",
    "Precondition", "Property", "Reversibility",
    # device
    "Device", "DeviceDescriptor", "DeviceEvent", "DeviceState",
    "ExecutionContext", "SimulationResult", "TelemetrySample",
    # errors
    "LabBenchError", "Recovery",
    # jobs
    "Artifact", "Job", "JobManager", "JobStatus",
    # provenance
    "Ledger", "Record",
    # registry
    "DeviceConfig", "DeviceManager", "DriverRegistry", "LabConfig",
    # safety
    "AutonomyLevel", "Decision", "Effect", "PolicyRule", "SafetyKernel",
    "SafetyPolicy",
]
