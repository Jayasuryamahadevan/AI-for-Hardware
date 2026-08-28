"""Error taxonomy for LabBench.

Errors are deliberately *typed* rather than free-text: an agent needs to know
whether a failure is retryable, whether it left the instrument in an unknown
state, and whether a human must intervene. That distinction is what makes
autonomous recovery possible instead of guesswork.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class Recovery(str, Enum):
    """What an agent may legitimately do next after a failure."""

    RETRY = "retry"                  # transient; same call may succeed
    RETRY_AFTER_FIX = "retry_after_fix"  # fix the arguments, then retry
    REINITIALIZE = "reinitialize"    # device must be re-homed/re-initialized
    HUMAN_REQUIRED = "human_required"  # physical intervention needed
    ABORT = "abort"                  # do not retry; the plan is invalid


class LabBenchError(Exception):
    """Base class. Carries structured detail so it survives the MCP boundary."""

    code = "labbench_error"
    recovery = Recovery.ABORT
    #: True when the physical state of the instrument may no longer match the model.
    state_uncertain = False

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "recovery": self.recovery.value,
            "state_uncertain": self.state_uncertain,
            "detail": self.detail,
        }


class DeviceNotFound(LabBenchError):
    code = "device_not_found"
    recovery = Recovery.RETRY_AFTER_FIX


class CapabilityNotFound(LabBenchError):
    code = "capability_not_found"
    recovery = Recovery.RETRY_AFTER_FIX


class ValidationError(LabBenchError):
    """Arguments failed schema or constraint checking. Never reached hardware."""

    code = "validation_error"
    recovery = Recovery.RETRY_AFTER_FIX


class ConstraintViolation(ValidationError):
    """A declared operating-envelope (ODD) constraint would be violated."""

    code = "constraint_violation"


class SafetyViolation(LabBenchError):
    """Blocked by the safety kernel. Not retryable without a policy change."""

    code = "safety_violation"
    recovery = Recovery.ABORT


class ApprovalRequired(LabBenchError):
    """Autonomy level demands a human signature this call did not carry."""

    code = "approval_required"
    recovery = Recovery.HUMAN_REQUIRED


class ApprovalDenied(LabBenchError):
    code = "approval_denied"
    recovery = Recovery.ABORT


class DeviceBusy(LabBenchError):
    code = "device_busy"
    recovery = Recovery.RETRY


class DeviceNotReady(LabBenchError):
    """Wrong lifecycle state, e.g. commanded before initialize()."""

    code = "device_not_ready"
    recovery = Recovery.REINITIALIZE


class TransportError(LabBenchError):
    """Lost contact with the instrument. State is, by definition, unknown."""

    code = "transport_error"
    recovery = Recovery.REINITIALIZE
    state_uncertain = True


class DeviceFault(LabBenchError):
    """The instrument reported a hardware fault."""

    code = "device_fault"
    recovery = Recovery.HUMAN_REQUIRED
    state_uncertain = True


class JobNotFound(LabBenchError):
    code = "job_not_found"
    recovery = Recovery.RETRY_AFTER_FIX


class Cancelled(LabBenchError):
    code = "cancelled"
    recovery = Recovery.ABORT


class DriverUnavailable(LabBenchError):
    """Driver requires an optional dependency that is not installed."""

    code = "driver_unavailable"
    recovery = Recovery.HUMAN_REQUIRED
