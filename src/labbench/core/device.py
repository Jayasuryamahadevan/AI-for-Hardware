"""Device abstraction: the contract every driver implements.

A driver's whole job is to answer four questions — what can you do, what is
your state, do this, and what *would* happen if you did this. That last one
(`simulate`) is unusual and is the point of the design: it lets the safety
kernel check a proposed action against a model before any matter moves.
"""

from __future__ import annotations

import abc
import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .capability import Command, Feature, Hazard, Property
from .errors import (
    CapabilityNotFound,
    DeviceNotReady,
    LabBenchError,
)


class DeviceState(str, Enum):
    """Lifecycle state, aligned with the OPC UA LADS device state machine.

    Commands declare which states they tolerate; the kernel refuses the rest.
    """

    OFFLINE = "offline"            # not connected
    CONNECTING = "connecting"
    INITIALIZING = "initializing"  # homing / calibrating
    IDLE = "idle"                  # connected, ready, not executing
    BUSY = "busy"                  # executing a command
    PAUSED = "paused"
    FAULT = "fault"                # hardware error; needs clearing
    MAINTENANCE = "maintenance"    # deliberately withheld from agents

    @property
    def operational(self) -> bool:
        return self in (DeviceState.IDLE, DeviceState.BUSY, DeviceState.PAUSED)


class DeviceDescriptor(BaseModel):
    """Identity and provenance of one instrument.

    `serial` and `firmware` are not decoration: reproducing a result a year
    later requires knowing precisely which box produced it.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    display_name: str = ""
    kind: str = "instrument"          # microscope | liquid_handler | reader | ...
    vendor: str = "unknown"
    model: str = "unknown"
    serial: str | None = None
    firmware: str | None = None
    #: Which southbound protocol this driver speaks.
    protocol: str = "native"          # sila2 | opcua-lads | scpi | mmcore | wot | http
    #: Physical siting, used for collision and containment reasoning.
    location: str | None = None
    driver: str = ""
    driver_version: str = "0.1.0"
    #: True when this device is entirely simulated; agents must be able to tell.
    simulated: bool = False
    labels: dict[str, str] = Field(default_factory=dict)


class TelemetrySample(BaseModel):
    device_id: str
    feature: str
    property: str
    value: Any
    unit: str | None = None
    timestamp: float = Field(default_factory=time.time)


class DeviceEvent(BaseModel):
    device_id: str
    feature: str
    name: str
    payload: dict[str, Any] = Field(default_factory=dict)
    severity: str = "info"
    timestamp: float = Field(default_factory=time.time)


class ExecutionContext(BaseModel):
    """Passed into every command invocation.

    Carries the cooperative-cancellation handle and the progress sink. Drivers
    are expected to poll `cancelled` at every natural interruption point — a
    long acquisition that cannot be stopped is a safety problem, not a feature.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    job_id: str = "inline"
    run_id: str | None = None
    actor: str = "agent"
    dry_run: bool = False
    cancel_event: asyncio.Event | None = None
    _progress: Callable[[float, str], Awaitable[None]] | None = None

    @property
    def cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    async def progress(self, fraction: float, message: str = "") -> None:
        if self._progress is not None:
            await self._progress(max(0.0, min(1.0, fraction)), message)

    def with_progress(
        self, fn: Callable[[float, str], Awaitable[None]]
    ) -> ExecutionContext:
        object.__setattr__(self, "_progress", fn)
        return self

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            from .errors import Cancelled

            raise Cancelled("cancelled by operator", job_id=self.job_id)


class SimulationResult(BaseModel):
    """Prediction produced by `Device.simulate` — the CRUTD 'Undergo' phase.

    A driver that cannot predict must say so (`fidelity="none"`) rather than
    silently returning success, otherwise the safety gate becomes theatre.
    """

    feasible: bool
    fidelity: str = "kinematic"   # none | kinematic | reduced_order | high
    predicted_state: dict[str, Any] = Field(default_factory=dict)
    predicted_duration_s: float | None = None
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: str = ""


EventSink = Callable[[DeviceEvent], Awaitable[None] | None]


class Device(abc.ABC):
    """Base class for all instrument drivers.

    Subclasses implement `_features`, `_read`, `_write` and `_invoke`; the base
    handles validation, state gating, precondition checks and event fan-out so
    that no driver can accidentally skip them.
    """

    #: Optional import name whose absence means the driver cannot run.
    requires_package: str | None = None

    def __init__(self, descriptor: DeviceDescriptor, **config: Any) -> None:
        self.descriptor = descriptor
        self.config = config
        self._state = DeviceState.OFFLINE
        self._fault: str | None = None
        self._lock = asyncio.Lock()
        self._sinks: list[EventSink] = []
        self._features_cache: dict[str, Feature] | None = None
        self._last_error: LabBenchError | None = None

    # -- identity ---------------------------------------------------------

    @property
    def id(self) -> str:
        return self.descriptor.id

    @property
    def state(self) -> DeviceState:
        return self._state

    @property
    def fault(self) -> str | None:
        return self._fault

    def features(self) -> dict[str, Feature]:
        if self._features_cache is None:
            self._features_cache = {f.identifier: f for f in self._features()}
        return self._features_cache

    def invalidate_features(self) -> None:
        """Call after runtime discovery narrows constraints (e.g. after homing)."""
        self._features_cache = None

    def resolve(self, feature: str, command: str) -> tuple[Feature, Command]:
        feats = self.features()
        if feature not in feats:
            raise CapabilityNotFound(
                f"device {self.id!r} has no feature {feature!r}",
                device=self.id, available=sorted(feats),
            )
        f = feats[feature]
        cmd = f.command(command)
        if cmd is None:
            raise CapabilityNotFound(
                f"feature {feature!r} has no command {command!r}",
                device=self.id, feature=feature,
                available=[c.name for c in f.commands],
            )
        return f, cmd

    def resolve_property(self, feature: str, name: str) -> tuple[Feature, Property]:
        feats = self.features()
        if feature not in feats:
            raise CapabilityNotFound(
                f"device {self.id!r} has no feature {feature!r}",
                device=self.id, available=sorted(feats),
            )
        f = feats[feature]
        prop = f.property(name)
        if prop is None:
            raise CapabilityNotFound(
                f"feature {feature!r} has no property {name!r}",
                device=self.id, feature=feature,
                available=[p.name for p in f.properties],
            )
        return f, prop

    # -- events -----------------------------------------------------------

    def subscribe(self, sink: EventSink) -> Callable[[], None]:
        self._sinks.append(sink)
        return lambda: self._sinks.remove(sink) if sink in self._sinks else None

    async def emit(
        self, feature: str, name: str, payload: dict[str, Any] | None = None,
        severity: str = "info",
    ) -> None:
        evt = DeviceEvent(
            device_id=self.id, feature=feature, name=name,
            payload=payload or {}, severity=severity,
        )
        for sink in list(self._sinks):
            res = sink(evt)
            if asyncio.iscoroutine(res):
                await res

    async def _set_state(self, state: DeviceState, reason: str = "") -> None:
        if state == self._state:
            return
        old, self._state = self._state, state
        await self.emit(
            "DeviceLifecycle", "state_changed",
            {"from": old.value, "to": state.value, "reason": reason},
            severity="warning" if state is DeviceState.FAULT else "info",
        )

    # -- lifecycle --------------------------------------------------------

    async def connect(self) -> None:
        await self._set_state(DeviceState.CONNECTING)
        try:
            await self._connect()
        except Exception as exc:
            await self.fail(f"connect failed: {exc}")
            raise
        await self._set_state(DeviceState.IDLE, "connected")

    async def disconnect(self) -> None:
        try:
            await self._disconnect()
        finally:
            await self._set_state(DeviceState.OFFLINE, "disconnected")

    async def initialize(self, ctx: ExecutionContext | None = None) -> None:
        """Home / calibrate. Many commands are gated behind this."""
        prev = self._state
        await self._set_state(DeviceState.INITIALIZING)
        try:
            await self._initialize(ctx or ExecutionContext())
        except Exception as exc:
            await self.fail(f"initialize failed: {exc}")
            raise
        self.invalidate_features()
        await self._set_state(
            DeviceState.IDLE if prev is not DeviceState.MAINTENANCE else prev,
            "initialized",
        )

    async def fail(self, reason: str) -> None:
        self._fault = reason
        await self._set_state(DeviceState.FAULT, reason)

    async def clear_fault(self) -> None:
        self._fault = None
        await self._set_state(DeviceState.IDLE, "fault cleared")

    async def estop(self, reason: str = "operator e-stop") -> None:
        """Halt everything now. Must be safe to call from any state.

        Deliberately does *not* try to be graceful: it stops motion first and
        reports afterwards.
        """
        try:
            await self._estop()
        finally:
            await self.emit(
                "DeviceLifecycle", "emergency_stop",
                {"reason": reason}, severity="critical",
            )
            await self.fail(f"emergency stop: {reason}")

    # -- data plane -------------------------------------------------------

    async def read(self, feature: str, name: str) -> TelemetrySample:
        _, prop = self.resolve_property(feature, name)
        value = await self._read(feature, name)
        return TelemetrySample(
            device_id=self.id, feature=feature, property=name,
            value=value, unit=prop.schema_.unit,
        )

    async def read_all(self, feature: str | None = None) -> dict[str, Any]:
        """Snapshot of every readable property, flattened as ``Feature.prop``.

        Used by the safety kernel for precondition evaluation and by the
        provenance log for before/after state capture.
        """
        out: dict[str, Any] = {}
        for fid, f in self.features().items():
            if feature is not None and fid != feature:
                continue
            for prop in f.properties:
                try:
                    out[f"{fid}.{prop.name}"] = await self._read(fid, prop.name)
                except Exception as exc:  # noqa: BLE001 - a broken sensor must not blind the rest
                    out[f"{fid}.{prop.name}"] = {"error": str(exc)}
        return out

    async def write(self, feature: str, name: str, value: Any) -> None:
        _, prop = self.resolve_property(feature, name)
        if not prop.writable:
            raise CapabilityNotFound(
                f"property {feature}.{name} is read-only",
                device=self.id, feature=feature, property=name,
            )
        coerced = prop.schema_.validate_value(value)
        self._require_operational()
        await self._write(feature, name, coerced)

    async def invoke(
        self, feature: str, command: str, args: dict[str, Any],
        ctx: ExecutionContext | None = None,
    ) -> dict[str, Any]:
        """Validated, state-gated, precondition-checked command execution."""
        ctx = ctx or ExecutionContext()
        _, cmd = self.resolve(feature, command)
        clean = cmd.validate_args(args)
        if cmd.hazard is not Hazard.NONE:
            self._require_operational()
        await self._check_preconditions(cmd)

        if ctx.dry_run:
            sim = await self.simulate(feature, command, clean)
            return {"dry_run": True, **sim.model_dump()}

        acquired = cmd.exclusive
        if acquired:
            await self._lock.acquire()
        prev_state = self._state
        try:
            if cmd.observable:
                await self._set_state(DeviceState.BUSY, f"{feature}.{command}")
            result = await self._invoke(feature, command, clean, ctx)
            return result if isinstance(result, dict) else {"value": result}
        except LabBenchError as exc:
            self._last_error = exc
            if exc.state_uncertain:
                await self.fail(exc.message)
            raise
        except Exception as exc:
            await self.fail(f"{feature}.{command}: {exc}")
            raise
        finally:
            if self._state is DeviceState.BUSY:
                await self._set_state(
                    prev_state if prev_state.operational else DeviceState.IDLE, "done"
                )
            if acquired:
                self._lock.release()

    async def simulate(
        self, feature: str, command: str, args: dict[str, Any]
    ) -> SimulationResult:
        """Predict the outcome without touching hardware."""
        _, cmd = self.resolve(feature, command)
        clean = cmd.validate_args(args)
        if not cmd.simulatable:
            return SimulationResult(
                feasible=True, fidelity="none",
                warnings=[
                    (f"{feature}.{command} has no simulation model; "
                     "outcome cannot be verified before execution")
                ],
            )
        return await self._simulate(feature, command, clean)

    # -- helpers for subclasses -------------------------------------------

    def _require_operational(self) -> None:
        if not self._state.operational:
            raise DeviceNotReady(
                f"device {self.id!r} is {self._state.value}"
                + (f" ({self._fault})" if self._fault else ""),
                device=self.id, state=self._state.value, fault=self._fault,
            )

    async def _check_preconditions(self, cmd: Command) -> None:
        if not cmd.preconditions:
            return
        snapshot = await self.read_all()
        # Preconditions may name properties unqualified; accept both forms.
        flat = dict(snapshot)
        for key, val in snapshot.items():
            flat.setdefault(key.split(".", 1)[-1], val)
        failures = []
        for pc in cmd.preconditions:
            ok, why = pc.evaluate(flat)
            if not ok:
                failures.append(why)
        if failures:
            raise DeviceNotReady(
                f"{cmd.name}: preconditions not met: " + "; ".join(failures),
                device=self.id, command=cmd.name, failures=failures,
            )

    # -- abstract ---------------------------------------------------------

    @abc.abstractmethod
    def _features(self) -> Sequence[Feature]: ...

    @abc.abstractmethod
    async def _read(self, feature: str, name: str) -> Any: ...

    async def _write(self, feature: str, name: str, value: Any) -> None:
        raise CapabilityNotFound(f"{self.id} has no writable properties", device=self.id)

    @abc.abstractmethod
    async def _invoke(
        self, feature: str, command: str, args: dict[str, Any], ctx: ExecutionContext
    ) -> Any: ...

    async def _simulate(
        self, feature: str, command: str, args: dict[str, Any]
    ) -> SimulationResult:
        return SimulationResult(
            feasible=True, fidelity="none",
            warnings=["driver provides no digital twin"],
        )

    async def _connect(self) -> None: ...
    async def _disconnect(self) -> None: ...
    async def _initialize(self, ctx: ExecutionContext) -> None: ...

    async def _estop(self) -> None:
        """Default e-stop is a no-op. Motion-capable drivers MUST override."""
