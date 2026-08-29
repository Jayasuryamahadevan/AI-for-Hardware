"""A simulated CO2 incubator with plate storage.

Thermal and gas dynamics are first-order lags with real time constants, which
is the only thing that makes this instrument interesting to an agent: setpoints
are not achievements. Setting 37 C and immediately loading cells is a mistake
the simulation will let you make and then show you, because the chamber takes
minutes to arrive and the log records what the plate actually experienced.

Opening the door dumps heat and CO2 and starts a recovery transient. An agent
that opens the door every two minutes to check on a culture never lets the
chamber stabilise -- which is exactly the behaviour a human trainee has to be
taught out of, and which no static simulator can teach.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Sequence
from typing import Any

from ...core.capability import (
    Command,
    Constraint,
    Event,
    Feature,
    Hazard,
    Parameter,
    Property,
    Reversibility,
)
from ...core.device import Device, DeviceDescriptor, ExecutionContext, SimulationResult
from ...core.errors import ConstraintViolation, DeviceNotReady
from . import _labware

#: Time constants in seconds. Chosen so a 22 -> 37 C pull takes roughly twenty
#: minutes to get inside a tenth of a degree, which is what a real box does.
TAU_TEMPERATURE_S = 420.0
TAU_CO2_S = 150.0
TAU_HUMIDITY_S = 900.0

AMBIENT_C = 22.0
AMBIENT_CO2_PCT = 0.04
AMBIENT_RH_PCT = 40.0


class SimulatedIncubator(Device):
    """CO2 incubator with shelves, gas control and a door that costs you."""

    def __init__(self, descriptor: DeviceDescriptor, **config: Any) -> None:
        super().__init__(descriptor, **config)
        self.slots = int(config.get("slots", 12))
        self.temperature_c = AMBIENT_C
        self.target_temperature_c = AMBIENT_C
        self.co2_pct = AMBIENT_CO2_PCT
        self.target_co2_pct = AMBIENT_CO2_PCT
        self.humidity_pct = AMBIENT_RH_PCT
        self.target_humidity_pct = AMBIENT_RH_PCT
        self.door_open = False
        self.shaking = False
        self.shake_rpm = 0.0
        self.stored: dict[int, str] = {}
        self.door_openings = 0
        self._last_update = time.time()
        #: Compresses simulated time so a demo does not take twenty minutes.
        #: Reported honestly in every reading rather than hidden.
        self.time_scale = float(config.get("time_scale", 60.0))

    # -- dynamics ---------------------------------------------------------

    def _advance(self) -> float:
        """Integrate the chamber forward to now. Called before every read.

        Evaluating lazily rather than on a timer means the model is exact at
        the moment anyone looks, and the device holds no background task.
        """
        now = time.time()
        dt = (now - self._last_update) * self.time_scale
        self._last_update = now
        if dt <= 0:
            return 0.0

        # An open door couples the chamber to the room hard.
        if self.door_open:
            temperature_target, tau_t = AMBIENT_C, 60.0
            co2_target, tau_c = AMBIENT_CO2_PCT, 25.0
            humidity_target, tau_h = AMBIENT_RH_PCT, 120.0
        else:
            temperature_target, tau_t = self.target_temperature_c, TAU_TEMPERATURE_S
            co2_target, tau_c = self.target_co2_pct, TAU_CO2_S
            humidity_target, tau_h = self.target_humidity_pct, TAU_HUMIDITY_S

        self.temperature_c += (temperature_target - self.temperature_c) * (
            1 - math.exp(-dt / tau_t)
        )
        self.co2_pct += (co2_target - self.co2_pct) * (1 - math.exp(-dt / tau_c))
        self.humidity_pct += (humidity_target - self.humidity_pct) * (1 - math.exp(-dt / tau_h))

        # Plates inside experience the chamber, and evaporate accordingly.
        for barcode in self.stored.values():
            try:
                plate = _labware.BENCH.get(barcode)
            except KeyError:
                continue
            plate.temperature_c = self.temperature_c
            # Humidified air is the whole point of a CO2 incubator: it is what
            # stops a culture plate drying out overnight.
            dryness = max(0.0, (95.0 - self.humidity_pct) / 95.0)
            plate.apply_evaporation(dt * dryness, self.temperature_c)
        return dt

    @property
    def at_setpoint(self) -> bool:
        return (
            abs(self.temperature_c - self.target_temperature_c) < 0.2
            and abs(self.co2_pct - self.target_co2_pct) < 0.15
            and not self.door_open
        )

    # -- capabilities -----------------------------------------------------

    def _features(self) -> Sequence[Feature]:
        return [
            Feature(
                identifier="TemperatureControl",
                display_name="Temperature",
                namespace="org.labbench.environment",
                properties=[
                    Property(name="temperature_c", description="Measured chamber temperature.",
                             schema=Parameter(name="temperature_c", unit="degC"), observable=True),
                    Property(name="target_temperature_c", description="Setpoint.",
                             schema=Parameter(name="target_temperature_c", unit="degC")),
                    Property(name="at_setpoint",
                             description="True once temperature and CO2 have both settled and "
                                         "the door is shut. Setting a target does not make "
                                         "this true.",
                             schema=Parameter(name="at_setpoint", type="boolean")),
                ],
                commands=[
                    Command(
                        name="set_temperature",
                        description="Set the temperature setpoint. The chamber approaches it "
                                    "with a time constant of several minutes.",
                        parameters=[Parameter(name="temperature_c", unit="degC",
                                              description="Setpoint.",
                                              constraint=Constraint(minimum=4.0, maximum=50.0))],
                        duration_estimate_s=1.0,
                        hazard=Hazard.THERMAL, reversibility=Reversibility.RESTORABLE,
                        tags={"affects_culture"},
                    ),
                    Command(
                        name="wait_for_setpoint",
                        description="Block until the chamber has settled. Long-running.",
                        parameters=[
                            Parameter(name="timeout_s", unit="s", default=1800.0, required=False,
                                      constraint=Constraint(minimum=1.0, maximum=86400.0)),
                            Parameter(name="tolerance_c", unit="degC", default=0.2,
                                      required=False,
                                      constraint=Constraint(minimum=0.05, maximum=5.0)),
                        ],
                        observable=True, duration_estimate_s=600.0,
                        hazard=Hazard.NONE, reversibility=Reversibility.REVERSIBLE,
                    ),
                ],
                events=[
                    Event(name="excursion",
                          description="Temperature left the acceptable band around the setpoint.",
                          severity="warning"),
                ],
            ),
            Feature(
                identifier="GasControl",
                display_name="Atmosphere",
                namespace="org.labbench.environment",
                properties=[
                    Property(name="co2_pct", description="Measured CO2.",
                             schema=Parameter(name="co2_pct", unit="%"), observable=True),
                    Property(name="target_co2_pct", description="CO2 setpoint.",
                             schema=Parameter(name="target_co2_pct", unit="%")),
                    Property(name="humidity_pct", description="Relative humidity.",
                             schema=Parameter(name="humidity_pct", unit="%"), observable=True),
                ],
                commands=[
                    Command(
                        name="set_co2",
                        description="Set the CO2 setpoint. Buffered media depend on this; "
                                    "losing it swings culture pH.",
                        parameters=[Parameter(name="co2_pct", unit="%",
                                              constraint=Constraint(minimum=0.0, maximum=20.0))],
                        duration_estimate_s=1.0,
                        # Gas composition acts directly on living cells.
                        hazard=Hazard.BIOLOGICAL, reversibility=Reversibility.RESTORABLE,
                        tags={"affects_culture"},
                    ),
                    Command(
                        name="set_humidity",
                        description="Set the relative humidity setpoint.",
                        parameters=[Parameter(name="humidity_pct", unit="%",
                                              constraint=Constraint(minimum=20.0, maximum=98.0))],
                        duration_estimate_s=1.0, hazard=Hazard.BENIGN,
                        reversibility=Reversibility.RESTORABLE,
                    ),
                ],
            ),
            Feature(
                identifier="PlateStorage",
                display_name="Shelves",
                namespace="org.labbench.handling",
                properties=[
                    Property(name="door_open", description="Door position.",
                             schema=Parameter(name="door_open", type="boolean")),
                    Property(name="occupied_slots", description="Slots in use.",
                             schema=Parameter(name="occupied_slots", type="integer")),
                    Property(name="capacity", description="Total slots.",
                             schema=Parameter(name="capacity", type="integer")),
                    Property(name="contents", description="slot -> plate barcode.",
                             schema=Parameter(name="contents", type="object")),
                    Property(name="door_openings",
                             description="Door cycles since connect. Each one costs a transient.",
                             schema=Parameter(name="door_openings", type="integer")),
                ],
                commands=[
                    Command(
                        name="open_door",
                        description="Open the door. Temperature and CO2 begin falling to room "
                                    "conditions immediately.",
                        duration_estimate_s=1.0,
                        hazard=Hazard.BIOLOGICAL, reversibility=Reversibility.RESTORABLE,
                        inverse="close_door", tags={"affects_culture", "containment"},
                    ),
                    Command(name="close_door", description="Close the door and begin recovery.",
                            duration_estimate_s=1.0, hazard=Hazard.BENIGN,
                            reversibility=Reversibility.REVERSIBLE, inverse="open_door"),
                    Command(
                        name="store_plate",
                        description="Put a plate from the bench into a shelf slot.",
                        parameters=[
                            Parameter(name="barcode", type="string"),
                            Parameter(name="slot", type="integer", required=False,
                                      description="Slot number. Omit to use the first free one.",
                                      constraint=Constraint(minimum=1)),
                        ],
                        duration_estimate_s=6.0, hazard=Hazard.BIOLOGICAL,
                        reversibility=Reversibility.REVERSIBLE, inverse="retrieve_plate",
                        tags={"moves_labware", "containment"},
                    ),
                    Command(
                        name="retrieve_plate",
                        description="Take a plate off a shelf and return it to the bench.",
                        parameters=[Parameter(name="barcode", type="string")],
                        duration_estimate_s=6.0, hazard=Hazard.BIOLOGICAL,
                        reversibility=Reversibility.REVERSIBLE, inverse="store_plate",
                        tags={"moves_labware", "containment"},
                    ),
                ],
                events=[
                    Event(name="door_left_open",
                          description="The door has been open long enough to matter.",
                          severity="warning"),
                ],
            ),
        ]

    # -- lifecycle --------------------------------------------------------

    async def _connect(self) -> None:
        self._last_update = time.time()
        await asyncio.sleep(0.05)

    async def _estop(self) -> None:
        """Stop heating, gas and shaking. The door state is left alone: forcing
        it either way during an emergency is its own hazard."""
        self.target_temperature_c = AMBIENT_C
        self.target_co2_pct = AMBIENT_CO2_PCT
        self.shaking = False
        self.shake_rpm = 0.0

    async def _read(self, feature: str, name: str) -> Any:
        self._advance()
        return {
            ("TemperatureControl", "temperature_c"): round(self.temperature_c, 3),
            ("TemperatureControl", "target_temperature_c"): self.target_temperature_c,
            ("TemperatureControl", "at_setpoint"): self.at_setpoint,
            ("GasControl", "co2_pct"): round(self.co2_pct, 3),
            ("GasControl", "target_co2_pct"): self.target_co2_pct,
            ("GasControl", "humidity_pct"): round(self.humidity_pct, 2),
            ("PlateStorage", "door_open"): self.door_open,
            ("PlateStorage", "occupied_slots"): len(self.stored),
            ("PlateStorage", "capacity"): self.slots,
            ("PlateStorage", "contents"): {str(k): v for k, v in sorted(self.stored.items())},
            ("PlateStorage", "door_openings"): self.door_openings,
        }[(feature, name)]

    async def _invoke(
        self, feature: str, command: str, args: dict[str, Any], ctx: ExecutionContext
    ) -> Any:
        self._advance()
        return await getattr(self, f"_cmd_{command}")(ctx, **args)

    # -- environment ------------------------------------------------------

    async def _cmd_set_temperature(
        self, ctx: ExecutionContext, temperature_c: float
    ) -> dict[str, Any]:
        self.target_temperature_c = float(temperature_c)
        gap = abs(self.temperature_c - self.target_temperature_c)
        # An honest estimate rather than a claim of arrival: 3 tau reaches ~95%.
        eta = 3 * TAU_TEMPERATURE_S / self.time_scale if gap > 0.2 else 0.0
        return {
            "target_temperature_c": self.target_temperature_c,
            "temperature_c": round(self.temperature_c, 3),
            "at_setpoint": self.at_setpoint,
            "estimated_settle_s": round(eta, 1),
            "note": "the setpoint is set, not reached; poll temperature_c or call "
                    "wait_for_setpoint",
        }

    async def _cmd_set_co2(self, ctx: ExecutionContext, co2_pct: float) -> dict[str, Any]:
        self.target_co2_pct = float(co2_pct)
        return {"target_co2_pct": self.target_co2_pct, "co2_pct": round(self.co2_pct, 3),
                "at_setpoint": self.at_setpoint}

    async def _cmd_set_humidity(
        self, ctx: ExecutionContext, humidity_pct: float
    ) -> dict[str, Any]:
        self.target_humidity_pct = float(humidity_pct)
        return {"target_humidity_pct": self.target_humidity_pct,
                "humidity_pct": round(self.humidity_pct, 2)}

    async def _cmd_wait_for_setpoint(
        self, ctx: ExecutionContext, timeout_s: float = 1800.0, tolerance_c: float = 0.2
    ) -> dict[str, Any]:
        started = time.time()
        deadline = started + timeout_s / self.time_scale
        while time.time() < deadline:
            ctx.raise_if_cancelled()
            self._advance()
            gap = abs(self.temperature_c - self.target_temperature_c)
            if gap <= tolerance_c and not self.door_open:
                return {
                    "settled": True,
                    "temperature_c": round(self.temperature_c, 3),
                    "co2_pct": round(self.co2_pct, 3),
                    "waited_s": round((time.time() - started) * self.time_scale, 1),
                }
            await ctx.progress(
                min(0.99, 1.0 - gap / max(abs(self.target_temperature_c - AMBIENT_C), 1e-6)),
                f"{self.temperature_c:.2f} degC, {gap:.2f} to go",
            )
            await asyncio.sleep(0.05)
        return {
            "settled": False,
            "temperature_c": round(self.temperature_c, 3),
            "gap_c": round(abs(self.temperature_c - self.target_temperature_c), 3),
            "waited_s": round(timeout_s, 1),
            "note": "did not settle within the timeout; the door may be open or the "
                    "setpoint unreachable",
        }

    # -- storage ----------------------------------------------------------

    async def _cmd_open_door(self, ctx: ExecutionContext) -> dict[str, Any]:
        if self.door_open:
            return {"door_open": True, "already": True}
        self.door_open = True
        self.door_openings += 1
        before = self.temperature_c
        await self.emit(
            "PlateStorage", "door_opened",
            {"temperature_c": round(before, 2), "cultures_inside": len(self.stored)},
            severity="warning" if self.stored else "info",
        )
        return {
            "door_open": True,
            "temperature_c": round(self.temperature_c, 3),
            "plates_exposed": len(self.stored),
            "warning": "temperature and CO2 are now falling toward room conditions",
        }

    async def _cmd_close_door(self, ctx: ExecutionContext) -> dict[str, Any]:
        self.door_open = False
        gap = abs(self.temperature_c - self.target_temperature_c)
        if gap > 1.0:
            await self.emit(
                "TemperatureControl", "excursion",
                {"temperature_c": round(self.temperature_c, 2),
                 "target_c": self.target_temperature_c, "gap_c": round(gap, 2)},
                severity="warning",
            )
        return {
            "door_open": False,
            "temperature_c": round(self.temperature_c, 3),
            "recovering": gap > 0.2,
            "gap_c": round(gap, 3),
        }

    async def _cmd_store_plate(
        self, ctx: ExecutionContext, barcode: str, slot: int | None = None
    ) -> dict[str, Any]:
        try:
            _labware.BENCH.get(barcode)  # fail fast on an unknown barcode before taking the lock
        except KeyError as exc:
            raise ConstraintViolation(str(exc), barcode=barcode) from None
        # Held across the door-open/settle awaits below: two instruments
        # racing to claim the same plate must not both pass the `location`
        # check before either has actually claimed it.
        async with _labware.BENCH.hold(barcode) as plate:
            if plate.location not in ("bench", self.id):
                raise ConstraintViolation(
                    f"plate {barcode!r} is at {plate.location!r}, not on the bench",
                    barcode=barcode, location=plate.location,
                )
            if barcode in self.stored.values():
                raise ConstraintViolation(f"plate {barcode!r} is already stored", barcode=barcode)
            if slot is None:
                free = [s for s in range(1, self.slots + 1) if s not in self.stored]
                if not free:
                    raise DeviceNotReady(
                        f"every one of the {self.slots} slots is occupied",
                        device=self.id, capacity=self.slots,
                    )
                slot = free[0]
            if slot in self.stored:
                raise ConstraintViolation(
                    f"slot {slot} already holds {self.stored[slot]!r}",
                    slot=slot, occupant=self.stored[slot],
                )
            if not 1 <= slot <= self.slots:
                raise ConstraintViolation(
                    f"slot {slot} does not exist; this incubator has slots 1-{self.slots}",
                    slot=slot, capacity=self.slots,
                )
            if not self.door_open:
                await self._cmd_open_door(ctx)
            await asyncio.sleep(0.2)
            self.stored[slot] = barcode
            plate.location = self.id
        await self._cmd_close_door(ctx)
        return {"stored": barcode, "slot": slot, "occupied_slots": len(self.stored),
                "temperature_c": round(self.temperature_c, 3)}

    async def _cmd_retrieve_plate(self, ctx: ExecutionContext, barcode: str) -> dict[str, Any]:
        slot = next((s for s, b in self.stored.items() if b == barcode), None)
        if slot is None:
            raise ConstraintViolation(
                f"no plate {barcode!r} in this incubator; it holds "
                f"{sorted(self.stored.values()) or 'nothing'}",
                barcode=barcode, contents=sorted(self.stored.values()),
            )
        if not self.door_open:
            await self._cmd_open_door(ctx)
        await asyncio.sleep(0.2)
        async with _labware.BENCH.hold(barcode) as plate:
            del self.stored[slot]
            plate.location = "bench"
            incubated_for_s = round(plate.age_s(), 1)
            summary = plate.summary()
        await self._cmd_close_door(ctx)
        return {
            "retrieved": barcode, "slot": slot,
            "incubated_for_s": incubated_for_s,
            "plate": summary,
        }

    # -- prediction -------------------------------------------------------

    async def _simulate(
        self, feature: str, command: str, args: dict[str, Any]
    ) -> SimulationResult:
        self._advance()
        if command == "set_temperature":
            target = args.get("temperature_c", self.target_temperature_c)
            gap = abs(target - self.temperature_c)
            warnings = []
            if self.stored and target > 42.0:
                warnings.append(
                    f"{len(self.stored)} plate(s) are inside and {target:.1f} degC will kill "
                    "most mammalian cultures"
                )
            if self.stored and target < 15.0:
                warnings.append(
                    f"{len(self.stored)} plate(s) are inside and {target:.1f} degC will arrest "
                    "growth"
                )
            return SimulationResult(
                feasible=True, fidelity="reduced_order",
                predicted_state={"temperature_c": target},
                predicted_duration_s=3 * TAU_TEMPERATURE_S / self.time_scale,
                warnings=warnings,
                notes=f"first-order approach, tau={TAU_TEMPERATURE_S:.0f}s; "
                      f"{gap:.1f} degC to cover",
            )
        if command == "open_door":
            warnings = []
            if self.stored:
                warnings.append(
                    f"{len(self.stored)} plate(s) will be exposed to room air and the "
                    "chamber will need several minutes to recover"
                )
            return SimulationResult(
                feasible=True, fidelity="reduced_order",
                predicted_state={"temperature_c": round(AMBIENT_C, 1)},
                predicted_duration_s=1.0, warnings=warnings,
            )
        if command == "store_plate":
            violations = []
            if len(self.stored) >= self.slots:
                violations.append(f"all {self.slots} slots are occupied")
            barcode = args.get("barcode", "")
            try:
                plate = _labware.BENCH.get(barcode)
            except KeyError as exc:
                violations.append(str(exc))
            else:
                # Matches the check `_cmd_store_plate` actually enforces at
                # runtime -- a plate already loaded in a reader, or in
                # another incubator, cannot also be feasible to store here.
                if plate.location not in ("bench", self.id):
                    violations.append(
                        f"plate {barcode!r} is at {plate.location!r}, not on the bench"
                    )
            warnings = []
            if not self.at_setpoint:
                warnings.append(
                    f"the chamber is at {self.temperature_c:.1f} degC against a "
                    f"{self.target_temperature_c:.1f} degC setpoint; the plate will not be "
                    "incubated at the intended temperature until it settles"
                )
            return SimulationResult(
                feasible=not violations, fidelity="kinematic",
                predicted_state={"occupied_slots": len(self.stored) + 1},
                predicted_duration_s=6.0, violations=violations, warnings=warnings,
            )
        return SimulationResult(feasible=True, fidelity="none",
                                warnings=[f"no predictive model for {feature}.{command}"])
