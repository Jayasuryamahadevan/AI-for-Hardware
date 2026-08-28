"""A simulated multi-mode microplate reader.

Reads absorbance, fluorescence and luminescence off plates from the shared
bench, so what it measures is what the liquid handler actually dispensed. A
reader that invented its own numbers would let an agent "confirm" a dilution
series it had never made.

The optics are shallow but honest: absorbance follows Beer-Lambert through the
real path length implied by the well volume, fluorescence is proportional to
the amount of fluorophore present and to the spectral overlap between the
chosen filters and the dye, and every channel carries detector noise. So a
badly chosen filter pair reads near zero, an empty well reads background, and
an overfilled well reads a longer path than the user expected.
"""

from __future__ import annotations

import asyncio
import csv
import math
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ...core.capability import (
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
from ...core.device import Device, DeviceDescriptor, ExecutionContext, SimulationResult
from ...core.errors import ConstraintViolation, DeviceNotReady
from . import _labware

#: Well cross-sectional area by format, in mm^2, used to turn a volume into a
#: path length. Absorbance that ignored fill volume would be a fiction.
_WELL_AREA_MM2 = {"6": 950.0, "24": 190.0, "96": 32.0, "384": 8.5, "1536": 2.2}


class SimulatedPlateReader(Device):
    """Absorbance / fluorescence / luminescence plate reader with incubation."""

    def __init__(self, descriptor: DeviceDescriptor, **config: Any) -> None:
        super().__init__(descriptor, **config)
        self.data_dir = Path(config.get("data_dir", "./labbench-data/reads")).expanduser()
        self.rng = np.random.default_rng(int(config.get("seed", 11)))
        self.drawer_open = False
        self.loaded_barcode: str | None = None
        self.temperature_c = 22.0
        self.target_temperature_c = 22.0
        self.shaking = False
        self.reads_performed = 0

    # -- capabilities -----------------------------------------------------

    def _features(self) -> Sequence[Feature]:
        loaded = Precondition(
            property="loaded_barcode", operator="!=", value=None,
            message="no plate is loaded; call PlateTransport.load_plate first",
        )
        return [
            Feature(
                identifier="PlateTransport",
                display_name="Plate Drawer",
                namespace="org.labbench.handling",
                properties=[
                    Property(name="drawer_open", description="Drawer position.",
                             schema=Parameter(name="drawer_open", type="boolean")),
                    Property(name="loaded_barcode",
                             description="Barcode of the plate in the read chamber, or null.",
                             schema=Parameter(name="loaded_barcode", type="string",
                                              required=False)),
                ],
                commands=[
                    Command(name="open_drawer", description="Extend the plate drawer.",
                            duration_estimate_s=2.0, hazard=Hazard.MOTION,
                            reversibility=Reversibility.REVERSIBLE, inverse="close_drawer"),
                    Command(name="close_drawer", description="Retract the plate drawer.",
                            duration_estimate_s=2.0, hazard=Hazard.MOTION,
                            reversibility=Reversibility.REVERSIBLE, inverse="open_drawer"),
                    Command(
                        name="load_plate",
                        description="Take a plate from the bench into the read chamber.",
                        parameters=[Parameter(name="barcode", type="string",
                                              description="Barcode of a plate on the bench.")],
                        duration_estimate_s=4.0, hazard=Hazard.MOTION,
                        reversibility=Reversibility.REVERSIBLE, inverse="eject_plate",
                        tags={"moves_labware"},
                    ),
                    Command(name="eject_plate", description="Return the plate to the bench.",
                            duration_estimate_s=4.0, hazard=Hazard.MOTION,
                            reversibility=Reversibility.REVERSIBLE, inverse="load_plate",
                            tags={"moves_labware"}),
                ],
            ),
            Feature(
                identifier="Reader",
                display_name="Optical Detector",
                namespace="org.labbench.detection",
                properties=[
                    Property(name="reads_performed", description="Reads since connect.",
                             schema=Parameter(name="reads_performed", type="integer")),
                ],
                commands=[
                    Command(
                        name="read_absorbance",
                        description="Measure optical density at one wavelength. "
                                    "Beer-Lambert through the actual fill height.",
                        parameters=[
                            Parameter(name="wavelength_nm", unit="nm", default=600.0,
                                      required=False,
                                      constraint=Constraint(minimum=220.0, maximum=1000.0)),
                            Parameter(name="wells", type="array", required=False,
                                      description="Wells to read, e.g. ['A1','A2']. "
                                                  "Omit to read every well.",
                                      items=Parameter(name="well", type="string")),
                        ],
                        returns=[Parameter(name="artifact_uri", type="string"),
                                 Parameter(name="wells_read", type="integer")],
                        duration_estimate_s=12.0,
                        # Light through a plate changes nothing physical.
                        hazard=Hazard.NONE, reversibility=Reversibility.REVERSIBLE,
                        exclusive=True, preconditions=[loaded], tags={"produces_artifact"},
                    ),
                    Command(
                        name="read_fluorescence",
                        description="Measure fluorescence intensity. Signal depends on the "
                                    "overlap between the chosen filters and the dye's spectra.",
                        parameters=[
                            Parameter(name="excitation_nm", unit="nm", default=485.0,
                                      required=False,
                                      constraint=Constraint(minimum=230.0, maximum=900.0)),
                            Parameter(name="emission_nm", unit="nm", default=520.0,
                                      required=False,
                                      constraint=Constraint(minimum=250.0, maximum=950.0)),
                            Parameter(name="gain", default=60.0, required=False,
                                      description="Detector gain. Too high saturates.",
                                      constraint=Constraint(minimum=1.0, maximum=200.0)),
                            Parameter(name="wells", type="array", required=False,
                                      items=Parameter(name="well", type="string")),
                        ],
                        returns=[Parameter(name="artifact_uri", type="string"),
                                 Parameter(name="saturated_wells", type="integer")],
                        duration_estimate_s=15.0, hazard=Hazard.NONE,
                        preconditions=[loaded], tags={"produces_artifact"},
                    ),
                    Command(
                        name="read_kinetic",
                        description="Repeat a read on an interval. Long-running: returns a job "
                                    "handle with progress.",
                        parameters=[
                            Parameter(name="mode", type="string", default="fluorescence",
                                      required=False,
                                      constraint=Constraint(enum=["absorbance", "fluorescence"])),
                            Parameter(name="cycles", type="integer", default=6, required=False,
                                      constraint=Constraint(minimum=2, maximum=500)),
                            Parameter(name="interval_s", unit="s", default=30.0, required=False,
                                      constraint=Constraint(minimum=1.0, maximum=3600.0)),
                            Parameter(name="wells", type="array", required=False,
                                      items=Parameter(name="well", type="string")),
                        ],
                        returns=[Parameter(name="artifact_uri", type="string"),
                                 Parameter(name="cycles", type="integer")],
                        observable=True, duration_estimate_s=180.0,
                        # It occupies the plate for its whole duration, and the
                        # sample ages while it runs.
                        hazard=Hazard.SAMPLE, reversibility=Reversibility.IRREVERSIBLE,
                        preconditions=[loaded], tags={"produces_artifact", "long_running"},
                    ),
                ],
                events=[
                    Event(name="saturation", description="Wells exceeded the detector range.",
                          severity="warning"),
                ],
            ),
            Feature(
                identifier="TemperatureControl",
                display_name="Incubation",
                namespace="org.labbench.environment",
                properties=[
                    Property(name="temperature_c", description="Chamber temperature.",
                             schema=Parameter(name="temperature_c", unit="degC")),
                    Property(name="target_temperature_c", description="Setpoint.",
                             schema=Parameter(name="target_temperature_c", unit="degC"),
                             writable=True),
                ],
                commands=[
                    Command(
                        name="set_temperature",
                        description="Set the incubation setpoint. Approach is not instant.",
                        parameters=[Parameter(name="temperature_c", unit="degC",
                                              constraint=Constraint(minimum=18.0, maximum=45.0))],
                        duration_estimate_s=1.0, hazard=Hazard.THERMAL,
                        reversibility=Reversibility.RESTORABLE,
                    ),
                ],
            ),
            Feature(
                identifier="Shaker",
                display_name="Orbital Shaker",
                namespace="org.labbench.environment",
                properties=[
                    Property(name="shaking", description="Shaker running.",
                             schema=Parameter(name="shaking", type="boolean")),
                ],
                commands=[
                    Command(name="shake",
                            description="Orbital shake for a fixed duration.",
                            parameters=[
                                Parameter(name="duration_s", unit="s", default=10.0,
                                          required=False,
                                          constraint=Constraint(minimum=1.0, maximum=600.0)),
                                Parameter(name="amplitude_mm", unit="mm", default=2.0,
                                          required=False,
                                          constraint=Constraint(minimum=0.5, maximum=6.0)),
                            ],
                            observable=True, duration_estimate_s=10.0,
                            # Shaking a full plate splashes between wells.
                            hazard=Hazard.SAMPLE, reversibility=Reversibility.IRREVERSIBLE,
                            tags={"cross_contamination_risk"}),
                ],
            ),
        ]

    # -- lifecycle --------------------------------------------------------

    async def _connect(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.sleep(0.05)

    async def _estop(self) -> None:
        self.shaking = False

    async def _read(self, feature: str, name: str) -> Any:
        return {
            ("PlateTransport", "drawer_open"): self.drawer_open,
            ("PlateTransport", "loaded_barcode"): self.loaded_barcode,
            ("Reader", "reads_performed"): self.reads_performed,
            ("TemperatureControl", "temperature_c"): round(self.temperature_c, 2),
            ("TemperatureControl", "target_temperature_c"): self.target_temperature_c,
            ("Shaker", "shaking"): self.shaking,
        }[(feature, name)]

    async def _write(self, feature: str, name: str, value: Any) -> None:
        if (feature, name) == ("TemperatureControl", "target_temperature_c"):
            self.target_temperature_c = float(value)

    async def _invoke(
        self, feature: str, command: str, args: dict[str, Any], ctx: ExecutionContext
    ) -> Any:
        handler = getattr(self, f"_cmd_{command}")
        return await handler(ctx, **args)

    # -- plate transport --------------------------------------------------

    async def _cmd_open_drawer(self, ctx: ExecutionContext) -> dict[str, Any]:
        await asyncio.sleep(0.3)
        self.drawer_open = True
        return {"drawer_open": True}

    async def _cmd_close_drawer(self, ctx: ExecutionContext) -> dict[str, Any]:
        await asyncio.sleep(0.3)
        self.drawer_open = False
        return {"drawer_open": False}

    async def _cmd_load_plate(self, ctx: ExecutionContext, barcode: str) -> dict[str, Any]:
        if self.loaded_barcode is not None:
            raise DeviceNotReady(
                f"plate {self.loaded_barcode!r} is already loaded; eject it first",
                device=self.id, loaded=self.loaded_barcode,
            )
        try:
            plate = _labware.BENCH.get(barcode)
        except KeyError as exc:
            raise ConstraintViolation(str(exc), barcode=barcode) from None
        if plate.location not in ("bench", self.id):
            raise ConstraintViolation(
                f"plate {barcode!r} is not on the bench; it is at {plate.location!r}. "
                "Something else has it.",
                barcode=barcode, location=plate.location,
            )
        await asyncio.sleep(0.5)
        plate.location = self.id
        self.loaded_barcode = barcode
        self.drawer_open = False
        return {"loaded_barcode": barcode, "plate": plate.summary()}

    async def _cmd_eject_plate(self, ctx: ExecutionContext) -> dict[str, Any]:
        if self.loaded_barcode is None:
            raise DeviceNotReady("no plate is loaded", device=self.id)
        plate = _labware.BENCH.get(self.loaded_barcode)
        plate.location = "bench"
        barcode, self.loaded_barcode = self.loaded_barcode, None
        self.drawer_open = True
        await asyncio.sleep(0.5)
        return {"ejected": barcode}

    # -- reading ----------------------------------------------------------

    def _plate(self) -> _labware.Plate:
        if self.loaded_barcode is None:
            raise DeviceNotReady("no plate is loaded", device=self.id)
        return _labware.BENCH.get(self.loaded_barcode)

    def _select(self, plate: _labware.Plate, wells: list[str] | None) -> list[str]:
        if not wells:
            return sorted(plate.wells, key=_labware.parse_well)
        out = []
        for name in wells:
            try:
                out.append(plate.normalise(name))
            except ValueError as exc:
                raise ConstraintViolation(str(exc), well=name) from None
        return out

    def _path_length_mm(self, plate: _labware.Plate, well: _labware.Well) -> float:
        """Fill height in mm: volume (uL = mm^3) over the well cross-section."""
        area = _WELL_AREA_MM2.get(plate.format, 32.0)
        return well.volume_ul / area

    async def _cmd_read_absorbance(
        self, ctx: ExecutionContext, wavelength_nm: float = 600.0, wells: list[str] | None = None
    ) -> dict[str, Any]:
        plate = self._plate()
        names = self._select(plate, wells)
        rows = []
        for index, name in enumerate(names):
            ctx.raise_if_cancelled()
            well = plate.wells[name]
            od = self._absorbance(well, plate, wavelength_nm)
            rows.append({"well": name, "wavelength_nm": wavelength_nm, "od": round(od, 5),
                         "volume_ul": round(well.volume_ul, 2)})
            if index % 24 == 0:
                await ctx.progress((index + 1) / len(names), f"well {name}")
        await asyncio.sleep(min(0.3, len(names) * 0.002))
        self.reads_performed += 1
        artifact = self._write_table(rows, "absorbance", {"wavelength_nm": wavelength_nm})
        values = [r["od"] for r in rows]
        return {
            "artifact_uri": artifact["uri"], "artifacts": [artifact],
            "wells_read": len(rows), "mode": "absorbance",
            "wavelength_nm": wavelength_nm,
            "max_od": round(max(values), 4), "mean_od": round(float(np.mean(values)), 4),
        }

    def _absorbance(
        self, well: _labware.Well, plate: _labware.Plate, wavelength_nm: float
    ) -> float:
        path_cm = self._path_length_mm(plate, well) / 10.0
        od = 0.0
        for reagent, pmol in well.contents.items():
            epsilon, ex, _em, _qy = _labware.REAGENTS.get(reagent, (0.0, None, None, 0.0))
            if epsilon <= 0:
                continue
            # Absorbance peak: a crude Gaussian around the reagent's own peak,
            # so reading at the wrong wavelength genuinely under-reads.
            peak = ex if ex is not None else 520.0
            overlap = math.exp(-((wavelength_nm - peak) ** 2) / (2 * 45.0 ** 2))
            molar = (pmol * 1e-12) / max(well.volume_ul * 1e-6, 1e-12)  # mol/L
            od += epsilon * molar * path_cm * overlap
        if well.contents.get("cells", 0.0) > 0:
            # Cells scatter rather than absorb; this is the usual OD600 proxy.
            od += 0.0011 * well.contents["cells"] * path_cm
        return max(0.0, od + float(self.rng.normal(0.0, 0.0015)))

    async def _cmd_read_fluorescence(
        self,
        ctx: ExecutionContext,
        excitation_nm: float = 485.0,
        emission_nm: float = 520.0,
        gain: float = 60.0,
        wells: list[str] | None = None,
    ) -> dict[str, Any]:
        plate = self._plate()
        names = self._select(plate, wells)
        rows, saturated = [], 0
        for index, name in enumerate(names):
            ctx.raise_if_cancelled()
            rfu = self._fluorescence(plate.wells[name], excitation_nm, emission_nm, gain)
            if rfu >= 65535:
                saturated += 1
            rows.append({"well": name, "rfu": round(rfu, 1),
                         "volume_ul": round(plate.wells[name].volume_ul, 2)})
            if index % 24 == 0:
                await ctx.progress((index + 1) / len(names), f"well {name}")
        await asyncio.sleep(min(0.3, len(names) * 0.002))
        self.reads_performed += 1
        if saturated:
            await self.emit("Reader", "saturation",
                            {"wells": saturated, "gain": gain}, severity="warning")
        artifact = self._write_table(
            rows, "fluorescence",
            {"excitation_nm": excitation_nm, "emission_nm": emission_nm, "gain": gain},
        )
        values = [r["rfu"] for r in rows]
        return {
            "artifact_uri": artifact["uri"], "artifacts": [artifact],
            "wells_read": len(rows), "mode": "fluorescence",
            "saturated_wells": saturated,
            "max_rfu": round(max(values), 1), "mean_rfu": round(float(np.mean(values)), 1),
            "excitation_nm": excitation_nm, "emission_nm": emission_nm, "gain": gain,
        }

    def _fluorescence(
        self, well: _labware.Well, excitation_nm: float, emission_nm: float, gain: float
    ) -> float:
        signal = 0.0
        for reagent, pmol in well.contents.items():
            _eps, ex, em, qy = _labware.REAGENTS.get(reagent, (0.0, None, None, 0.0))
            if ex is None or em is None or qy <= 0:
                continue
            # Filter mismatch is the point: choosing 485/520 for rhodamine
            # (555/580) must read near nothing, exactly as on a real reader.
            ex_overlap = math.exp(-((excitation_nm - ex) ** 2) / (2 * 25.0 ** 2))
            em_overlap = math.exp(-((emission_nm - em) ** 2) / (2 * 30.0 ** 2))
            signal += pmol * qy * ex_overlap * em_overlap * 12.0
        background = 40.0 + well.volume_ul * 0.12
        rfu = (signal + background) * (gain / 60.0) ** 1.6
        rfu += float(self.rng.normal(0.0, 3.0 + rfu * 0.004))
        return float(min(max(rfu, 0.0), 65535.0))

    async def _cmd_read_kinetic(
        self,
        ctx: ExecutionContext,
        mode: str = "fluorescence",
        cycles: int = 6,
        interval_s: float = 30.0,
        wells: list[str] | None = None,
    ) -> dict[str, Any]:
        plate = self._plate()
        names = self._select(plate, wells)
        started = time.time()
        rows: list[dict[str, Any]] = []
        for cycle in range(cycles):
            ctx.raise_if_cancelled()
            # The plate ages between cycles, which is the whole reason a
            # kinetic read is hazard SAMPLE rather than a free observation.
            if cycle:
                plate.apply_evaporation(interval_s, self.temperature_c)
            elapsed = round(time.time() - started, 2)
            for name in names:
                well = plate.wells[name]
                value = (
                    self._fluorescence(well, 485.0, 520.0, 60.0)
                    if mode == "fluorescence"
                    else self._absorbance(well, plate, 600.0)
                )
                rows.append({"cycle": cycle + 1, "elapsed_s": elapsed, "well": name,
                             "value": round(value, 4)})
            await ctx.progress((cycle + 1) / cycles, f"cycle {cycle + 1}/{cycles}")
            if cycle < cycles - 1:
                # Compressed: a real interval would make a demo unusable, and
                # the shape of the curve is what matters.
                await asyncio.sleep(min(interval_s, 0.15))
        self.reads_performed += cycles
        artifact = self._write_table(
            rows, f"kinetic_{mode}", {"cycles": cycles, "interval_s": interval_s}
        )
        return {
            "artifact_uri": artifact["uri"], "artifacts": [artifact],
            "cycles": cycles, "wells_read": len(names), "mode": mode,
            "rows": len(rows),
        }

    # -- environment ------------------------------------------------------

    async def _cmd_set_temperature(
        self, ctx: ExecutionContext, temperature_c: float
    ) -> dict[str, Any]:
        self.target_temperature_c = float(temperature_c)
        # First-order approach, not a jump. An agent that reads the temperature
        # immediately must see that it has not arrived yet.
        self.temperature_c += (self.target_temperature_c - self.temperature_c) * 0.15
        return {"target_temperature_c": self.target_temperature_c,
                "temperature_c": round(self.temperature_c, 2),
                "at_setpoint": abs(self.temperature_c - self.target_temperature_c) < 0.3}

    async def _cmd_shake(
        self, ctx: ExecutionContext, duration_s: float = 10.0, amplitude_mm: float = 2.0
    ) -> dict[str, Any]:
        plate = self._plate()
        self.shaking = True
        try:
            steps = max(1, int(min(duration_s, 8)))
            for i in range(steps):
                ctx.raise_if_cancelled()
                await ctx.progress((i + 1) / steps, f"shaking {i + 1}/{steps}s")
                await asyncio.sleep(0.05)
        finally:
            self.shaking = False
        # Splash risk rises with amplitude and fill. This is why shaking is
        # tagged cross_contamination_risk rather than treated as free.
        overfull = [
            name for name, w in plate.occupied().items()
            if w.volume_ul > plate.working_volume_ul * 0.85
        ]
        splashed = bool(overfull) and amplitude_mm > 3.0
        if splashed:
            await self.emit(
                "Shaker", "splash",
                {"wells": overfull[:8], "amplitude_mm": amplitude_mm}, severity="warning",
            )
        return {
            "duration_s": duration_s, "amplitude_mm": amplitude_mm,
            "wells_near_capacity": len(overfull),
            "splash_risk": splashed,
        }

    # -- artifacts --------------------------------------------------------

    def _write_table(
        self, rows: list[dict[str, Any]], kind: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Write a read as CSV.

        CSV rather than a proprietary format because the next thing that
        happens to plate data is that a person opens it in a spreadsheet.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        name = f"{self.id}_{kind}_{stamp}_{uuid.uuid4().hex[:6]}.csv"
        path = self.data_dir / name
        with path.open("w", newline="", encoding="utf-8") as fh:
            if rows:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return {
            "uri": path.resolve().as_uri(), "kind": "table", "mime_type": "text/csv",
            "bytes": path.stat().st_size, "shape": [len(rows), len(rows[0]) if rows else 0],
            "metadata": {"device": self.id, "plate": self.loaded_barcode,
                         "read_kind": kind, "simulated": True, **metadata},
        }

    # -- prediction -------------------------------------------------------

    async def _simulate(
        self, feature: str, command: str, args: dict[str, Any]
    ) -> SimulationResult:
        if command in ("read_absorbance", "read_fluorescence", "read_kinetic"):
            if self.loaded_barcode is None:
                return SimulationResult(
                    feasible=False, fidelity="kinematic",
                    violations=["no plate is loaded"],
                )
            plate = _labware.BENCH.get(self.loaded_barcode)
            warnings = []
            if not plate.occupied():
                warnings.append(
                    "every well is empty; this read will return background only"
                )
            if command == "read_fluorescence":
                ex = args.get("excitation_nm", 485.0)
                em = args.get("emission_nm", 520.0)
                present = {r for w in plate.occupied().values() for r in w.contents}
                fluorophores = [
                    r for r in present
                    if _labware.REAGENTS.get(r, (0, None, None, 0))[1] is not None
                ]
                matched = [
                    r for r in fluorophores
                    if abs(_labware.REAGENTS[r][1] - ex) < 40
                    and abs(_labware.REAGENTS[r][2] - em) < 45
                ]
                if fluorophores and not matched:
                    warnings.append(
                        f"filters {ex:.0f}/{em:.0f} nm do not match any fluorophore on the "
                        f"plate ({', '.join(sorted(fluorophores))}); the read will be blank"
                    )
                if args.get("gain", 60.0) > 140:
                    warnings.append("gain above 140 will saturate most wells")
            cycles = args.get("cycles", 1)
            return SimulationResult(
                feasible=True, fidelity="reduced_order",
                predicted_state={"wells": len(plate.occupied())},
                predicted_duration_s=cycles * (len(plate.wells) * 0.02 + 1.0),
                warnings=warnings,
                notes="predicted from the plate's actual contents",
            )
        if command == "load_plate":
            barcode = args.get("barcode", "")
            try:
                plate = _labware.BENCH.get(barcode)
            except KeyError as exc:
                return SimulationResult(feasible=False, fidelity="kinematic",
                                        violations=[str(exc)])
            violations = []
            if self.loaded_barcode is not None:
                violations.append(f"plate {self.loaded_barcode!r} is already loaded")
            if plate.location not in ("bench", self.id):
                violations.append(f"plate is at {plate.location!r}, not on the bench")
            return SimulationResult(feasible=not violations, fidelity="kinematic",
                                    predicted_state={"loaded_barcode": barcode},
                                    violations=violations)
        return SimulationResult(feasible=True, fidelity="none",
                                warnings=[f"no predictive model for {feature}.{command}"])
