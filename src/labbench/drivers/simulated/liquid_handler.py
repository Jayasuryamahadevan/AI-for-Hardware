"""A simulated eight-channel liquid handler.

The instrument most able to quietly ruin an experiment, and therefore the one
worth simulating carefully. It works on plates from the shared bench, so what
it dispenses is what the plate reader will later measure.

What is modelled, and why each one earns its place:

**Tips have state.** You cannot aspirate without them, you carry residue
between wells if you do not change them, and the residue is tracked as real
amounts so a cross-contaminated control reads wrong instead of merely being
flagged. Tip reuse is the classic way a dilution series comes out subtly wrong.

**Volume is conserved.** Aspirating more than a well holds is refused, as is
dispensing past a well's working volume. A handler that let an agent overfill a
well would hide the overflow that in reality runs across the plate seal.

**Accuracy depends on volume.** Small volumes are proportionally less accurate,
which is why a 1 uL transfer is a bad way to build a dilution series and a
5 uL one is acceptable. The error is systematic plus random, as on real
hardware.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import numpy as np

from ...core.capability import (
    Command, Constraint, Event, Feature, Hazard, Parameter, Precondition,
    Property, Reversibility,
)
from ...core.device import Device, DeviceDescriptor, ExecutionContext, SimulationResult
from ...core.errors import ConstraintViolation, DeviceNotReady
from . import _labware

#: Channel count for a standard multichannel head.
CHANNELS = 8
#: Pipette range. Below the minimum the error is unacceptable; the driver
#: refuses rather than silently delivering something else.
MIN_VOLUME_UL = 0.5
MAX_VOLUME_UL = 300.0


class SimulatedLiquidHandler(Device):
    """Eight-channel pipetting robot with a tip deck and reagent troughs."""

    def __init__(self, descriptor: DeviceDescriptor, **config: Any) -> None:
        super().__init__(descriptor, **config)
        self.rng = np.random.default_rng(int(config.get("seed", 13)))
        self.has_tips = False
        self.tips_used = 0
        self.tip_boxes = int(config.get("tip_boxes", 4))
        self.tips_remaining = self.tip_boxes * 96
        #: What is currently held in the tips, in pmol, plus its volume.
        self.tip_volume_ul = 0.0
        self.tip_contents: dict[str, float] = {}
        #: Residue left on the outside/inside of a used tip.
        self.tip_residue: dict[str, float] = {}
        self.transfers = 0
        self.volume_dispensed_ul = 0.0
        #: Bulk reagent troughs, effectively unlimited.
        self.troughs: dict[str, float] = dict(
            config.get("troughs", {"water": 1e9, "buffer": 1e9})
        )

    # -- capabilities -----------------------------------------------------

    def _features(self) -> Sequence[Feature]:
        tips_on = Precondition(
            property="has_tips", operator="is_true",
            message="no tips are fitted; call TipManagement.pick_up_tips first",
        )
        volume = Constraint(minimum=MIN_VOLUME_UL, maximum=MAX_VOLUME_UL)
        return [
            Feature(
                identifier="TipManagement",
                display_name="Tip Handling",
                namespace="org.labbench.liquid",
                properties=[
                    Property(name="has_tips", description="Tips currently fitted.",
                             schema=Parameter(name="has_tips", type="boolean")),
                    Property(name="tips_remaining", description="Unused tips on the deck.",
                             schema=Parameter(name="tips_remaining", type="integer")),
                    Property(name="tips_used", description="Tips consumed since connect.",
                             schema=Parameter(name="tips_used", type="integer")),
                    Property(name="tip_contamination",
                             description="Reagents left as residue on the fitted tips. "
                                         "Non-empty means the next aspirate carries them over.",
                             schema=Parameter(name="tip_contamination", type="object")),
                ],
                commands=[
                    Command(
                        name="pick_up_tips",
                        description="Fit fresh tips. Any fitted tips are dropped first.",
                        duration_estimate_s=3.0, hazard=Hazard.MOTION,
                        reversibility=Reversibility.RESTORABLE, inverse="drop_tips",
                        tags={"consumes_tips"},
                    ),
                    Command(
                        name="drop_tips",
                        description="Eject tips to waste. Discards anything still held.",
                        duration_estimate_s=2.0, hazard=Hazard.MOTION,
                        reversibility=Reversibility.IRREVERSIBLE,
                        tags={"consumes_tips"},
                    ),
                ],
                events=[
                    Event(name="tips_exhausted", description="No tips remain on the deck.",
                          severity="error"),
                    Event(name="carryover",
                          description="A transfer was made with contaminated tips.",
                          severity="warning"),
                ],
            ),
            Feature(
                identifier="Pipette",
                display_name="8-Channel Pipette",
                namespace="org.labbench.liquid",
                properties=[
                    Property(name="tip_volume_ul", description="Volume currently held in tips.",
                             schema=Parameter(name="tip_volume_ul", unit="uL")),
                    Property(name="transfers", description="Transfers since connect.",
                             schema=Parameter(name="transfers", type="integer")),
                    Property(name="volume_dispensed_ul", description="Cumulative dispensed.",
                             schema=Parameter(name="volume_dispensed_ul", unit="uL")),
                ],
                commands=[
                    Command(
                        name="aspirate",
                        description="Draw liquid from a well or a bulk trough into the tips.",
                        parameters=[
                            Parameter(name="volume_ul", unit="uL", constraint=volume,
                                      description="Volume to draw."),
                            Parameter(name="plate", type="string", required=False,
                                      description="Source plate barcode. Omit when using a trough."),
                            Parameter(name="well", type="string", required=False,
                                      description="Source well, e.g. 'A1'."),
                            Parameter(name="trough", type="string", required=False,
                                      description="Bulk reagent name, e.g. 'buffer'."),
                        ],
                        returns=[Parameter(name="tip_volume_ul", unit="uL")],
                        duration_estimate_s=3.0, hazard=Hazard.SAMPLE,
                        reversibility=Reversibility.IRREVERSIBLE,
                        preconditions=[tips_on], tags={"moves_liquid"},
                    ),
                    Command(
                        name="dispense",
                        description="Deliver liquid from the tips into a well.",
                        parameters=[
                            Parameter(name="volume_ul", unit="uL", constraint=volume),
                            Parameter(name="plate", type="string", description="Destination plate."),
                            Parameter(name="well", type="string", description="Destination well."),
                        ],
                        returns=[Parameter(name="delivered_ul", unit="uL")],
                        duration_estimate_s=3.0, hazard=Hazard.SAMPLE,
                        reversibility=Reversibility.IRREVERSIBLE,
                        preconditions=[tips_on], tags={"moves_liquid"},
                    ),
                    Command(
                        name="transfer",
                        description="Aspirate and dispense in one operation, optionally "
                                    "changing tips between each pair to avoid carryover.",
                        parameters=[
                            Parameter(name="volume_ul", unit="uL", constraint=volume),
                            Parameter(name="source_plate", type="string", required=False),
                            Parameter(name="source_well", type="string", required=False),
                            Parameter(name="source_trough", type="string", required=False),
                            Parameter(name="dest_plate", type="string"),
                            Parameter(name="dest_wells", type="array",
                                      description="One or more destination wells.",
                                      items=Parameter(name="well", type="string")),
                            Parameter(name="new_tips_each", type="boolean", default=True,
                                      required=False,
                                      description="Fit fresh tips before every destination. "
                                                  "Setting this false reuses tips and will "
                                                  "carry reagent between wells."),
                        ],
                        returns=[Parameter(name="dispensed", type="integer")],
                        observable=True, duration_estimate_s=20.0,
                        hazard=Hazard.SAMPLE, reversibility=Reversibility.IRREVERSIBLE,
                        preconditions=[], tags={"moves_liquid", "consumes_tips"},
                    ),
                    Command(
                        name="mix",
                        description="Aspirate and re-dispense in place to homogenise a well.",
                        parameters=[
                            Parameter(name="plate", type="string"),
                            Parameter(name="well", type="string"),
                            Parameter(name="volume_ul", unit="uL", constraint=volume),
                            Parameter(name="cycles", type="integer", default=3, required=False,
                                      constraint=Constraint(minimum=1, maximum=20)),
                        ],
                        duration_estimate_s=6.0, hazard=Hazard.SAMPLE,
                        reversibility=Reversibility.IRREVERSIBLE,
                        preconditions=[tips_on], tags={"moves_liquid"},
                    ),
                ],
                events=[
                    Event(name="overflow", description="A dispense would exceed a well's capacity.",
                          severity="error"),
                ],
            ),
            Feature(
                identifier="Labware",
                display_name="Deck",
                namespace="org.labbench.handling",
                properties=[
                    Property(name="plates", description="Plates on the bench, by barcode.",
                             schema=Parameter(name="plates", type="array",
                                              items=Parameter(name="barcode", type="string"))),
                    Property(name="troughs", description="Bulk reagents available.",
                             schema=Parameter(name="troughs", type="array",
                                              items=Parameter(name="reagent", type="string"))),
                ],
                commands=[
                    Command(
                        name="create_plate",
                        description="Put a fresh, empty plate on the bench.",
                        parameters=[
                            Parameter(name="barcode", type="string"),
                            Parameter(name="plate_format", type="string", default="96",
                                      required=False,
                                      constraint=Constraint(enum=sorted(_labware.FORMATS))),
                            Parameter(name="label", type="string", required=False, default=""),
                        ],
                        duration_estimate_s=1.0, hazard=Hazard.BENIGN,
                        reversibility=Reversibility.RESTORABLE,
                    ),
                    Command(
                        name="inspect_plate",
                        description="Report every non-empty well: volume and concentrations.",
                        parameters=[Parameter(name="barcode", type="string")],
                        duration_estimate_s=0.2,
                        # Reading the model is free; it touches nothing.
                        hazard=Hazard.NONE, reversibility=Reversibility.REVERSIBLE,
                        exclusive=False,
                    ),
                ],
            ),
        ]

    # -- lifecycle --------------------------------------------------------

    async def _connect(self) -> None:
        await asyncio.sleep(0.05)

    async def _estop(self) -> None:
        """Stop the head. Tips stay on, and whatever is in them stays in them:
        blowing it out somewhere unplanned would be worse than holding it."""
        return None

    async def _read(self, feature: str, name: str) -> Any:
        return {
            ("TipManagement", "has_tips"): self.has_tips,
            ("TipManagement", "tips_remaining"): self.tips_remaining,
            ("TipManagement", "tips_used"): self.tips_used,
            ("TipManagement", "tip_contamination"): {
                k: round(v, 6) for k, v in self.tip_residue.items()
            },
            ("Pipette", "tip_volume_ul"): round(self.tip_volume_ul, 3),
            ("Pipette", "transfers"): self.transfers,
            ("Pipette", "volume_dispensed_ul"): round(self.volume_dispensed_ul, 3),
            ("Labware", "plates"): sorted(_labware.BENCH.all()),
            ("Labware", "troughs"): sorted(self.troughs),
        }[(feature, name)]

    async def _invoke(
        self, feature: str, command: str, args: dict[str, Any], ctx: ExecutionContext
    ) -> Any:
        return await getattr(self, f"_cmd_{command}")(ctx, **args)

    # -- tips -------------------------------------------------------------

    async def _cmd_pick_up_tips(self, ctx: ExecutionContext) -> dict[str, Any]:
        if self.tips_remaining < CHANNELS:
            await self.emit("TipManagement", "tips_exhausted",
                            {"remaining": self.tips_remaining}, severity="error")
            raise DeviceNotReady(
                f"only {self.tips_remaining} tips remain on the deck and a pick-up needs "
                f"{CHANNELS}; a human must reload the tip boxes",
                device=self.id, remaining=self.tips_remaining,
            )
        if self.has_tips:
            await self._cmd_drop_tips(ctx)
        await asyncio.sleep(0.15)
        self.has_tips = True
        self.tips_remaining -= CHANNELS
        self.tips_used += CHANNELS
        self.tip_residue = {}
        self.tip_volume_ul = 0.0
        self.tip_contents = {}
        return {"has_tips": True, "tips_remaining": self.tips_remaining}

    async def _cmd_drop_tips(self, ctx: ExecutionContext) -> dict[str, Any]:
        discarded = round(self.tip_volume_ul, 3)
        await asyncio.sleep(0.1)
        self.has_tips = False
        self.tip_volume_ul = 0.0
        self.tip_contents = {}
        self.tip_residue = {}
        return {"has_tips": False, "discarded_ul": discarded}

    # -- pipetting --------------------------------------------------------

    def _plate(self, barcode: str) -> _labware.Plate:
        try:
            return _labware.BENCH.get(barcode)
        except KeyError as exc:
            raise ConstraintViolation(str(exc), barcode=barcode) from None

    def _delivered(self, requested_ul: float) -> float:
        """Actual volume for a requested one.

        Systematic bias plus random scatter, both worse proportionally at small
        volumes. This is why a serial dilution built from 1 uL steps drifts and
        one built from 20 uL steps does not.
        """
        relative_error = 0.008 + 0.05 / max(requested_ul, 0.5)
        bias = -0.004 * requested_ul
        return max(0.0, requested_ul + bias + float(self.rng.normal(0.0, requested_ul * relative_error)))

    async def _absorb_residue(
        self, contents: dict[str, float], source: str
    ) -> dict[str, float]:
        """Fold anything left on the tips into what was just drawn.

        This must happen on *every* aspiration path, not only when drawing from
        a well. Drawing fresh buffer from a bulk trough with dirty tips is the
        classic way a control well ends up containing the thing it is supposed
        to be a control for, and a model that cleaned the tips on the way to
        the trough would hide exactly that.
        """
        if not self.tip_residue:
            return contents
        await self.emit(
            "TipManagement", "carryover",
            {"reagents": sorted(self.tip_residue), "into": source,
             "pmol": {k: round(v, 6) for k, v in self.tip_residue.items()}},
            severity="warning",
        )
        for reagent, pmol in self.tip_residue.items():
            contents[reagent] = contents.get(reagent, 0.0) + pmol
        self.tip_residue = {}
        return contents

    async def _cmd_aspirate(
        self,
        ctx: ExecutionContext,
        volume_ul: float,
        plate: str | None = None,
        well: str | None = None,
        trough: str | None = None,
    ) -> dict[str, Any]:
        if self.tip_volume_ul > 0:
            raise DeviceNotReady(
                f"the tips already hold {self.tip_volume_ul:.2f} uL; dispense it before "
                "aspirating again",
                device=self.id, tip_volume_ul=self.tip_volume_ul,
            )
        if trough is not None:
            if trough not in self.troughs:
                raise ConstraintViolation(
                    f"no trough {trough!r}; available: {sorted(self.troughs)}",
                    trough=trough, available=sorted(self.troughs),
                )
            drawn = self._delivered(volume_ul)
            self.tip_volume_ul = drawn
            # A bulk trough of buffer is buffer: one reagent, at unit activity.
            contents = {trough: drawn} if trough not in ("water",) else {}
            self.tip_contents = await self._absorb_residue(contents, f"trough:{trough}")
            await asyncio.sleep(0.12)
            return {"tip_volume_ul": round(drawn, 3), "source": f"trough:{trough}"}

        if plate is None or well is None:
            raise ConstraintViolation(
                "aspirate needs either a trough, or both a plate and a well",
                given={"plate": plate, "well": well, "trough": trough},
            )
        target = self._plate(plate)
        try:
            source = target.well(well)
        except ValueError as exc:
            raise ConstraintViolation(str(exc), well=well) from None
        available = source.volume_ul - target.dead_volume_ul
        if volume_ul > available:
            raise ConstraintViolation(
                f"{well} holds {source.volume_ul:.2f} uL, of which {target.dead_volume_ul:.1f} uL "
                f"is dead volume the pipette cannot reach; at most {max(0.0, available):.2f} uL "
                f"is available and {volume_ul:.2f} uL was requested",
                well=well, volume_ul=source.volume_ul,
                dead_volume_ul=target.dead_volume_ul, available_ul=max(0.0, available),
            )
        drawn = self._delivered(volume_ul)
        moved = source.remove(drawn)
        self.tip_volume_ul = drawn
        self.tip_contents = await self._absorb_residue(moved, f"{plate}:{well}")
        await asyncio.sleep(0.12)
        return {
            "tip_volume_ul": round(drawn, 3),
            "requested_ul": volume_ul,
            "source": f"{plate}:{target.normalise(well)}",
            "source_remaining_ul": round(source.volume_ul, 3),
        }

    async def _cmd_dispense(
        self, ctx: ExecutionContext, volume_ul: float, plate: str, well: str
    ) -> dict[str, Any]:
        target = self._plate(plate)
        try:
            destination = target.well(well)
        except ValueError as exc:
            raise ConstraintViolation(str(exc), well=well) from None
        # A pipette delivers what it actually drew. Aspirating a nominal 50 uL
        # leaves something like 49.8 uL in the tip, and refusing the matching
        # 50 uL dispense over that would model nothing real - it would just
        # make every aspirate/dispense pair fail. So a shortfall inside the
        # pipette's own tolerance delivers what is held and reports it; a
        # genuine over-request still fails, because that is a planning error.
        tolerance = max(0.5, volume_ul * 0.03)
        if volume_ul > self.tip_volume_ul + tolerance:
            raise ConstraintViolation(
                f"the tips hold {self.tip_volume_ul:.2f} uL but {volume_ul:.2f} uL was "
                f"requested, which is beyond the {tolerance:.2f} uL delivery tolerance",
                held_ul=self.tip_volume_ul, requested_ul=volume_ul,
                tolerance_ul=round(tolerance, 3),
            )
        requested_ul = volume_ul
        volume_ul = min(volume_ul, self.tip_volume_ul)
        if destination.volume_ul + volume_ul > target.working_volume_ul:
            await self.emit(
                "Pipette", "overflow",
                {"well": well, "current_ul": destination.volume_ul,
                 "requested_ul": volume_ul, "capacity_ul": target.working_volume_ul},
                severity="error",
            )
            raise ConstraintViolation(
                f"{well} holds {destination.volume_ul:.2f} uL and the working volume of a "
                f"{target.format}-well plate is {target.working_volume_ul:.0f} uL; adding "
                f"{volume_ul:.2f} uL would overflow it onto the plate seal",
                well=well, current_ul=destination.volume_ul,
                capacity_ul=target.working_volume_ul, requested_ul=volume_ul,
            )
        fraction = volume_ul / self.tip_volume_ul if self.tip_volume_ul > 0 else 0.0
        moved = {r: pmol * fraction for r, pmol in self.tip_contents.items()}
        destination.add(volume_ul, moved)
        for reagent in list(self.tip_contents):
            self.tip_contents[reagent] -= moved[reagent]
            if self.tip_contents[reagent] <= 1e-12:
                del self.tip_contents[reagent]
        self.tip_volume_ul -= volume_ul
        self.volume_dispensed_ul += volume_ul
        # Residue: about 1% of what was carried stays on the tip walls.
        if self.tip_volume_ul <= 1e-9:
            self.tip_residue = {r: pmol * 0.01 for r, pmol in moved.items() if pmol > 0}
            self.tip_contents = {}
            self.tip_volume_ul = 0.0
        target.last_touched = asyncio.get_running_loop().time()
        await asyncio.sleep(0.12)
        return {
            "delivered_ul": round(volume_ul, 3),
            "requested_ul": round(requested_ul, 3),
            "destination": f"{plate}:{target.normalise(well)}",
            "well_volume_ul": round(destination.volume_ul, 3),
            "tip_volume_ul": round(self.tip_volume_ul, 3),
        }

    async def _cmd_transfer(
        self,
        ctx: ExecutionContext,
        volume_ul: float,
        dest_plate: str,
        dest_wells: list[str],
        source_plate: str | None = None,
        source_well: str | None = None,
        source_trough: str | None = None,
        new_tips_each: bool = True,
    ) -> dict[str, Any]:
        if not dest_wells:
            raise ConstraintViolation("dest_wells must name at least one well")
        dispensed, carryover_wells = [], []
        for index, well in enumerate(dest_wells):
            ctx.raise_if_cancelled()
            if new_tips_each or not self.has_tips:
                await self._cmd_pick_up_tips(ctx)
            elif self.tip_residue:
                carryover_wells.append(well)
            await self._cmd_aspirate(
                ctx, volume_ul=volume_ul, plate=source_plate,
                well=source_well, trough=source_trough,
            )
            result = await self._cmd_dispense(
                ctx, volume_ul=min(volume_ul, self.tip_volume_ul),
                plate=dest_plate, well=well,
            )
            dispensed.append(result["delivered_ul"])
            await ctx.progress(
                (index + 1) / len(dest_wells), f"{index + 1}/{len(dest_wells)} -> {well}"
            )
        self.transfers += 1
        return {
            "dispensed": len(dispensed),
            "total_ul": round(sum(dispensed), 3),
            "mean_ul": round(float(np.mean(dispensed)), 3),
            "cv_pct": round(float(np.std(dispensed) / max(np.mean(dispensed), 1e-9) * 100), 2),
            "new_tips_each": new_tips_each,
            "wells_with_carryover": carryover_wells,
        }

    async def _cmd_mix(
        self, ctx: ExecutionContext, plate: str, well: str, volume_ul: float, cycles: int = 3
    ) -> dict[str, Any]:
        target = self._plate(plate)
        try:
            destination = target.well(well)
        except ValueError as exc:
            raise ConstraintViolation(str(exc), well=well) from None
        usable = destination.volume_ul - target.dead_volume_ul
        if volume_ul > usable:
            raise ConstraintViolation(
                f"cannot mix {volume_ul:.1f} uL in {well}: only {max(0.0, usable):.2f} uL is "
                "reachable above the dead volume",
                well=well, available_ul=max(0.0, usable),
            )
        for cycle in range(cycles):
            ctx.raise_if_cancelled()
            await ctx.progress((cycle + 1) / cycles, f"mix cycle {cycle + 1}/{cycles}")
            await asyncio.sleep(0.04)
        # Mixing homogenises, which the well model already assumes; what it
        # really does here is leave residue on the tips.
        self.tip_residue = {
            r: pmol * 0.01 for r, pmol in destination.contents.items() if pmol > 0
        }
        return {"well": target.normalise(well), "cycles": cycles,
                "volume_ul": volume_ul, "well_volume_ul": round(destination.volume_ul, 3)}

    # -- labware ----------------------------------------------------------

    async def _cmd_create_plate(
        self, ctx: ExecutionContext, barcode: str, plate_format: str = "96", label: str = ""
    ) -> dict[str, Any]:
        try:
            plate = _labware.BENCH.create(barcode, plate_format=plate_format, label=label)
        except ValueError as exc:
            raise ConstraintViolation(str(exc), barcode=barcode) from None
        return plate.summary()

    async def _cmd_inspect_plate(self, ctx: ExecutionContext, barcode: str) -> dict[str, Any]:
        plate = self._plate(barcode)
        return {**plate.summary(), "wells": plate.well_table()}

    # -- prediction -------------------------------------------------------

    async def _simulate(
        self, feature: str, command: str, args: dict[str, Any]
    ) -> SimulationResult:
        if command == "dispense":
            try:
                plate = self._plate(args.get("plate", ""))
                well = plate.well(args.get("well", "A1"))
            except (ConstraintViolation, ValueError) as exc:
                return SimulationResult(feasible=False, fidelity="kinematic",
                                        violations=[str(exc)])
            volume = args.get("volume_ul", 0.0)
            violations = []
            if volume > self.tip_volume_ul + 1e-9:
                violations.append(
                    f"the tips hold {self.tip_volume_ul:.2f} uL, less than the "
                    f"{volume:.2f} uL requested"
                )
            if well.volume_ul + volume > plate.working_volume_ul:
                violations.append(
                    f"would fill the well to {well.volume_ul + volume:.1f} uL against a "
                    f"{plate.working_volume_ul:.0f} uL working volume: overflow"
                )
            return SimulationResult(
                feasible=not violations, fidelity="high",
                predicted_state={"well_volume_ul": round(well.volume_ul + volume, 3)},
                predicted_duration_s=0.4, violations=violations,
            )

        if command in ("aspirate", "transfer"):
            volume = args.get("volume_ul", 0.0)
            warnings = []
            if volume < 2.0:
                # Not a violation: it is legal and it is a bad idea, and saying
                # which is the useful thing.
                warnings.append(
                    f"{volume:.2f} uL is near the pipette's floor; expected error is "
                    f"{(0.008 + 0.05 / max(volume, 0.5)) * 100:.1f}%, which will show up "
                    "as a drifting dilution series"
                )
            if command == "transfer" and not args.get("new_tips_each", True):
                warnings.append(
                    "reusing tips across destinations will carry reagent between wells"
                )
            if command == "transfer":
                needed = len(args.get("dest_wells", [])) * CHANNELS
                if args.get("new_tips_each", True) and needed > self.tips_remaining:
                    return SimulationResult(
                        feasible=False, fidelity="kinematic",
                        violations=[
                            f"needs {needed} tips with new_tips_each, but only "
                            f"{self.tips_remaining} remain"
                        ],
                    )
            return SimulationResult(
                feasible=True, fidelity="reduced_order",
                predicted_duration_s=len(args.get("dest_wells", [1])) * 0.6,
                warnings=warnings,
            )
        return SimulationResult(feasible=True, fidelity="none",
                                warnings=[f"no predictive model for {feature}.{command}"])
