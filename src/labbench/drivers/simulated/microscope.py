"""A simulated widefield fluorescence microscope.

Backed by the optical model in `_optics`, which means this instrument is a
genuine control problem rather than a stub that returns canned arrays:

* Defocus actually blurs, following the classical depth of field for the
  objective's numerical aperture, so autofocus has to search.
* Illumination actually bleaches the specimen, irreversibly, so a badly planned
  time-lapse actually destroys the sample and the agent can measure that it did.
* The focal plane is tilted, so an agent that autofocuses once and then tiles
  five millimetres produces visibly bad images -- the failure mode that matters
  in real tiling and the one a flat simulation hides.
* The camera has shot noise and read noise, so focus scores are noisy and a
  greedy hill-climb can be fooled.

Feature layout follows the substitutability rule: capabilities are grouped by
function, not by this instrument's model number. An agent driving
`MotionControl/move_absolute` here drives any stage that implements that
feature, from any vendor.
"""

from __future__ import annotations

import asyncio
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
from ...core.device import (
    Device,
    DeviceDescriptor,
    ExecutionContext,
    SimulationResult,
)
from ...core.errors import ConstraintViolation, DeviceNotReady
from .. import _png
from . import _optics

write_png = _png.write

#: Stage travel. Beyond this the carriage hits its hard stop, which on a real
#: instrument means a crashed objective and a service call.
X_TRAVEL_UM = (0.0, 20000.0)
Y_TRAVEL_UM = (0.0, 20000.0)
Z_TRAVEL_UM = (0.0, 300.0)


class SimulatedMicroscope(Device):
    """Widefield fluorescence microscope with a motorised XYZ stage."""

    def __init__(self, descriptor: DeviceDescriptor, **config: Any) -> None:
        super().__init__(descriptor, **config)
        self.data_dir = Path(config.get("data_dir", "./labbench-data/images")).expanduser()
        seed = int(config.get("seed", 7))
        self.rng = np.random.default_rng(seed)
        # The slide is centred on the stage travel, so home and the middle of
        # the specimen are the same place.
        centre = (
            (X_TRAVEL_UM[0] + X_TRAVEL_UM[1]) / 2,
            (Y_TRAVEL_UM[0] + Y_TRAVEL_UM[1]) / 2,
        )
        self.specimen = _optics.Specimen(
            seed=seed,
            extent_um=float(config.get("specimen_extent_um", 18000.0)),
            density_per_mm2=float(config.get("density_per_mm2", 3000.0)),
            centre_um=centre,
        )
        #: How long a micrometre of travel takes. Motion that completed
        #: instantly would let an agent learn a movement policy that is
        #: catastrophic on real hardware.
        self.speed_um_per_s = float(config.get("speed_um_per_s", 4000.0))
        self.settle_s = float(config.get("settle_s", 0.08))

        self.x_um = 10000.0
        self.y_um = 10000.0
        self.z_um = 150.0
        self.homed = False
        self.objective = str(config.get("objective", "20x"))
        self.channel = "BF"
        self.intensity_pct = 20.0
        self.exposure_ms = 50.0
        self.shutter_open = False
        self.width = int(config.get("width", 512))
        self.height = int(config.get("height", 512))
        self.bit_depth = 16
        self.frames_acquired = 0
        self.illumination_dose_j = 0.0
        self._last_metrics: dict[str, float] = {}

    # -- capability model -------------------------------------------------

    def _features(self) -> Sequence[Feature]:
        return [
            self._motion_feature(),
            self._focus_feature(),
            self._objective_feature(),
            self._illumination_feature(),
            self._camera_feature(),
        ]

    def _motion_feature(self) -> Feature:
        # Travel limits are only trustworthy once homed. Before that the
        # controller's zero is wherever it happened to power up, so the
        # advertised envelope would be a guess.
        limit = Constraint(minimum=X_TRAVEL_UM[0], maximum=X_TRAVEL_UM[1]) if self.homed else Constraint()
        ylimit = Constraint(minimum=Y_TRAVEL_UM[0], maximum=Y_TRAVEL_UM[1]) if self.homed else Constraint()
        needs_home = Precondition(
            property="homed", operator="is_true",
            message="the stage must be homed before it can be commanded to a position; "
                    "call MotionControl.home first",
        )
        return Feature(
            identifier="MotionControl",
            display_name="XY Stage",
            description="Motorised specimen stage.",
            namespace="org.labbench.motion",
            properties=[
                Property(name="x_um", description="Stage X position.",
                         schema=Parameter(name="x_um", unit="um", constraint=limit)),
                Property(name="y_um", description="Stage Y position.",
                         schema=Parameter(name="y_um", unit="um", constraint=ylimit)),
                Property(name="homed", description="True once the stage has found its datum.",
                         schema=Parameter(name="homed", type="boolean")),
                Property(name="speed_um_per_s", description="Traverse speed.",
                         schema=Parameter(name="speed_um_per_s", unit="um/s")),
            ],
            commands=[
                Command(
                    name="home",
                    description="Find the stage datum. Required before any commanded move.",
                    observable=True, duration_estimate_s=4.0,
                    hazard=Hazard.MOTION, reversibility=Reversibility.RESTORABLE,
                    tags={"moves_stage"},
                ),
                Command(
                    name="move_absolute",
                    description="Move the stage to an absolute XY position.",
                    parameters=[
                        Parameter(name="x_um", unit="um", description="Target X.",
                                  constraint=limit),
                        Parameter(name="y_um", unit="um", description="Target Y.",
                                  constraint=ylimit),
                    ],
                    returns=[Parameter(name="x_um", unit="um"), Parameter(name="y_um", unit="um")],
                    duration_estimate_s=1.0, hazard=Hazard.MOTION,
                    reversibility=Reversibility.REVERSIBLE, inverse="move_absolute",
                    preconditions=[needs_home], tags={"moves_stage"},
                ),
                Command(
                    name="move_relative",
                    description="Move the stage by an offset from where it is now.",
                    parameters=[
                        Parameter(name="dx_um", unit="um", default=0.0, required=False),
                        Parameter(name="dy_um", unit="um", default=0.0, required=False),
                    ],
                    returns=[Parameter(name="x_um", unit="um"), Parameter(name="y_um", unit="um")],
                    duration_estimate_s=0.5, hazard=Hazard.MOTION,
                    reversibility=Reversibility.REVERSIBLE, inverse="move_relative",
                    preconditions=[needs_home], tags={"moves_stage"},
                ),
            ],
            events=[
                Event(name="limit_reached", description="A commanded move was clipped at a hard stop.",
                      severity="warning"),
            ],
        )

    def _focus_feature(self) -> Feature:
        zlimit = Constraint(minimum=Z_TRAVEL_UM[0], maximum=Z_TRAVEL_UM[1])
        return Feature(
            identifier="FocusControl",
            display_name="Focus Drive",
            description="Objective Z drive and autofocus.",
            namespace="org.labbench.motion",
            properties=[
                Property(name="z_um", description="Objective Z height.",
                         schema=Parameter(name="z_um", unit="um", constraint=zlimit)),
                Property(name="focus_score",
                         description="Normalised variance of the Laplacian for the last frame. "
                                     "Higher is sharper; only comparable within one field.",
                         schema=Parameter(name="focus_score")),
            ],
            commands=[
                Command(
                    name="move_z",
                    description="Move the objective to an absolute Z height.",
                    parameters=[Parameter(name="z_um", unit="um", constraint=zlimit,
                                          description="Target focus height.")],
                    returns=[Parameter(name="z_um", unit="um")],
                    duration_estimate_s=0.4, hazard=Hazard.MOTION,
                    reversibility=Reversibility.REVERSIBLE, inverse="move_z",
                    # Driving Z into the specimen is the classic way to destroy
                    # both an objective and the sample.
                    tags={"moves_z", "collision_risk"},
                ),
                Command(
                    name="autofocus",
                    description="Search Z for maximum sharpness and move there. "
                                "Acquires frames, so it costs illumination dose.",
                    parameters=[
                        Parameter(name="range_um", unit="um", default=30.0, required=False,
                                  description="Total Z range to search, centred on the current Z.",
                                  constraint=Constraint(minimum=2.0, maximum=200.0)),
                        Parameter(name="steps", type="integer", default=15, required=False,
                                  description="Number of Z planes to sample.",
                                  constraint=Constraint(minimum=3, maximum=101)),
                    ],
                    returns=[
                        Parameter(name="z_um", unit="um"),
                        Parameter(name="focus_score"),
                        Parameter(name="improved", type="boolean"),
                    ],
                    observable=True, duration_estimate_s=8.0,
                    # It bleaches the sample, and that cannot be undone.
                    hazard=Hazard.SAMPLE, reversibility=Reversibility.IRREVERSIBLE,
                    tags={"moves_z", "photobleaching"},
                ),
            ],
        )

    def _objective_feature(self) -> Feature:
        return Feature(
            identifier="ObjectiveTurret",
            display_name="Objective Turret",
            namespace="org.labbench.optics",
            properties=[
                Property(name="objective", description="Objective currently in the light path.",
                         schema=Parameter(name="objective", type="string",
                                          constraint=Constraint(enum=sorted(_optics.OBJECTIVES)))),
                Property(name="numerical_aperture", description="NA of the current objective.",
                         schema=Parameter(name="numerical_aperture")),
                Property(name="working_distance_um",
                         description="Free space between objective and coverslip.",
                         schema=Parameter(name="working_distance_um", unit="um")),
            ],
            commands=[
                Command(
                    name="set_objective",
                    description="Rotate a different objective into the light path.",
                    parameters=[Parameter(name="objective", type="string",
                                          constraint=Constraint(enum=sorted(_optics.OBJECTIVES)))],
                    duration_estimate_s=2.5,
                    # A turret rotating with the objective near the coverslip is
                    # how objectives get destroyed.
                    hazard=Hazard.MOTION, reversibility=Reversibility.REVERSIBLE,
                    inverse="set_objective", tags={"collision_risk"},
                ),
            ],
        )

    def _illumination_feature(self) -> Feature:
        return Feature(
            identifier="Illumination",
            display_name="Illumination",
            namespace="org.labbench.optics",
            properties=[
                Property(name="channel", description="Excitation/emission channel.",
                         schema=Parameter(name="channel", type="string",
                                          constraint=Constraint(enum=sorted(_optics.CHANNELS))),
                         writable=True),
                Property(name="intensity_pct", description="Lamp power.",
                         schema=Parameter(name="intensity_pct", unit="%",
                                          constraint=Constraint(minimum=0, maximum=100)),
                         writable=True),
                Property(name="shutter_open", description="Excitation shutter state.",
                         schema=Parameter(name="shutter_open", type="boolean")),
                Property(name="cumulative_dose_j",
                         description="Total illumination energy delivered to this specimen. "
                                     "Monotonic; the sample cannot be un-bleached.",
                         schema=Parameter(name="cumulative_dose_j", unit="J")),
            ],
            commands=[
                Command(
                    name="set_channel",
                    description="Select the excitation/emission channel.",
                    parameters=[Parameter(name="channel", type="string",
                                          constraint=Constraint(enum=sorted(_optics.CHANNELS)))],
                    duration_estimate_s=0.6, hazard=Hazard.BENIGN,
                    reversibility=Reversibility.REVERSIBLE, inverse="set_channel",
                ),
                Command(
                    name="set_intensity",
                    description="Set lamp power.",
                    parameters=[Parameter(name="intensity_pct", unit="%",
                                          constraint=Constraint(minimum=0, maximum=100))],
                    duration_estimate_s=0.2, hazard=Hazard.BENIGN,
                    reversibility=Reversibility.REVERSIBLE, inverse="set_intensity",
                ),
                Command(
                    name="open_shutter",
                    description="Open the excitation shutter. The specimen begins bleaching.",
                    duration_estimate_s=0.15, hazard=Hazard.SAMPLE,
                    reversibility=Reversibility.IRREVERSIBLE, inverse="close_shutter",
                    tags={"photobleaching"},
                ),
                Command(
                    name="close_shutter", description="Close the excitation shutter.",
                    duration_estimate_s=0.15, hazard=Hazard.BENIGN,
                    reversibility=Reversibility.REVERSIBLE,
                ),
            ],
        )

    def _camera_feature(self) -> Feature:
        return Feature(
            identifier="Camera",
            display_name="sCMOS Camera",
            namespace="org.labbench.imaging",
            properties=[
                Property(name="exposure_ms", description="Integration time.",
                         schema=Parameter(name="exposure_ms", unit="ms",
                                          constraint=Constraint(minimum=0.1, maximum=10000.0)),
                         writable=True),
                Property(name="width", description="Frame width.",
                         schema=Parameter(name="width", type="integer", unit="px")),
                Property(name="height", description="Frame height.",
                         schema=Parameter(name="height", type="integer", unit="px")),
                Property(name="frames_acquired", description="Frames taken since connect.",
                         schema=Parameter(name="frames_acquired", type="integer")),
            ],
            commands=[
                Command(
                    name="snap",
                    description="Acquire one frame and write it as an artifact. "
                                "Returns image statistics and a focus score, not pixels.",
                    parameters=[
                        Parameter(name="exposure_ms", unit="ms", required=False,
                                  description="Override the exposure for this frame only.",
                                  constraint=Constraint(minimum=0.1, maximum=10000.0)),
                        Parameter(name="channel", type="string", required=False,
                                  constraint=Constraint(enum=sorted(_optics.CHANNELS))),
                    ],
                    returns=[
                        Parameter(name="artifact_uri", type="string"),
                        Parameter(name="focus_score"),
                        Parameter(name="saturated_fraction"),
                    ],
                    duration_estimate_s=0.6,
                    hazard=Hazard.SAMPLE, reversibility=Reversibility.IRREVERSIBLE,
                    tags={"photobleaching", "produces_artifact"},
                ),
                Command(
                    name="acquire_tile",
                    description="Raster a rectangular region, acquiring one frame per field. "
                                "Long-running: returns a job handle with progress.",
                    parameters=[
                        Parameter(name="columns", type="integer", default=3, required=False,
                                  constraint=Constraint(minimum=1, maximum=50)),
                        Parameter(name="rows", type="integer", default=3, required=False,
                                  constraint=Constraint(minimum=1, maximum=50)),
                        Parameter(name="overlap_pct", unit="%", default=10.0, required=False,
                                  constraint=Constraint(minimum=0.0, maximum=50.0)),
                        Parameter(name="autofocus_every", type="integer", default=0, required=False,
                                  description="Re-autofocus every N fields. 0 disables. "
                                              "The focal plane is tilted, so 0 over a large "
                                              "region gives progressively worse focus.",
                                  constraint=Constraint(minimum=0, maximum=100)),
                    ],
                    returns=[
                        Parameter(name="fields", type="integer"),
                        Parameter(name="mean_focus_score"),
                    ],
                    observable=True, duration_estimate_s=60.0,
                    hazard=Hazard.SAMPLE, reversibility=Reversibility.IRREVERSIBLE,
                    tags={"photobleaching", "moves_stage", "produces_artifact"},
                ),
            ],
            events=[
                Event(name="frame_acquired", description="One frame was written.", severity="debug"),
                Event(name="saturation", description="A frame was substantially saturated.",
                      severity="warning"),
            ],
        )

    # -- lifecycle --------------------------------------------------------

    async def _connect(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.sleep(0.05)

    async def _initialize(self, ctx: ExecutionContext) -> None:
        await self._home(ctx)

    async def _estop(self) -> None:
        """Stop motion and shut the light out. Both, immediately."""
        self.shutter_open = False
        # Position after an e-stop mid-move is not knowable, so the datum is
        # forfeit. Claiming otherwise would be the dangerous answer.
        self.homed = False
        self.invalidate_features()

    # -- property access --------------------------------------------------

    async def _read(self, feature: str, name: str) -> Any:
        _mag, na, wd = _optics.OBJECTIVES[self.objective]
        table: dict[tuple[str, str], Any] = {
            ("MotionControl", "x_um"): self.x_um,
            ("MotionControl", "y_um"): self.y_um,
            ("MotionControl", "homed"): self.homed,
            ("MotionControl", "speed_um_per_s"): self.speed_um_per_s,
            ("FocusControl", "z_um"): self.z_um,
            ("FocusControl", "focus_score"): self._last_metrics.get("focus_score", 0.0),
            ("ObjectiveTurret", "objective"): self.objective,
            ("ObjectiveTurret", "numerical_aperture"): na,
            ("ObjectiveTurret", "working_distance_um"): wd,
            ("Illumination", "channel"): self.channel,
            ("Illumination", "intensity_pct"): self.intensity_pct,
            ("Illumination", "shutter_open"): self.shutter_open,
            ("Illumination", "cumulative_dose_j"): self.illumination_dose_j,
            ("Camera", "exposure_ms"): self.exposure_ms,
            ("Camera", "width"): self.width,
            ("Camera", "height"): self.height,
            ("Camera", "frames_acquired"): self.frames_acquired,
        }
        return table[(feature, name)]

    async def _write(self, feature: str, name: str, value: Any) -> None:
        if (feature, name) == ("Illumination", "channel"):
            self.channel = value
        elif (feature, name) == ("Illumination", "intensity_pct"):
            self.intensity_pct = float(value)
        elif (feature, name) == ("Camera", "exposure_ms"):
            self.exposure_ms = float(value)
        else:  # pragma: no cover - the base class checks `writable` first
            raise DeviceNotReady(f"{feature}.{name} is not writable", device=self.id)

    # -- commands ---------------------------------------------------------

    async def _invoke(
        self, feature: str, command: str, args: dict[str, Any], ctx: ExecutionContext
    ) -> Any:
        handler = getattr(self, f"_cmd_{feature.lower()}_{command}", None)
        if handler is None:  # pragma: no cover - resolve() checks first
            raise DeviceNotReady(f"{feature}.{command} is not implemented", device=self.id)
        return await handler(ctx, **args)

    # MotionControl -------------------------------------------------------

    async def _cmd_motioncontrol_home(self, ctx: ExecutionContext) -> dict[str, Any]:
        return await self._home(ctx)

    async def _home(self, ctx: ExecutionContext) -> dict[str, Any]:
        for i, message in enumerate(
            ["seeking X datum", "seeking Y datum", "seeking Z datum", "verifying"]
        ):
            ctx.raise_if_cancelled()
            await ctx.progress((i + 1) / 4, message)
            await asyncio.sleep(0.25)
        self.x_um, self.y_um, self.z_um = 0.0, 0.0, 100.0
        self.homed = True
        # Travel limits become knowable only now, so the advertised envelope
        # changes and any cached capability description is stale.
        self.invalidate_features()
        await self.emit("MotionControl", "homed", {"x_um": 0.0, "y_um": 0.0})
        return {"x_um": self.x_um, "y_um": self.y_um, "z_um": self.z_um, "homed": True}

    async def _cmd_motioncontrol_move_absolute(
        self, ctx: ExecutionContext, x_um: float, y_um: float
    ) -> dict[str, Any]:
        return await self._travel(ctx, x_um, y_um)

    async def _cmd_motioncontrol_move_relative(
        self, ctx: ExecutionContext, dx_um: float = 0.0, dy_um: float = 0.0
    ) -> dict[str, Any]:
        return await self._travel(ctx, self.x_um + dx_um, self.y_um + dy_um)

    async def _travel(self, ctx: ExecutionContext, x_um: float, y_um: float) -> dict[str, Any]:
        clipped_x = min(max(x_um, X_TRAVEL_UM[0]), X_TRAVEL_UM[1])
        clipped_y = min(max(y_um, Y_TRAVEL_UM[0]), Y_TRAVEL_UM[1])
        if (clipped_x, clipped_y) != (x_um, y_um):
            # Refuse rather than silently clip. An agent that asked for a
            # position outside the envelope has a wrong model of the stage, and
            # quietly moving somewhere else would leave that model wrong while
            # producing plausible images from the wrong place.
            await self.emit(
                "MotionControl", "limit_reached",
                {"requested": [x_um, y_um], "limits": [X_TRAVEL_UM, Y_TRAVEL_UM]},
                severity="warning",
            )
            raise ConstraintViolation(
                f"({x_um:.0f}, {y_um:.0f}) um is outside the stage travel "
                f"X{X_TRAVEL_UM} Y{Y_TRAVEL_UM}",
                requested={"x_um": x_um, "y_um": y_um},
                limits={"x_um": list(X_TRAVEL_UM), "y_um": list(Y_TRAVEL_UM)},
            )
        distance = math.hypot(clipped_x - self.x_um, clipped_y - self.y_um)
        duration = distance / max(self.speed_um_per_s, 1.0) + self.settle_s
        await self._interruptible_sleep(ctx, duration)
        self.x_um, self.y_um = clipped_x, clipped_y
        return {"x_um": self.x_um, "y_um": self.y_um, "travelled_um": round(distance, 2)}

    # FocusControl --------------------------------------------------------

    async def _cmd_focuscontrol_move_z(self, ctx: ExecutionContext, z_um: float) -> dict[str, Any]:
        if not Z_TRAVEL_UM[0] <= z_um <= Z_TRAVEL_UM[1]:
            raise ConstraintViolation(
                f"z={z_um} um is outside the focus travel {Z_TRAVEL_UM}",
                requested=z_um, limits=list(Z_TRAVEL_UM),
            )
        await self._interruptible_sleep(ctx, abs(z_um - self.z_um) / 500.0 + 0.05)
        self.z_um = z_um
        return {"z_um": self.z_um}

    async def _cmd_focuscontrol_autofocus(
        self, ctx: ExecutionContext, range_um: float = 30.0, steps: int = 15
    ) -> dict[str, Any]:
        start_z = self.z_um
        start_score = self._last_metrics.get("focus_score", 0.0)
        lo = max(Z_TRAVEL_UM[0], start_z - range_um / 2)
        hi = min(Z_TRAVEL_UM[1], start_z + range_um / 2)
        planes = np.linspace(lo, hi, steps)

        best_z, best_score = start_z, -1.0
        for index, z in enumerate(planes):
            ctx.raise_if_cancelled()
            self.z_um = float(z)
            _, metrics = self._render(apply_bleaching=True)
            if metrics["focus_score"] > best_score:
                best_z, best_score = float(z), metrics["focus_score"]
            await ctx.progress(
                (index + 1) / steps, f"plane {index + 1}/{steps} at z={z:.1f} um"
            )
            await asyncio.sleep(0.01)

        self.z_um = best_z
        self._last_metrics["focus_score"] = best_score
        truth = self.specimen.surface_z(self.x_um, self.y_um)
        return {
            "z_um": round(best_z, 3),
            "focus_score": round(best_score, 4),
            "improved": bool(best_score > start_score),
            "search_range_um": [round(lo, 2), round(hi, 2)],
            "planes_sampled": steps,
            # Ground truth, which a real instrument cannot report. Present so a
            # simulated run can be *scored*, not so an agent can cheat with it:
            # the agent-facing tool layer strips keys prefixed `truth_`.
            "truth_residual_um": round(best_z - truth, 3),
        }

    # ObjectiveTurret -----------------------------------------------------

    async def _cmd_objectiveturret_set_objective(
        self, ctx: ExecutionContext, objective: str
    ) -> dict[str, Any]:
        _, _, working_distance = _optics.OBJECTIVES[objective]
        # Rotating a high-magnification objective in while the stage sits above
        # its working distance is exactly how objectives are destroyed.
        if self.homed and self.z_um > working_distance:
            raise ConstraintViolation(
                f"cannot select {objective}: its working distance is "
                f"{working_distance:.0f} um but the focus drive is at {self.z_um:.0f} um. "
                f"Lower Z below {working_distance:.0f} um first, or the objective will "
                "collide with the specimen.",
                objective=objective, working_distance_um=working_distance, z_um=self.z_um,
            )
        await self._interruptible_sleep(ctx, 1.2)
        self.objective = objective
        self.invalidate_features()
        return {"objective": objective, "numerical_aperture": _optics.OBJECTIVES[objective][1]}

    # Illumination --------------------------------------------------------

    async def _cmd_illumination_set_channel(
        self, ctx: ExecutionContext, channel: str
    ) -> dict[str, Any]:
        await self._interruptible_sleep(ctx, 0.4)
        self.channel = channel
        return {"channel": channel}

    async def _cmd_illumination_set_intensity(
        self, ctx: ExecutionContext, intensity_pct: float
    ) -> dict[str, Any]:
        self.intensity_pct = float(intensity_pct)
        return {"intensity_pct": self.intensity_pct}

    async def _cmd_illumination_open_shutter(self, ctx: ExecutionContext) -> dict[str, Any]:
        self.shutter_open = True
        return {"shutter_open": True}

    async def _cmd_illumination_close_shutter(self, ctx: ExecutionContext) -> dict[str, Any]:
        self.shutter_open = False
        return {"shutter_open": False}

    # Camera --------------------------------------------------------------

    async def _cmd_camera_snap(
        self, ctx: ExecutionContext, exposure_ms: float | None = None, channel: str | None = None
    ) -> dict[str, Any]:
        exposure = float(exposure_ms if exposure_ms is not None else self.exposure_ms)
        previous_channel = self.channel
        if channel is not None:
            self.channel = channel
        try:
            await self._interruptible_sleep(ctx, min(exposure / 1000.0, 1.5) + 0.05)
            image, metrics = self._render(exposure_ms=exposure, apply_bleaching=True)
            artifact = self._write_frame(image, metrics, exposure)
        finally:
            self.channel = previous_channel
        if metrics["saturated_fraction"] > 0.02:
            await self.emit(
                "Camera", "saturation",
                {"fraction": metrics["saturated_fraction"], "exposure_ms": exposure},
                severity="warning",
            )
        await self.emit("Camera", "frame_acquired", {"uri": artifact["uri"]}, severity="debug")
        return {
            "artifact_uri": artifact["uri"],
            "artifacts": [artifact],
            "focus_score": round(metrics["focus_score"], 4),
            "saturated_fraction": round(metrics["saturated_fraction"], 5),
            "snr_estimate": round(metrics["snr_estimate"], 3),
            "pixel_size_um": round(metrics["pixel_size_um"], 4),
            "exposure_ms": exposure,
            "position": {"x_um": self.x_um, "y_um": self.y_um, "z_um": self.z_um},
        }

    async def _cmd_camera_acquire_tile(
        self,
        ctx: ExecutionContext,
        columns: int = 3,
        rows: int = 3,
        overlap_pct: float = 10.0,
        autofocus_every: int = 0,
    ) -> dict[str, Any]:
        if not self.homed:
            raise DeviceNotReady(
                "the stage must be homed before tiling", device=self.id, state=self.state.value
            )
        mag, _, _ = _optics.OBJECTIVES[self.objective]
        field_w = self.width * _optics.CAMERA_PIXEL_UM / mag
        field_h = self.height * _optics.CAMERA_PIXEL_UM / mag
        step_x = field_w * (1.0 - overlap_pct / 100.0)
        step_y = field_h * (1.0 - overlap_pct / 100.0)
        origin_x, origin_y = self.x_um, self.y_um

        total = columns * rows
        artifacts: list[dict[str, Any]] = []
        scores: list[float] = []
        for index in range(total):
            ctx.raise_if_cancelled()
            col, row = index % columns, index // columns
            # Serpentine raster: reversing alternate rows halves the travel and
            # is what any real tiling routine does.
            if row % 2:
                col = columns - 1 - col
            target_x = origin_x + col * step_x
            target_y = origin_y + row * step_y
            if not (X_TRAVEL_UM[0] <= target_x <= X_TRAVEL_UM[1]
                    and Y_TRAVEL_UM[0] <= target_y <= Y_TRAVEL_UM[1]):
                raise ConstraintViolation(
                    f"tile {index + 1} would need ({target_x:.0f}, {target_y:.0f}) um, "
                    f"outside the stage travel. Reduce columns/rows or start nearer the centre.",
                    tile=index + 1, requested={"x_um": target_x, "y_um": target_y},
                )
            await self._travel(ctx, target_x, target_y)
            if autofocus_every and index % autofocus_every == 0:
                await self._cmd_focuscontrol_autofocus(ctx, range_um=12.0, steps=7)
            image, metrics = self._render(apply_bleaching=True)
            artifacts.append(self._write_frame(image, metrics, self.exposure_ms, tile=(col, row)))
            scores.append(metrics["focus_score"])
            await ctx.progress(
                (index + 1) / total,
                f"field {index + 1}/{total} at ({target_x:.0f}, {target_y:.0f}) um",
            )

        return {
            "fields": total,
            "artifacts": artifacts,
            "mean_focus_score": round(float(np.mean(scores)), 4),
            "min_focus_score": round(float(np.min(scores)), 4),
            # A large spread over a tiled region is the signature of focus
            # drift across the tilted plane, and is the number worth surfacing.
            "focus_score_spread": round(float(np.max(scores) - np.min(scores)), 4),
            "grid": {"columns": columns, "rows": rows, "overlap_pct": overlap_pct},
        }

    # -- rendering --------------------------------------------------------

    def _render(
        self, *, exposure_ms: float | None = None, apply_bleaching: bool = True
    ) -> tuple[np.ndarray, dict[str, float]]:
        exposure = float(exposure_ms if exposure_ms is not None else self.exposure_ms)
        image, metrics = _optics.render(
            self.specimen,
            x_um=self.x_um, y_um=self.y_um, z_um=self.z_um,
            objective=self.objective, channel=self.channel,
            exposure_ms=exposure, intensity_pct=self.intensity_pct,
            width=self.width, height=self.height, bit_depth=self.bit_depth,
            rng=self.rng, apply_bleaching=apply_bleaching,
        )
        if apply_bleaching:
            self.frames_acquired += 1
            self.illumination_dose_j += exposure / 1000.0 * self.intensity_pct / 100.0
            self._last_metrics = metrics
        return image, metrics

    def _write_frame(
        self,
        image: np.ndarray,
        metrics: dict[str, float],
        exposure_ms: float,
        tile: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        suffix = f"_c{tile[0]:02d}r{tile[1]:02d}" if tile else ""
        name = f"{self.id}_{stamp}_{uuid.uuid4().hex[:6]}{suffix}.png"
        path = self.data_dir / name
        size = write_png(str(path), image, bit_depth=self.bit_depth)
        return {
            "uri": path.resolve().as_uri(),
            "kind": "image",
            "mime_type": "image/png",
            "bytes": size,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "metadata": {
                "device": self.id,
                "objective": self.objective,
                "channel": self.channel,
                "exposure_ms": exposure_ms,
                "intensity_pct": self.intensity_pct,
                "x_um": self.x_um, "y_um": self.y_um, "z_um": self.z_um,
                "pixel_size_um": round(metrics["pixel_size_um"], 4),
                "focus_score": round(metrics["focus_score"], 4),
                "simulated": True,
            },
        }

    # -- prediction -------------------------------------------------------

    async def _simulate(
        self, feature: str, command: str, args: dict[str, Any]
    ) -> SimulationResult:
        """Predict an outcome without moving anything.

        This is the third safety gate. Its job is to catch the collision and
        the out-of-envelope move *before* the motors turn, so the answers here
        are deliberately conservative.
        """
        if feature == "MotionControl" and command in ("move_absolute", "move_relative"):
            if command == "move_absolute":
                target_x, target_y = args.get("x_um", self.x_um), args.get("y_um", self.y_um)
            else:
                target_x = self.x_um + args.get("dx_um", 0.0)
                target_y = self.y_um + args.get("dy_um", 0.0)
            violations = []
            if not X_TRAVEL_UM[0] <= target_x <= X_TRAVEL_UM[1]:
                violations.append(f"X target {target_x:.0f} um is outside travel {X_TRAVEL_UM}")
            if not Y_TRAVEL_UM[0] <= target_y <= Y_TRAVEL_UM[1]:
                violations.append(f"Y target {target_y:.0f} um is outside travel {Y_TRAVEL_UM}")
            warnings = []
            if not self.homed:
                warnings.append("the stage is not homed; its true position is unknown")
            distance = math.hypot(target_x - self.x_um, target_y - self.y_um)
            return SimulationResult(
                feasible=not violations,
                fidelity="kinematic",
                predicted_state={"x_um": target_x, "y_um": target_y},
                predicted_duration_s=distance / self.speed_um_per_s + self.settle_s,
                violations=violations, warnings=warnings,
                notes=f"straight-line traverse of {distance:.0f} um",
            )

        if feature == "FocusControl" and command == "move_z":
            z = args.get("z_um", self.z_um)
            _, _, working_distance = _optics.OBJECTIVES[self.objective]
            violations = []
            if not Z_TRAVEL_UM[0] <= z <= Z_TRAVEL_UM[1]:
                violations.append(f"Z target {z:.0f} um is outside travel {Z_TRAVEL_UM}")
            warnings = []
            if z > working_distance:
                violations.append(
                    f"Z {z:.0f} um exceeds the {self.objective} working distance "
                    f"{working_distance:.0f} um: the objective would contact the specimen"
                )
            surface = self.specimen.surface_z(self.x_um, self.y_um)
            if abs(z - surface) > 25:
                warnings.append(
                    f"predicted defocus {z - surface:+.1f} um; the field will be badly blurred"
                )
            return SimulationResult(
                feasible=not violations, fidelity="reduced_order",
                predicted_state={"z_um": z},
                predicted_duration_s=abs(z - self.z_um) / 500.0,
                violations=violations, warnings=warnings,
            )

        if feature == "ObjectiveTurret" and command == "set_objective":
            objective = args.get("objective", self.objective)
            _, _, working_distance = _optics.OBJECTIVES[objective]
            violations = []
            if self.homed and self.z_um > working_distance:
                violations.append(
                    f"selecting {objective} at Z {self.z_um:.0f} um would drive its "
                    f"{working_distance:.0f} um working distance into the specimen"
                )
            return SimulationResult(
                feasible=not violations, fidelity="kinematic",
                predicted_state={"objective": objective},
                predicted_duration_s=1.2, violations=violations,
            )

        if feature == "Camera" and command in ("snap", "acquire_tile"):
            # Render without bleaching: the whole point of a dry run is that it
            # must not consume the sample it is predicting about.
            _, metrics = self._render(
                exposure_ms=args.get("exposure_ms"), apply_bleaching=False
            )
            warnings = []
            if metrics["saturated_fraction"] > 0.02:
                warnings.append(
                    f"{metrics['saturated_fraction'] * 100:.1f}% of pixels would saturate; "
                    "reduce exposure or lamp intensity"
                )
            if metrics["snr_estimate"] < 2.0:
                warnings.append("predicted SNR is very low; the frame will be mostly noise")
            fields = args.get("columns", 1) * args.get("rows", 1)
            return SimulationResult(
                feasible=True, fidelity="high",
                predicted_state={"focus_score": round(metrics["focus_score"], 4)},
                predicted_duration_s=fields * (self.exposure_ms / 1000.0 + 0.35),
                warnings=warnings,
                notes="predicted from the optical model without illuminating the specimen",
            )

        return SimulationResult(
            feasible=True, fidelity="none",
            warnings=[f"no predictive model for {feature}.{command}"],
        )

    # -- helpers ----------------------------------------------------------

    async def _interruptible_sleep(self, ctx: ExecutionContext, seconds: float) -> None:
        """Sleep in slices so cancellation is honoured promptly.

        A long move that cannot be interrupted is a safety problem: the whole
        point of an e-stop is that it takes effect now, not at the end of the
        current command.
        """
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            ctx.raise_if_cancelled()
            await asyncio.sleep(min(0.05, deadline - time.monotonic()))
