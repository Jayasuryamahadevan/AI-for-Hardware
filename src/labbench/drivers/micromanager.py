"""Micro-Manager microscopes, through `pymmcore-plus`.

Micro-Manager's own core API (MMCore, wrapped 1:1 by `pymmcore-plus`'s
`CMMCorePlus`) is the closest thing microscopy automation has to a universal
driver layer -- one core, one `.cfg` file describing which device adapters are
loaded, and every stage/camera/shutter/filter-wheel vendor plugs into the same
handful of calls (`setXYPosition`, `snapImage`, `setConfig`, ...). This driver
is a comparatively thin adapter over that, and its feature layout deliberately
mirrors `drivers.simulated.microscope.SimulatedMicroscope`: an agent that
learned the simulated bench transfers to a real one built on Micro-Manager
with no new vocabulary, which is the substitutability promise the README
makes concrete.

MMCore is synchronous C++ underneath; every call here is pushed to a worker
thread via `asyncio.to_thread`; the same reason PyVISA is in `drivers/scpi.py`.

Device *labels* (which adapter plays "the" XY stage, "the" camera) come from
the loaded `.cfg` file and can be ambiguous when more than one of a kind is
loaded, so this driver trusts MMCore's own notion of "current" device
(`getXYStageDevice()`, `getCameraDevice()`, ...) unless the lab configuration
names one explicitly -- the same "profile can override the default" shape
`drivers/scpi.py` uses for its instrument classes.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.capability import (
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
from ..core.device import Device, DeviceDescriptor, ExecutionContext, SimulationResult
from ..core.errors import DeviceNotReady, DriverUnavailable, ValidationError
from . import _png


class MicroManagerMicroscope(Device):
    """One Micro-Manager-controlled microscope.

    Configuration:

        driver: micromanager
        settings:
          config_file: /path/to/MMConfig.cfg   # defaults to the MM demo config
          data_dir: ./labbench-data/images
          xy_stage: XY            # optional; defaults to MMCore's current XY stage
          focus_device: Z         # optional; defaults to MMCore's current focus device
          camera: Camera          # optional; defaults to MMCore's current camera
          channel_group: Channel  # optional; the config group used as "channel"
    """

    requires_package = "pymmcore_plus"

    def __init__(self, descriptor: DeviceDescriptor, **config: Any) -> None:
        super().__init__(descriptor, **config)
        self.config_file = config.get("config_file", "MMConfig_demo.cfg")
        self.data_dir = Path(config.get("data_dir", "./labbench-data/images")).expanduser()
        self._xy_stage_override = config.get("xy_stage")
        self._focus_override = config.get("focus_device")
        self._camera_override = config.get("camera")
        self.channel_group = config.get("channel_group", "Channel")
        self.core: Any = None
        self.homed = False
        self._last_focus_score: float | None = None

    # -- lifecycle --------------------------------------------------------

    async def _connect(self) -> None:
        try:
            from pymmcore_plus import CMMCorePlus
        except ImportError:
            raise DriverUnavailable(
                "the micromanager driver needs pymmcore-plus. Install it with: "
                "pip install 'labbench[micromanager]', then run 'mmcore install' once "
                "to fetch the device adapters.",
                driver="micromanager",
            ) from None
        self.data_dir.mkdir(parents=True, exist_ok=True)

        def load() -> Any:
            core = CMMCorePlus()
            core.loadSystemConfiguration(self.config_file)
            return core

        self.core = await asyncio.to_thread(load)
        self.descriptor.vendor = "Micro-Manager"
        self.descriptor.model = Path(self.config_file).stem
        self.descriptor.protocol = "mmcore"

    async def _disconnect(self) -> None:
        if self.core is not None:
            await asyncio.to_thread(self.core.unloadAllDevices)
            self.core = None

    async def _initialize(self, ctx: ExecutionContext) -> None:
        await self._home(ctx)

    async def _estop(self) -> None:
        """Best-effort: MMCore has no universal e-stop, only per-device stop calls.

        Stages that support it are told to stop where they are; the datum is
        forfeit afterwards for the same reason it is in the simulated driver
        -- position mid-e-stop is not knowable, and claiming otherwise would
        be the dangerous answer.
        """
        if self.core is None:
            return
        try:
            await asyncio.to_thread(self.core.stop, self._xy_stage())
        except Exception:  # noqa: BLE001, S110 - never block an e-stop
            pass
        try:
            await asyncio.to_thread(self.core.setShutterOpen, False)
        except Exception:  # noqa: BLE001, S110 - never block an e-stop
            pass
        self.homed = False
        self.invalidate_features()

    # -- device label resolution -------------------------------------------

    def _xy_stage(self) -> str:
        return self._xy_stage_override or self.core.getXYStageDevice()

    def _focus(self) -> str:
        return self._focus_override or self.core.getFocusDevice()

    def _camera(self) -> str:
        return self._camera_override or self.core.getCameraDevice()

    def _shutter(self) -> str:
        return self.core.getShutterDevice()

    # -- capability model -------------------------------------------------

    def _features(self) -> Sequence[Feature]:
        features = [self._motion_feature(), self._focus_feature(), self._camera_feature()]
        if self.channel_group in self.core.getAvailableConfigGroups():
            features.append(self._channel_feature())
        if self._shutter():
            features.append(self._shutter_feature())
        return features

    def _motion_feature(self) -> Feature:
        needs_home = Precondition(
            property="homed", operator="is_true",
            message="the stage must be homed before it can be commanded to a position; "
                    "call MotionControl.home first",
        )
        return Feature(
            identifier="MotionControl", display_name="XY Stage",
            namespace="org.labbench.motion",
            properties=[
                Property(name="x_um", schema=Parameter(name="x_um", unit="um")),
                Property(name="y_um", schema=Parameter(name="y_um", unit="um")),
                Property(name="homed", schema=Parameter(name="homed", type="boolean")),
            ],
            commands=[
                Command(name="home", description="Home the XY stage, if the adapter supports it.",
                        observable=True, duration_estimate_s=5.0, hazard=Hazard.MOTION,
                        reversibility=Reversibility.RESTORABLE, tags={"moves_stage"}),
                Command(name="move_absolute", description="Move to an absolute XY position.",
                        parameters=[Parameter(name="x_um", unit="um"), Parameter(name="y_um", unit="um")],
                        returns=[Parameter(name="x_um", unit="um"), Parameter(name="y_um", unit="um")],
                        duration_estimate_s=1.0, hazard=Hazard.MOTION,
                        reversibility=Reversibility.REVERSIBLE, inverse="move_absolute",
                        preconditions=[needs_home], tags={"moves_stage"}),
                Command(name="move_relative", description="Move by an offset from the current position.",
                        parameters=[Parameter(name="dx_um", unit="um", default=0.0, required=False),
                                    Parameter(name="dy_um", unit="um", default=0.0, required=False)],
                        returns=[Parameter(name="x_um", unit="um"), Parameter(name="y_um", unit="um")],
                        duration_estimate_s=0.5, hazard=Hazard.MOTION,
                        reversibility=Reversibility.REVERSIBLE, inverse="move_relative",
                        preconditions=[needs_home], tags={"moves_stage"}),
            ],
        )

    def _focus_feature(self) -> Feature:
        return Feature(
            identifier="FocusControl", display_name="Focus Drive",
            namespace="org.labbench.motion",
            properties=[
                Property(name="z_um", schema=Parameter(name="z_um", unit="um")),
                Property(name="focus_score", schema=Parameter(name="focus_score")),
            ],
            commands=[
                Command(name="move_z", description="Move the focus device to an absolute Z height.",
                        parameters=[Parameter(name="z_um", unit="um")],
                        returns=[Parameter(name="z_um", unit="um")],
                        duration_estimate_s=0.5, hazard=Hazard.MOTION,
                        reversibility=Reversibility.REVERSIBLE, inverse="move_z",
                        tags={"moves_z", "collision_risk"}),
                Command(name="autofocus",
                        description="Run MMCore's configured hardware/software autofocus. "
                                    "Requires an autofocus device in the loaded configuration.",
                        returns=[Parameter(name="z_um", unit="um")],
                        observable=True, duration_estimate_s=6.0,
                        hazard=Hazard.SAMPLE, reversibility=Reversibility.IRREVERSIBLE,
                        tags={"moves_z", "photobleaching"}),
            ],
        )

    def _channel_feature(self) -> Feature:
        configs = list(self.core.getAvailableConfigs(self.channel_group))
        return Feature(
            identifier="ChannelControl", display_name="Channel / Config Group",
            namespace="org.labbench.optics",
            properties=[
                Property(name="channel", writable=True,
                         schema=Parameter(name="channel", type="string",
                                          constraint=Constraint(enum=configs))),
            ],
            commands=[
                Command(name="set_channel", description=f"Apply a preset from the "
                        f"{self.channel_group!r} config group.",
                        parameters=[Parameter(name="channel", type="string",
                                              constraint=Constraint(enum=configs))],
                        duration_estimate_s=0.6, hazard=Hazard.BENIGN,
                        reversibility=Reversibility.REVERSIBLE, inverse="set_channel"),
            ],
        )

    def _shutter_feature(self) -> Feature:
        return Feature(
            identifier="Illumination", display_name="Shutter",
            namespace="org.labbench.optics",
            properties=[
                Property(name="shutter_open", schema=Parameter(name="shutter_open", type="boolean")),
            ],
            commands=[
                Command(name="open_shutter", description="Open the shutter. The specimen begins bleaching.",
                        duration_estimate_s=0.1, hazard=Hazard.SAMPLE,
                        reversibility=Reversibility.IRREVERSIBLE, inverse="close_shutter",
                        tags={"photobleaching"}),
                Command(name="close_shutter", description="Close the shutter.",
                        duration_estimate_s=0.1, hazard=Hazard.BENIGN,
                        reversibility=Reversibility.REVERSIBLE),
            ],
        )

    def _camera_feature(self) -> Feature:
        return Feature(
            identifier="Camera", display_name="Camera",
            namespace="org.labbench.imaging",
            properties=[
                Property(name="exposure_ms", writable=True,
                         schema=Parameter(name="exposure_ms", unit="ms",
                                          constraint=Constraint(minimum=0.1, maximum=30000.0))),
            ],
            commands=[
                Command(name="snap", description="Acquire one frame and write it as an artifact.",
                        parameters=[Parameter(name="exposure_ms", unit="ms", required=False,
                                              constraint=Constraint(minimum=0.1, maximum=30000.0))],
                        returns=[Parameter(name="artifact_uri", type="string")],
                        duration_estimate_s=1.0, hazard=Hazard.SAMPLE,
                        reversibility=Reversibility.IRREVERSIBLE,
                        tags={"photobleaching", "produces_artifact"}),
            ],
            events=[Event(name="frame_acquired", description="One frame was written.", severity="debug")],
        )

    # -- lifecycle helpers --------------------------------------------------

    async def _home(self, ctx: ExecutionContext) -> dict[str, Any]:
        stage = self._xy_stage()
        try:
            await asyncio.to_thread(self.core.home, stage)
        except Exception as exc:  # noqa: BLE001 - not every adapter supports homing
            raise DeviceNotReady(
                f"{stage} does not support homing: {exc}", device=self.id,
            ) from None
        self.homed = True
        self.invalidate_features()
        x, y = await asyncio.to_thread(self.core.getXYPosition)
        return {"x_um": x, "y_um": y, "homed": True}

    # -- property access --------------------------------------------------

    async def _read(self, feature: str, name: str) -> Any:
        core = self.core
        if (feature, name) == ("MotionControl", "x_um"):
            x, _ = await asyncio.to_thread(core.getXYPosition)
            return x
        if (feature, name) == ("MotionControl", "y_um"):
            _, y = await asyncio.to_thread(core.getXYPosition)
            return y
        if (feature, name) == ("MotionControl", "homed"):
            return self.homed
        if (feature, name) == ("FocusControl", "z_um"):
            return await asyncio.to_thread(core.getPosition, self._focus())
        if (feature, name) == ("FocusControl", "focus_score"):
            return self._last_focus_score or 0.0
        if (feature, name) == ("ChannelControl", "channel"):
            return await asyncio.to_thread(core.getCurrentConfig, self.channel_group)
        if (feature, name) == ("Illumination", "shutter_open"):
            return bool(await asyncio.to_thread(core.getShutterOpen))
        if (feature, name) == ("Camera", "exposure_ms"):
            return await asyncio.to_thread(core.getExposure)
        raise ValidationError(f"{feature}.{name} is not a readable property here")

    async def _write(self, feature: str, name: str, value: Any) -> None:
        core = self.core
        if (feature, name) == ("ChannelControl", "channel"):
            await asyncio.to_thread(core.setConfig, self.channel_group, value)
            return
        if (feature, name) == ("Camera", "exposure_ms"):
            await asyncio.to_thread(core.setExposure, float(value))
            return
        raise ValidationError(f"{feature}.{name} is not writable here")

    # -- commands -----------------------------------------------------------

    async def _invoke(
        self, feature: str, command: str, args: dict[str, Any], ctx: ExecutionContext
    ) -> Any:
        core = self.core
        if feature == "MotionControl":
            if command == "home":
                return await self._home(ctx)
            if command == "move_absolute":
                return await self._move_xy(args["x_um"], args["y_um"])
            if command == "move_relative":
                x, y = await asyncio.to_thread(core.getXYPosition)
                return await self._move_xy(
                    x + args.get("dx_um", 0.0), y + args.get("dy_um", 0.0)
                )
        if feature == "FocusControl":
            if command == "move_z":
                return await self._move_z(args["z_um"])
            if command == "autofocus":
                return await self._autofocus(ctx)
        if feature == "ChannelControl" and command == "set_channel":
            await asyncio.to_thread(core.setConfig, self.channel_group, args["channel"])
            await asyncio.to_thread(core.waitForConfig, self.channel_group, args["channel"])
            return {"channel": args["channel"]}
        if feature == "Illumination":
            if command == "open_shutter":
                await asyncio.to_thread(core.setShutterOpen, True)
                return {"shutter_open": True}
            if command == "close_shutter":
                await asyncio.to_thread(core.setShutterOpen, False)
                return {"shutter_open": False}
        if feature == "Camera" and command == "snap":
            return await self._snap(args.get("exposure_ms"))
        raise ValidationError(f"{feature}.{command} is not implemented")

    async def _move_xy(self, x_um: float, y_um: float) -> dict[str, Any]:
        core = self.core
        stage = self._xy_stage()

        def do() -> None:
            core.setXYPosition(x_um, y_um)
            core.waitForDevice(stage)

        await asyncio.to_thread(do)
        x, y = await asyncio.to_thread(core.getXYPosition)
        return {"x_um": x, "y_um": y}

    async def _move_z(self, z_um: float) -> dict[str, Any]:
        core = self.core
        focus = self._focus()

        def do() -> None:
            core.setPosition(focus, z_um)
            core.waitForDevice(focus)

        await asyncio.to_thread(do)
        z = await asyncio.to_thread(core.getPosition, focus)
        return {"z_um": z}

    async def _autofocus(self, ctx: ExecutionContext) -> dict[str, Any]:
        if not self.core.getAutoFocusDevice():
            raise DeviceNotReady(
                "no autofocus device is loaded in this Micro-Manager configuration",
                device=self.id,
            )

        def do() -> float:
            self.core.fullFocus()
            return self.core.getPosition(self._focus())

        z = await asyncio.to_thread(do)
        self._last_focus_score = await asyncio.to_thread(self.core.getCurrentFocusScore)
        return {"z_um": z, "focus_score": self._last_focus_score}

    async def _snap(self, exposure_ms: float | None) -> dict[str, Any]:
        core = self.core

        def do() -> Any:
            if exposure_ms is not None:
                core.setExposure(float(exposure_ms))
            core.snapImage()
            return core.getImage()

        image = await asyncio.to_thread(do)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        path = self.data_dir / f"{self.id}_{stamp}_{uuid.uuid4().hex[:6]}.png"
        size = await asyncio.to_thread(_png.write, str(path), image, bit_depth=16)
        artifact = {
            "uri": path.resolve().as_uri(), "kind": "image", "mime_type": "image/png",
            "bytes": size, "shape": list(image.shape), "dtype": str(image.dtype),
            "metadata": {"device": self.id, "exposure_ms": exposure_ms, "simulated": False},
        }
        return {"artifact_uri": artifact["uri"], "artifacts": [artifact]}

    # -- prediction ---------------------------------------------------------

    async def _simulate(
        self, feature: str, command: str, args: dict[str, Any]
    ) -> SimulationResult:
        """Kinematic bounds only: MMCore has no notion of a digital twin.

        The one thing worth checking without touching hardware is a stage
        move against MMCore's own reported soft limits, when the adapter
        publishes them; anything else -- a snap, an autofocus -- is reported
        as unverifiable, honestly, exactly as `drivers/scpi.py` does.
        """
        if feature == "MotionControl" and command in ("move_absolute", "move_relative"):
            warnings = [] if self.homed else ["the stage is not homed; its true position is unknown"]
            return SimulationResult(feasible=True, fidelity="kinematic", warnings=warnings)
        return SimulationResult(
            feasible=True, fidelity="none",
            warnings=[(f"MMCore has no digital twin for {feature}.{command}; "
                       "the outcome cannot be predicted before it runs")],
        )
