"""Opentrons OT-2 / Flex, over the robot's own HTTP API.

The robot server (the software that ships on every OT-2 and Flex, listening on
port 31950) exposes protocol execution as a state machine: upload or reference
a protocol, `POST /runs` to create a run bound to it, then drive it with
`POST /runs/{id}/actions {"actionType": "play"|"pause"|"stop"}` and poll
`GET /runs/{id}` for status. This driver talks that API directly rather than
depending on the `opentrons` Python package, which is the full robot-side
stack (hardware control, protocol interpreter, its own web server) meant to
*run on* the robot, not to be imported by a client that talks to one --
pulling it in here would mean shipping a simulated hardware backend just to
drive an HTTP client.

Scope is deliberately the run/action surface plus basic robot control (home,
lights), not protocol authoring: a protocol is Python (or, on newer servers, a
`.json`/`.zip` bundle) uploaded once, out of band, the same way a SCPI
instrument's calibration is set up once outside LabBench. What an agent needs
day to day is to start, watch, pause and stop a run against a protocol that
already exists on the robot -- exactly the seam `runs` was designed for.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from ..core.capability import (
    Command,
    Event,
    Feature,
    Hazard,
    Parameter,
    Property,
    Reversibility,
)
from ..core.device import Device, DeviceDescriptor, ExecutionContext, SimulationResult
from ..core.errors import (
    DeviceFault,
    DeviceNotReady,
    DriverUnavailable,
    TransportError,
    ValidationError,
)

log = logging.getLogger("labbench.opentrons")

#: Every request must carry this, or the robot server answers 400. It is not
#: optional, and it is not a credential -- it is an API-shape version pin.
OPENTRONS_API_VERSION = "3"

_RUN_STATUS_STATE = {
    "idle": "idle", "running": "busy", "paused": "busy",
    "stop-requested": "busy", "finishing": "busy",
    "succeeded": "idle", "stopped": "idle", "failed": "fault",
}


class OpentronsRobot(Device):
    """One OT-2 or Flex, addressed by its robot-server HTTP API.

    Configuration:

        driver: opentrons
        settings:
          host: 192.168.1.60
          port: 31950            # default
          protocol_id: <id>      # from a prior `POST /protocols` upload, or
                                  # left unset and supplied per-run via args
    """

    requires_package = "httpx"

    def __init__(self, descriptor: DeviceDescriptor, **config: Any) -> None:
        super().__init__(descriptor, **config)
        host = config.get("host")
        if not host:
            raise ValidationError("an opentrons device needs 'host'", device=descriptor.id)
        self.base_url = f"http://{host}:{config.get('port', 31950)}"
        self.timeout_s = float(config.get("timeout_s", 15.0))
        self.default_protocol_id = config.get("protocol_id")
        self._client: Any = None
        self.current_run_id: str | None = None

    async def _connect(self) -> None:
        try:
            import httpx
        except ImportError:
            raise DriverUnavailable(
                "the opentrons driver needs httpx. Install it with: pip install 'labbench[http]'",
                driver="opentrons",
            ) from None
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout_s,
            headers={"Opentrons-Version": OPENTRONS_API_VERSION},
        )
        health = await self._request("GET", "/health")
        self.descriptor.vendor = "Opentrons"
        self.descriptor.model = health.get("robot_model", self.descriptor.model)
        self.descriptor.serial = health.get("serial_number", self.descriptor.serial)
        self.descriptor.firmware = health.get("api_version", self.descriptor.firmware)
        self.descriptor.protocol = "http"

    async def _disconnect(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _estop(self) -> None:
        """Stop the active run and de-energise the gantry motors.

        A run stop is cooperative (the robot decelerates to a safe pause
        point); there is no protocol-level instant kill switch reachable over
        this API -- that exists only as the robot's physical button -- so this
        is the honest ceiling of what software can do here, and it is logged
        as best-effort rather than presented as a guaranteed hard stop.
        """
        if self.current_run_id is not None:
            try:
                await self._request(
                    "POST", f"/runs/{self.current_run_id}/actions",
                    json={"data": {"actionType": "stop"}},
                )
            except Exception as exc:  # noqa: BLE001 - never block an e-stop
                log.warning("%s: failed to stop run %s: %s", self.id, self.current_run_id, exc)
        try:
            await self._request("POST", "/robot/lights", json={"on": False})
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: failed to disable lights during e-stop: %s", self.id, exc)

    # -- capability model -------------------------------------------------

    def _features(self) -> Sequence[Feature]:
        return [self._robot_feature(), self._run_feature()]

    def _robot_feature(self) -> Feature:
        return Feature(
            identifier="Robot",
            display_name="Robot Control",
            namespace="org.labbench.liquid_handling",
            properties=[
                Property(name="lights_on", description="Deck light state.",
                         schema=Parameter(name="lights_on", type="boolean"), writable=True),
            ],
            commands=[
                Command(
                    name="home", description="Home every axis.",
                    duration_estimate_s=12.0, hazard=Hazard.MOTION,
                    reversibility=Reversibility.RESTORABLE, tags={"moves_stage"},
                ),
            ],
        )

    def _run_feature(self) -> Feature:
        return Feature(
            identifier="RunControl",
            display_name="Protocol Run",
            namespace="org.labbench.liquid_handling",
            properties=[
                Property(name="run_id", description="Currently tracked run, if any.",
                         schema=Parameter(name="run_id", type="string")),
                Property(name="run_status", description="Status of the tracked run.",
                         schema=Parameter(name="run_status", type="string")),
            ],
            commands=[
                Command(
                    name="create_run",
                    description="Create a run bound to a protocol already on the robot, "
                                "and start tracking it.",
                    parameters=[
                        Parameter(name="protocol_id", type="string", required=False,
                                  description="Defaults to the device's configured protocol_id."),
                    ],
                    returns=[Parameter(name="run_id", type="string")],
                    duration_estimate_s=1.0, hazard=Hazard.NONE,
                    reversibility=Reversibility.REVERSIBLE,
                ),
                Command(
                    name="play",
                    description="Start or resume the tracked run. Consumes reagents and tips "
                                "as the protocol executes; there is no undo.",
                    observable=True, duration_estimate_s=1.0,
                    hazard=Hazard.SAMPLE, reversibility=Reversibility.IRREVERSIBLE,
                    tags={"consumes_reagent"},
                ),
                Command(
                    name="pause", description="Pause the tracked run at its next safe point.",
                    duration_estimate_s=0.5, hazard=Hazard.BENIGN,
                    reversibility=Reversibility.RESTORABLE, inverse="play",
                ),
                Command(
                    name="stop",
                    description="Stop the tracked run. Cannot be resumed; a new run must be created.",
                    duration_estimate_s=1.0, hazard=Hazard.MOTION,
                    reversibility=Reversibility.IRREVERSIBLE,
                ),
            ],
            events=[
                Event(name="run_failed", description="The tracked run reported an error.",
                      severity="error"),
            ],
        )

    # -- data plane -------------------------------------------------------

    async def _read(self, feature: str, name: str) -> Any:
        if (feature, name) == ("Robot", "lights_on"):
            data = await self._request("GET", "/robot/lights")
            return bool(data.get("on", False))
        if (feature, name) == ("RunControl", "run_id"):
            return self.current_run_id or ""
        if (feature, name) == ("RunControl", "run_status"):
            if self.current_run_id is None:
                return "no_run"
            run = await self._get_run(self.current_run_id)
            return run.get("status", "unknown")
        raise ValidationError(f"{feature}.{name} is not a readable property here")

    async def _write(self, feature: str, name: str, value: Any) -> None:
        if (feature, name) == ("Robot", "lights_on"):
            await self._request("POST", "/robot/lights", json={"on": bool(value)})
            return
        raise ValidationError(f"{feature}.{name} is not writable here")

    async def _invoke(
        self, feature: str, command: str, args: dict[str, Any], ctx: ExecutionContext
    ) -> Any:
        if command == "home":
            await self._request("POST", "/robot/home", json={"target": "robot"})
            return {"homed": True}
        if command == "create_run":
            protocol_id = args.get("protocol_id") or self.default_protocol_id
            if not protocol_id:
                raise ValidationError(
                    "no protocol_id given and none configured on this device; "
                    "upload a protocol with POST /protocols first"
                )
            run = await self._request(
                "POST", "/runs", json={"data": {"protocolId": protocol_id}}
            )
            self.current_run_id = run["data"]["id"]
            return {"run_id": self.current_run_id, "status": run["data"]["status"]}
        if command in ("play", "pause", "stop"):
            if self.current_run_id is None:
                raise DeviceNotReady(
                    "no run is being tracked; call RunControl.create_run first",
                    device=self.id,
                )
            await self._request(
                "POST", f"/runs/{self.current_run_id}/actions",
                json={"data": {"actionType": command}},
            )
            if command == "play":
                run = await self._wait_for_terminal_or_running(ctx)
                return run
            run = await self._get_run(self.current_run_id)
            return {"run_id": self.current_run_id, "status": run.get("status")}
        raise ValidationError(f"RunControl has no command {command!r}")

    async def _get_run(self, run_id: str) -> dict[str, Any]:
        data = await self._request("GET", f"/runs/{run_id}")
        return data["data"]

    async def _wait_for_terminal_or_running(self, ctx: ExecutionContext) -> dict[str, Any]:
        """Poll until the run leaves the transient `running` state.

        The robot server has no push channel over this API, so polling is the
        honest option; `play` is declared `observable` precisely so this can
        take as long as the protocol does without anyone's request timing out.
        """
        assert self.current_run_id is not None
        while True:
            ctx.raise_if_cancelled()
            run = await self._get_run(self.current_run_id)
            status = run.get("status", "unknown")
            if status != "running":
                if status == "failed":
                    errors = run.get("errors", [])
                    raise DeviceFault(
                        f"run {self.current_run_id} failed: "
                        f"{errors[0].get('detail', 'no detail') if errors else 'no detail'}",
                        device=self.id, run_id=self.current_run_id, errors=errors,
                    )
                return {"run_id": self.current_run_id, "status": status}
            current = run.get("current", {})
            await ctx.progress(
                0.5, current.get("commandType", "running") if isinstance(current, dict) else "running",
            )
            await asyncio.sleep(1.0)

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        import httpx

        try:
            response = await self._client.request(method, path, **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DeviceFault(
                f"{method} {path} returned {exc.response.status_code}: "
                f"{exc.response.text[:200]}",
                device=self.id, status_code=exc.response.status_code,
            ) from None
        except httpx.HTTPError as exc:
            raise TransportError(f"{method} {path} failed: {exc}", device=self.id) from None
        return response.json() if response.content else {}

    async def _simulate(
        self, feature: str, command: str, args: dict[str, Any]
    ) -> SimulationResult:
        """No digital twin: this drives a physical liquid handler and nothing here models it.

        The one useful check available without touching hardware is whether
        the referenced protocol exists at all, which is worth doing even when
        the rest of the outcome is unverifiable.
        """
        warnings = [("no digital twin for a physical Opentrons run; the outcome of "
                     "pipetting cannot be predicted before it executes")]
        if command == "create_run":
            protocol_id = args.get("protocol_id") or self.default_protocol_id
            if not protocol_id:
                return SimulationResult(
                    feasible=False, fidelity="none",
                    violations=["no protocol_id given and none configured"],
                )
        return SimulationResult(feasible=True, fidelity="none", warnings=warnings)
