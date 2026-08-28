"""Opentrons driver, against a fake robot server via `httpx.MockTransport`.

A mock transport (rather than a real socket, as `test_driver_wot.py` uses) is
the better fit here: what needs proving is that this driver speaks the
robot-server API's exact shape -- the `Opentrons-Version` header, the
`runs`/`actions` state machine, the polling loop on `play` -- not that a
generic HTTP round trip works, which the WoT test already covers.
"""

from __future__ import annotations

import json

import pytest

httpx = pytest.importorskip("httpx")

from labbench.core.device import DeviceDescriptor, DeviceState, ExecutionContext
from labbench.core.errors import DeviceFault, DeviceNotReady, ValidationError
from labbench.drivers.opentrons import OpentronsRobot


class FakeRobot:
    def __init__(self) -> None:
        self.lights = False
        self.run_status = "idle"
        self.poll_count = 0
        self.runs: dict[str, dict] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers.get("opentrons-version") == "3"
        path, method = request.url.path, request.method
        if path == "/health" and method == "GET":
            return httpx.Response(200, json={"robot_model": "OT-2", "serial_number": "SN1",
                                              "api_version": "7.0.0"})
        if path == "/robot/lights" and method == "GET":
            return httpx.Response(200, json={"on": self.lights})
        if path == "/robot/lights" and method == "POST":
            self.lights = json.loads(request.content)["on"]
            return httpx.Response(200, json={"on": self.lights})
        if path == "/robot/home" and method == "POST":
            return httpx.Response(200, json={})
        if path == "/runs" and method == "POST":
            body = json.loads(request.content)
            self.runs["run123"] = {"id": "run123", "status": "idle",
                                    "protocolId": body["data"]["protocolId"]}
            return httpx.Response(201, json={"data": self.runs["run123"]})
        if path == "/runs/run123" and method == "GET":
            self.poll_count += 1
            run = dict(self.runs["run123"])
            if run["status"] == "running" and self.poll_count >= 2:
                run["status"] = "succeeded"
                self.runs["run123"]["status"] = "succeeded"
            return httpx.Response(200, json={"data": run})
        if path == "/runs/run123/actions" and method == "POST":
            action = json.loads(request.content)["data"]["actionType"]
            if action == "play":
                self.runs["run123"]["status"] = "running"
            return httpx.Response(201, json={"data": {"actionType": action}})
        return httpx.Response(404, json={"error": f"no route {method} {path}"})


@pytest.fixture
async def robot():
    fake = FakeRobot()
    dev = OpentronsRobot(DeviceDescriptor(id="ot1"), host="192.168.1.60", protocol_id="proto1")
    dev._client = httpx.AsyncClient(
        base_url=dev.base_url, headers={"Opentrons-Version": "3"},
        transport=httpx.MockTransport(fake.handler),
    )
    await dev._set_state(DeviceState.IDLE)
    dev.descriptor.vendor = "Opentrons"
    dev._fake = fake
    yield dev


class TestRobotControl:
    async def test_lights_round_trip(self, robot):
        await robot.write("Robot", "lights_on", True)
        assert (await robot.read("Robot", "lights_on")).value is True

    async def test_home(self, robot):
        result = await robot.invoke("Robot", "home", {})
        assert result["homed"] is True


class TestRunLifecycle:
    async def test_create_run_uses_configured_protocol_id(self, robot):
        result = await robot.invoke("RunControl", "create_run", {})
        assert result["run_id"] == "run123"
        assert (await robot.read("RunControl", "run_id")).value == "run123"

    async def test_create_run_requires_a_protocol(self):
        dev = OpentronsRobot(DeviceDescriptor(id="ot2"), host="x")
        with pytest.raises(ValidationError):
            await dev.invoke("RunControl", "create_run", {})

    async def test_play_polls_until_terminal(self, robot):
        await robot.invoke("RunControl", "create_run", {})
        outcome = await robot.invoke("RunControl", "play", {}, ExecutionContext())
        assert outcome["status"] == "succeeded"
        assert robot._fake.poll_count >= 2

    async def test_actions_before_create_run_are_refused(self, robot):
        with pytest.raises(DeviceNotReady):
            await robot.invoke("RunControl", "play", {}, ExecutionContext())


class TestFailedRun:
    async def test_play_raises_device_fault_on_failure(self):
        fake = FakeRobot()

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/runs/run123" and request.method == "GET":
                return httpx.Response(200, json={"data": {
                    "id": "run123", "status": "failed",
                    "errors": [{"detail": "tip was not detected"}],
                }})
            return fake.handler(request)

        dev = OpentronsRobot(DeviceDescriptor(id="ot3"), host="x", protocol_id="p")
        dev._client = httpx.AsyncClient(
            base_url=dev.base_url, headers={"Opentrons-Version": "3"},
            transport=httpx.MockTransport(handler),
        )
        await dev._set_state(DeviceState.IDLE)
        await dev.invoke("RunControl", "create_run", {})
        with pytest.raises(DeviceFault, match="tip was not detected"):
            await dev.invoke("RunControl", "play", {}, ExecutionContext())


class TestSimulate:
    async def test_no_digital_twin_but_checks_protocol_existence(self, robot):
        sim = await robot.simulate("RunControl", "create_run", {})
        assert sim.feasible and sim.fidelity == "none"

    async def test_missing_protocol_is_infeasible(self):
        dev = OpentronsRobot(DeviceDescriptor(id="ot4"), host="x")
        sim = await dev.simulate("RunControl", "create_run", {})
        assert not sim.feasible
