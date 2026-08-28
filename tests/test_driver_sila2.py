"""SiLA2 driver, against a duck-typed fake `SilaClient`.

A *real* SiLA2 server needs code generated from a Feature Definition (FDL) by
`sila2-codegen`, which pulls in `black`/`jinja2`/`isort` -- fine for a one-off
manual check, too heavy to make every CI run depend on. So this test fakes
the one seam that matters: the parsed-FDL object graph `SilaClient` builds at
connect time (`feature._unobservable_commands`, `.parameters.fields`, and so
on -- the exact shapes read directly from the installed `sila2` package's
source while writing `drivers/sila2_adapter.py`). What is *not* faked is this
driver's own projection and invocation logic, which is what these tests
actually exercise.
"""

from __future__ import annotations

from typing import Any

import pytest

sila2 = pytest.importorskip("sila2")

from labbench.core.device import DeviceDescriptor, ExecutionContext
from labbench.core.errors import CapabilityNotFound, DeviceFault, ValidationError
from labbench.drivers.sila2_adapter import Sila2Device

# -- fakes matching the real sila2.framework/client object shapes -----------


class Integer:
    pass


class Real:
    pass


class Boolean:
    pass


class String:
    pass


class FakeField:
    """Stands in for a `NamedDataNode` (Parameter/Response)."""

    def __init__(self, identifier: str, data_type: Any, description: str = "") -> None:
        self._identifier = identifier
        self._description = description
        self.data_type = data_type


class FakeFields:
    def __init__(self, fields: list[FakeField]) -> None:
        self.fields = fields


class FakeCommand:
    """Stands in for `feature._unobservable_commands[name]` (a framework Command)."""

    def __init__(self, identifier, description, parameters, responses) -> None:
        self._identifier = identifier
        self._description = description
        self.parameters = FakeFields(parameters)
        self.responses = FakeFields(responses)


class FakeProperty:
    def __init__(self, description, data_type) -> None:
        self._description = description
        self.data_type = data_type


class FakeCallableCommand:
    """Stands in for `getattr(feature, name)` -- the callable client wrapper."""

    def __init__(self, fn) -> None:
        self._fn = fn

    def __call__(self, **kwargs):
        return self._fn(**kwargs)


class Response:
    """A minimal stand-in for the NamedTuple a real command response is."""

    def __init__(self, **fields) -> None:
        self._fields = fields

    def _asdict(self) -> dict:
        return dict(self._fields)


class FakeFeature:
    def __init__(self, identifier: str, fqid: str) -> None:
        self._identifier = identifier
        self._display_name = identifier
        self._description = f"{identifier} test feature"
        self.fully_qualified_identifier = fqid
        self._unobservable_commands: dict[str, FakeCommand] = {}
        self._observable_commands: dict[str, FakeCommand] = {}
        self._unobservable_properties: dict[str, FakeProperty] = {}
        self._observable_properties: dict[str, FakeProperty] = {}
        self._client_commands: dict[str, FakeCallableCommand] = {}

    def add_command(self, name, fn, parameters, responses, description="") -> None:
        self._unobservable_commands[name] = FakeCommand(name, description, parameters, responses)
        setattr(self, name, FakeCallableCommand(fn))
        self._client_commands[name] = getattr(self, name)

    def add_property(self, name, value, data_type, description="") -> None:
        self._unobservable_properties[name] = FakeProperty(description, data_type)

        class _Prop:
            def get(_self):
                return value

        setattr(self, name, _Prop())


class FakeSiLAServiceInfo:
    ServerUUID = type("_UUID", (), {"get": staticmethod(lambda: "11111111-1111-1111-1111-111111111111")})()


def build_temperature_feature() -> FakeFeature:
    feature = FakeFeature("TemperatureController", "com.example/testing/TemperatureController/v1")
    feature.add_property("CurrentTemperature", 21.5, Real(), "Current chamber temperature.")

    def set_target(**kwargs):
        return Response(Accepted=True)

    feature.add_command(
        "SetTargetTemperature", set_target,
        parameters=[FakeField("Target", Real(), "Target temperature.")],
        responses=[FakeField("Accepted", Boolean())],
        description="Set the target temperature.",
    )

    def crash(**kwargs):
        raise RuntimeError("heater fault")

    feature.add_command("Crash", crash, parameters=[], responses=[])
    return feature


def build_core_service_feature() -> FakeFeature:
    """A stand-in for the mandatory SiLAService feature, which must be filtered out."""
    return FakeFeature("SiLAService", "org.silastandard/core/SiLAService/v1")


class FakeSilaClient:
    def __init__(self, host, port, **kwargs) -> None:
        self.host, self.port = host, port
        temp = build_temperature_feature()
        self._features = {
            "SiLAService": build_core_service_feature(),
            "TemperatureController": temp,
        }
        self.SiLAService = FakeSiLAServiceInfo()


@pytest.fixture
def dev(monkeypatch):
    monkeypatch.setattr("sila2.client.SilaClient", FakeSilaClient)
    return Sila2Device(DeviceDescriptor(id="sila1"), host="127.0.0.1", port=50052, insecure=True)


class TestCapabilityModel:
    async def test_core_service_feature_is_filtered_out(self, dev):
        await dev.connect()
        assert "SiLAService" not in dev.features()

    async def test_feature_projection(self, dev):
        await dev.connect()
        feature = dev.features()["TemperatureController"]
        assert feature.namespace == "com.example.testing"
        assert feature.version == "1"
        prop = feature.property("CurrentTemperature")
        assert prop.schema_.type == "number"
        cmd = feature.command("SetTargetTemperature")
        assert cmd.parameters[0].name == "Target"
        assert cmd.parameters[0].type == "number"


class TestDataPlane:
    async def test_read_property(self, dev):
        await dev.connect()
        sample = await dev.read("TemperatureController", "CurrentTemperature")
        assert sample.value == 21.5

    async def test_invoke_command_returns_named_fields(self, dev):
        await dev.connect()
        result = await dev.invoke(
            "TemperatureController", "SetTargetTemperature", {"Target": 37.0}, ExecutionContext(),
        )
        assert result == {"Accepted": True}

    async def test_server_exception_becomes_device_fault(self, dev):
        await dev.connect()
        with pytest.raises(DeviceFault):
            await dev.invoke("TemperatureController", "Crash", {}, ExecutionContext())

    async def test_unknown_feature_raises(self, dev):
        # Device.resolve() (the base class) gates this before the driver's
        # own _find_feature ever runs.
        await dev.connect()
        with pytest.raises(CapabilityNotFound):
            await dev.invoke("NoSuchFeature", "X", {}, ExecutionContext())


class TestSimulate:
    async def test_no_digital_twin(self, dev):
        await dev.connect()
        sim = await dev.simulate("TemperatureController", "SetTargetTemperature", {"Target": 10.0})
        assert sim.fidelity == "none"


class TestConfiguration:
    def test_missing_host_is_rejected(self):
        with pytest.raises(ValidationError):
            Sila2Device(DeviceDescriptor(id="d"))
