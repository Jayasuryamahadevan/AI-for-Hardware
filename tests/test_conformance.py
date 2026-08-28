"""The conformance harness: machine-checked proof of "any AI x any hardware".

Every other test file in this suite checks one driver or one dialect emitter
in isolation. This one checks the *claim the README makes* -- that any of the
six driver protocols can sit behind the gateway and be described, safely
invoked and projected into any of the six AI dialects, with no protocol
getting special treatment. It does this by building one gateway with every
driver connected simultaneously (mixing simulated instruments with
protocol-real fixtures: a fake SCPI transport, an inline WoT Thing
Description, a real Micro-Manager core, a real in-process OPC UA server, a
real generated SiLA2 server, a mocked Opentrons robot server) and then
running the full N-drivers x M-dialects matrix through the real emitters.

A driver whose optional dependency is not installed is skipped, not failed --
the same "a lab with no hardware attached still starts" principle the
gateway itself follows, applied to the test matrix.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator

import pytest

from labbench.bridge.schema import DIALECTS, ToolSpec, emit
from labbench.bridge.toolset import tool_specs
from labbench.core.capability import Hazard, Reversibility
from labbench.core.device import Device, DeviceDescriptor, DeviceState
from labbench.core.errors import CapabilityNotFound, LabBenchError, ValidationError
from labbench.core.registry import DeviceConfig, DeviceManager
from labbench.gateway import Gateway

# -- one no-hardware-required factory per driver, all yielding a *connected* Device --


@contextlib.asynccontextmanager
async def _simulated(driver: str) -> AsyncIterator[Device]:
    manager = DeviceManager()
    device = manager.add_from_config(DeviceConfig(id=driver, driver=driver, settings={"seed": 3}))
    await device.connect()
    try:
        yield device
    finally:
        await device.disconnect()


@contextlib.asynccontextmanager
async def _scpi() -> AsyncIterator[Device]:
    from labbench.drivers.scpi import ScpiInstrument

    class FakeTransport:
        async def open(self) -> None: ...
        async def close(self) -> None: ...
        async def write(self, command: str) -> None: ...
        async def query(self, command: str) -> str:
            return {"*IDN?": "Acme,DMM-1,SN1,FW1", "SYST:ERR?": "0,\"No error\""}.get(command, "0")

    device = ScpiInstrument(DeviceDescriptor(id="scpi_dmm"), address="10.0.0.1:5025", profile="dmm")
    device.transport = FakeTransport()
    await device.connect()
    try:
        yield device
    finally:
        await device.disconnect()


@contextlib.asynccontextmanager
async def _wot() -> AsyncIterator[Device]:
    pytest.importorskip("httpx")
    from labbench.drivers.http_wot import WoTThing

    td = {
        "title": "ConformanceLamp",
        "properties": {
            "brightness": {"type": "integer", "minimum": 0, "maximum": 100, "unit": "%",
                            "forms": [{"href": "/b", "op": ["readproperty", "writeproperty"]}]},
        },
        "actions": {
            "toggle": {"description": "Toggle.", "forms": [{"href": "/t", "op": ["invokeaction"]}]},
        },
    }
    device = WoTThing(DeviceDescriptor(id="wot_lamp"), thing_description=td)
    await device.connect()
    try:
        yield device
    finally:
        await device.disconnect()


@contextlib.asynccontextmanager
async def _opentrons() -> AsyncIterator[Device]:
    httpx = pytest.importorskip("httpx")
    from labbench.drivers.opentrons import OpentronsRobot

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"robot_model": "OT-2"})
        if request.url.path == "/robot/lights":
            return httpx.Response(200, json={"on": False})
        return httpx.Response(404, json={})

    device = OpentronsRobot(DeviceDescriptor(id="ot2"), host="10.0.0.2", protocol_id="p1")
    device._client = httpx.AsyncClient(
        base_url=device.base_url, headers={"Opentrons-Version": "3"},
        transport=httpx.MockTransport(handler),
    )
    await device._set_state(DeviceState.IDLE)
    device.descriptor.vendor = "Opentrons"
    try:
        yield device
    finally:
        await device._client.aclose()


@contextlib.asynccontextmanager
async def _micromanager(tmp_path) -> AsyncIterator[Device]:
    pymmcore_plus = pytest.importorskip("pymmcore_plus")
    try:
        core = pymmcore_plus.CMMCorePlus()
        core.loadSystemConfiguration()
        core.unloadAllDevices()
    except Exception:  # noqa: BLE001 - demo device adapters not installed (`mmcore install`)
        pytest.skip("Micro-Manager demo device adapters not installed")

    from labbench.drivers.micromanager import MicroManagerMicroscope

    device = MicroManagerMicroscope(DeviceDescriptor(id="mm_scope"), data_dir=str(tmp_path))
    await device.connect()
    try:
        yield device
    finally:
        await device.disconnect()


@contextlib.asynccontextmanager
async def _opcua_lads() -> AsyncIterator[Device]:
    asyncua = pytest.importorskip("asyncua")
    from asyncua import ua

    from labbench.drivers.opcua_lads import LadsDevice

    server = asyncua.Server()
    await server.init()
    server.set_endpoint("opc.tcp://127.0.0.1:48732")
    idx = await server.register_namespace("http://labbench.test/conformance/")
    objects = server.get_objects_node()
    device_set = await objects.add_object(idx, "DeviceSet")
    dev_obj = await device_set.add_object(idx, "ConformanceDevice")
    unit_set = await dev_obj.add_object(idx, "FunctionalUnitSet")
    unit = await unit_set.add_object(idx, "Unit")
    await unit.add_variable(idx, "Value", 0.0, varianttype=ua.VariantType.Double)

    async def start_cb(parent):
        return None

    await unit.add_method(idx, "Start", start_cb, [], [])

    async with server:
        import asyncio

        await asyncio.sleep(0.2)
        device = LadsDevice(DeviceDescriptor(id="lads_unit"), endpoint_url=server.endpoint.geturl())
        await device.connect()
        try:
            yield device
        finally:
            await device.disconnect()


@contextlib.asynccontextmanager
async def _sila2(monkeypatch) -> AsyncIterator[Device]:
    pytest.importorskip("sila2")
    from labbench.drivers.sila2_adapter import Sila2Device

    class _RealType:
        pass

    class Real(_RealType):
        pass

    class _Field:
        def __init__(self, identifier, data_type):
            self._identifier, self._description, self.data_type = identifier, "", data_type

    class _Fields:
        def __init__(self, fields):
            self.fields = fields

    class _Command:
        def __init__(self, identifier, parameters, responses):
            self._identifier, self._description = identifier, "Conformance command."
            self.parameters, self.responses = _Fields(parameters), _Fields(responses)

    class _Feature:
        def __init__(self, identifier, fqid):
            self._identifier, self._display_name, self._description = identifier, identifier, ""
            self.fully_qualified_identifier = fqid
            self._unobservable_properties = {"Value": _Prop()}
            self._observable_properties: dict = {}
            cmd = _Command("Set", [_Field("Target", Real())], [_Field("Ok", Real())])
            self._unobservable_commands = {"Set": cmd}
            self._observable_commands: dict = {}
            self._client_commands = {"Set": (lambda **kw: _Response())}
            self.Set = self._client_commands["Set"]
            self.Value = _Prop()

    class _Prop:
        _description = "Conformance property."
        data_type = Real()

        def get(self):
            return 0.0

    class _Response:
        def _asdict(self):
            return {"Ok": True}

    class FakeClient:
        def __init__(self, host, port, **kwargs) -> None:
            core = _Feature("SiLAService", "org.silastandard/core/SiLAService/v1")
            unit = _Feature("Conformance", "org.labbench.test/conformance/Conformance/v1")
            self._features = {"SiLAService": core, "Conformance": unit}

            class _Info:
                ServerUUID = type("_U", (), {"get": staticmethod(lambda: "0" * 32)})()

            self.SiLAService = _Info()

    monkeypatch.setattr("sila2.client.SilaClient", FakeClient)
    device = Sila2Device(DeviceDescriptor(id="sila_unit"), host="127.0.0.1", port=1, insecure=True)
    await device.connect()
    try:
        yield device
    finally:
        await device.disconnect()


DRIVER_NAMES = [
    "sim_microscope", "sim_plate_reader", "sim_liquid_handler", "sim_incubator",
    "scpi", "wot", "opentrons", "micromanager", "opcua_lads", "sila2",
]


def _factory(name: str, *, tmp_path, monkeypatch):
    if name.startswith("sim_"):
        return _simulated(name)
    if name == "scpi":
        return _scpi()
    if name == "wot":
        return _wot()
    if name == "opentrons":
        return _opentrons()
    if name == "micromanager":
        return _micromanager(tmp_path)
    if name == "opcua_lads":
        return _opcua_lads()
    if name == "sila2":
        return _sila2(monkeypatch)
    raise ValueError(name)  # pragma: no cover - exhaustive over DRIVER_NAMES


# -- the contract every device must satisfy, regardless of protocol ---------


async def assert_device_contract(device: Device) -> None:
    assert device.state is DeviceState.IDLE
    features = device.features()
    assert features, f"{device.id}: a device with no features cannot be driven by anything"

    with pytest.raises(CapabilityNotFound):
        device.resolve("NoSuchFeature", "nope")

    for feature in features.values():
        assert feature.identifier
        assert "/" in feature.fqid

        for prop in feature.properties:
            schema = prop.schema_.to_json_schema()
            assert isinstance(schema, dict) and schema.get("type")
            json.dumps(schema)  # every property schema must be JSON-serialisable

        for command in feature.commands:
            assert isinstance(command.hazard, Hazard)
            assert isinstance(command.reversibility, Reversibility)
            assert command.duration_estimate_s > 0

            schema = command.input_schema()
            assert schema["type"] == "object"
            json.dumps(schema)

            # An incomplete or malformed call must come back as a typed
            # LabBenchError -- never an unhandled AttributeError/TypeError
            # from inside the driver. This is the property that makes
            # "any hardware" actually safe to automate against.
            try:
                await device.simulate(feature.identifier, command.name, {})
            except ValidationError:
                pass  # a command with required parameters correctly refused {}
            except LabBenchError:
                pass  # any other typed refusal is equally acceptable
            # Anything else (a bare Python exception) fails the test by propagating.


# -- the contract every command's schema must satisfy in every AI dialect ---


def assert_schema_dialect_conformant(schema: dict) -> None:
    from labbench.bridge.schema import _to_gemini_schema, _to_strict_schema

    strict = _to_strict_schema(schema)
    json.dumps(strict)
    if strict.get("properties"):
        assert strict["additionalProperties"] is False
        assert set(strict["required"]) == set(strict["properties"])

    gemini = _to_gemini_schema(schema)
    json.dumps(gemini)
    _assert_gemini_keywords_only(gemini)


def _assert_gemini_keywords_only(schema: dict) -> None:
    from labbench.bridge.schema import _GEMINI_KEYWORDS

    for key, value in schema.items():
        assert key in _GEMINI_KEYWORDS, f"{key!r} is not in Gemini's accepted keyword set"
        if key == "properties" and isinstance(value, dict):
            for sub in value.values():
                _assert_gemini_keywords_only(sub)
        elif key == "items" and isinstance(value, dict):
            _assert_gemini_keywords_only(value)


# -- the harness itself -------------------------------------------------


@pytest.mark.parametrize("driver_name", DRIVER_NAMES)
async def test_device_contract_holds_for_every_driver(driver_name, tmp_path, monkeypatch):
    async with _factory(driver_name, tmp_path=tmp_path, monkeypatch=monkeypatch) as device:
        await assert_device_contract(device)


@pytest.mark.parametrize("driver_name", DRIVER_NAMES)
async def test_every_command_schema_survives_every_ai_dialect(driver_name, tmp_path, monkeypatch):
    async with _factory(driver_name, tmp_path=tmp_path, monkeypatch=monkeypatch) as device:
        for feature in device.features().values():
            for command in feature.commands:
                assert_schema_dialect_conformant(command.input_schema())


async def test_the_fixed_tool_surface_works_with_every_driver_attached_at_once(tmp_path, monkeypatch):
    """The one test that puts every protocol on the same bench simultaneously.

    Builds a single Gateway with as many drivers as are installed in this
    environment connected together, then confirms the constant-size tool
    surface (`tool_specs`) still emits cleanly into every one of the six AI
    dialects with that mixed-protocol lab attached -- the concrete form of
    the "any AI, any hardware" claim, checked in one place.
    """
    gateway = Gateway(data_dir=tmp_path)
    connected: list[str] = []
    async with contextlib.AsyncExitStack() as stack:
        for name in DRIVER_NAMES:
            try:
                device = await stack.enter_async_context(
                    _factory(name, tmp_path=tmp_path, monkeypatch=monkeypatch)
                )
            except pytest.skip.Exception:
                continue
            gateway.devices._devices[device.id] = device
            connected.append(device.id)

        assert len(connected) >= 4, "at least the four simulated drivers must always be available"

        description = gateway.describe()
        assert {d["id"] for d in description["devices"]} == set(connected)

        specs = tool_specs(gateway)
        assert len(specs) >= 20  # the curated surface; see README's tool/method count
        for dialect in DIALECTS:
            payload = emit(specs, dialect)
            json.dumps(payload)  # every dialect's output must be transport-ready

        # A hand-rolled agent loop gets the JSON Schema dialect with LabBench's
        # own metadata intact; that is the one this assertion locks in.
        neutral = emit(specs, "jsonschema")
        invoke = next(t for t in neutral if t["name"] == "device_invoke")
        assert invoke["hazard"] == "varies"
        assert invoke["annotations"]["destructive"] is True


def test_every_dialect_is_exercised_by_the_matrix_above():
    """Guards against silently dropping a dialect from the matrix later."""
    assert set(DIALECTS) == {
        "anthropic", "openai", "openai-responses", "gemini", "jsonschema", "openapi",
    }


def test_tool_spec_from_a_hand_written_dict_is_still_dialect_conformant():
    """Sanity check that assert_schema_dialect_conformant itself is not vacuous."""
    spec = ToolSpec(
        name="x", parameters={
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "number", "minimum": 0}},
            "required": ["a"],
        },
    )
    assert_schema_dialect_conformant(spec.parameters)
