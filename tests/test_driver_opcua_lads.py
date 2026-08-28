"""OPC UA LADS driver, against a real in-process `asyncua.Server`.

`asyncua` ships both a client and a server, so this is a genuine protocol
round trip rather than a mock: a small LADS-shaped address space
(DeviceSet/Identification/FunctionalUnitSet, one Variable, one Method with
declared InputArguments) is built for real, and the driver connects, browses
it, and drives it exactly as it would a vendor's server.
"""

from __future__ import annotations

import asyncio

import pytest

asyncua = pytest.importorskip("asyncua")
from asyncua import ua

from labbench.core.device import DeviceDescriptor, ExecutionContext
from labbench.core.errors import CapabilityNotFound, ValidationError
from labbench.drivers.opcua_lads import LadsDevice

#: asyncua does not rewrite its stored endpoint URL to reflect an OS-assigned
#: port after binding, so a fixed port is used rather than ":0" -- safe here
#: since the suite runs one test process at a time and each server is fully
#: torn down (the fixture's `async with`) before the next test's starts.
ENDPOINT = "opc.tcp://127.0.0.1:48410"


@pytest.fixture
async def lads_server():
    server = asyncua.Server()
    await server.init()
    server.set_endpoint(ENDPOINT)
    idx = await server.register_namespace("http://labbench.test/lads/")

    objects = server.get_objects_node()
    device_set = await objects.add_object(idx, "DeviceSet")
    device = await device_set.add_object(idx, "TestDosingPump")

    ident = await device.add_object(idx, "Identification")
    await ident.add_variable(idx, "Manufacturer", "LabBench Testing")
    await ident.add_variable(idx, "Model", "DP-1")
    await ident.add_variable(idx, "SerialNumber", "SN-0001")
    await ident.add_variable(idx, "DeviceRevision", "1.0.0")

    unit_set = await device.add_object(idx, "FunctionalUnitSet")
    dosing = await unit_set.add_object(idx, "Dosing")

    flow_var = await dosing.add_variable(idx, "FlowRate", 0.0, varianttype=ua.VariantType.Double)
    await flow_var.set_writable()
    await flow_var.add_property(
        0, "EngineeringUnits", ua.EUInformation(DisplayName=ua.LocalizedText("mL/min"))
    )
    state_var = await dosing.add_variable(idx, "State", "idle", varianttype=ua.VariantType.String)
    await state_var.set_writable()
    # Deliberately left read-only, to exercise AccessLevel enforcement.
    await dosing.add_variable(idx, "PumpModel", "DP-1-head", varianttype=ua.VariantType.String)

    async def start_cb(parent, volume):
        # asyncua's low-level method dispatch passes raw ua.Variant objects,
        # not unwrapped Python values, to a callback registered this way.
        await state_var.write_value(f"dosing {float(volume.Value)} mL")
        return [ua.Variant(True, ua.VariantType.Boolean)]

    in_arg = ua.Argument()
    in_arg.Name, in_arg.ValueRank, in_arg.ArrayDimensions = "Volume", -1, []
    in_arg.DataType = ua.NodeId(ua.ObjectIds.Double)
    in_arg.Description = ua.LocalizedText("Volume to dose, mL")
    out_arg = ua.Argument()
    out_arg.Name, out_arg.ValueRank, out_arg.ArrayDimensions = "Accepted", -1, []
    out_arg.DataType = ua.NodeId(ua.ObjectIds.Boolean)
    out_arg.Description = ua.LocalizedText("Whether the dose was accepted")
    await dosing.add_method(idx, "Start", start_cb, [in_arg], [out_arg])

    async def stop_cb(parent):
        await state_var.write_value("idle")

    await dosing.add_method(idx, "Stop", stop_cb, [], [])

    async with server:
        await asyncio.sleep(0.2)  # let the endpoint actually start listening
        yield server, state_var


@pytest.fixture
async def dosing_pump(lads_server):
    server, _ = lads_server
    endpoint = server.endpoint.geturl()
    dev = LadsDevice(DeviceDescriptor(id="lads1"), endpoint_url=endpoint)
    await dev.connect()
    try:
        yield dev
    finally:
        await dev.disconnect()


class TestDiscovery:
    async def test_identification_populates_the_descriptor(self, dosing_pump):
        assert dosing_pump.descriptor.vendor == "LabBench Testing"
        assert dosing_pump.descriptor.model == "DP-1"
        assert dosing_pump.descriptor.serial == "SN-0001"

    async def test_functional_unit_becomes_a_feature(self, dosing_pump):
        assert "Dosing" in dosing_pump.features()

    async def test_variables_carry_type_and_unit(self, dosing_pump):
        feature = dosing_pump.features()["Dosing"]
        flow = feature.property("FlowRate")
        assert flow.schema_.type == "number"
        assert flow.schema_.unit == "mL/min"
        assert flow.writable is True
        state = feature.property("State")
        assert state.schema_.type == "string"
        assert state.writable is True  # explicitly set_writable() in the fixture
        model = feature.property("PumpModel")
        assert model.writable is False  # never set_writable()

    async def test_methods_become_commands(self, dosing_pump):
        feature = dosing_pump.features()["Dosing"]
        assert {"Start", "Stop"}.issubset({c.name for c in feature.commands})


class TestDataPlane:
    async def test_read_and_write_variable(self, dosing_pump):
        assert (await dosing_pump.read("Dosing", "FlowRate")).value == 0.0
        await dosing_pump.write("Dosing", "FlowRate", 12.5)
        assert (await dosing_pump.read("Dosing", "FlowRate")).value == 12.5

    async def test_write_to_read_only_variable_is_refused(self, dosing_pump):
        # The base Device.write() rejects this before the driver's own
        # AccessLevel check ever runs, since `writable=False` was already
        # projected from the server's AccessLevel at discovery time.
        with pytest.raises(CapabilityNotFound):
            await dosing_pump.write("Dosing", "PumpModel", "hacked")

    async def test_call_method_with_named_arguments(self, dosing_pump):
        result = await dosing_pump.invoke(
            "Dosing", "Start", {"args": {"Volume": 5.0}}, ExecutionContext(),
        )
        assert result["output"] is True
        assert (await dosing_pump.read("Dosing", "State")).value == "dosing 5.0 mL"

    async def test_call_method_missing_argument_raises(self, dosing_pump):
        with pytest.raises(ValidationError):
            await dosing_pump.invoke("Dosing", "Start", {"args": {}}, ExecutionContext())

    async def test_call_method_with_no_arguments(self, dosing_pump):
        await dosing_pump.invoke("Dosing", "Start", {"args": {"Volume": 1.0}}, ExecutionContext())
        await dosing_pump.invoke("Dosing", "Stop", {}, ExecutionContext())
        assert (await dosing_pump.read("Dosing", "State")).value == "idle"


class TestEstop:
    async def test_estop_calls_stop_on_every_functional_unit_that_has_it(self, dosing_pump):
        await dosing_pump.invoke("Dosing", "Start", {"args": {"Volume": 1.0}}, ExecutionContext())
        await dosing_pump.estop("test")
        assert (await dosing_pump.read("Dosing", "State")).value == "idle"


class TestSimulate:
    async def test_no_digital_twin(self, dosing_pump):
        sim = await dosing_pump.simulate("Dosing", "Start", {"args": {"Volume": 1.0}})
        assert sim.fidelity == "none"


class TestConfiguration:
    def test_missing_endpoint_url_is_rejected(self):
        with pytest.raises(ValidationError):
            LadsDevice(DeviceDescriptor(id="d"))
