"""SCPI driver: profile mapping, transport selection, error-queue draining.

Uses a fake transport (matching the small `open/close/write/query` surface
`ScpiTransport`/`VisaTransport` share) rather than a real socket or PyVISA
session, the same way the driver's own design separates the profile logic
from the transport so either can be tested alone.
"""

from __future__ import annotations

import pytest

from labbench.core.device import DeviceDescriptor
from labbench.core.errors import DeviceFault, ValidationError
from labbench.drivers.scpi import ScpiInstrument, _looks_like_visa


class FakeTransport:
    def __init__(self) -> None:
        self.written: list[str] = []
        self.queries: dict[str, str] = {
            "*IDN?": "Acme,DMM-1,SN123,FW1.0",
            "SYST:ERR?": "0,\"No error\"",
            "MEAS:VOLT:DC?": "3.30000",
        }
        self.connected = True

    async def open(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    async def write(self, command: str) -> None:
        self.written.append(command)

    async def query(self, command: str) -> str:
        self.written.append(command)
        return self.queries.get(command, "0")


@pytest.fixture
async def dmm():
    dev = ScpiInstrument(
        DeviceDescriptor(id="dmm1"), address="192.168.1.50:5025", profile="dmm",
    )
    dev.transport = FakeTransport()
    await dev.connect()
    try:
        yield dev
    finally:
        await dev.disconnect()


class TestTransportSelection:
    def test_host_port_uses_raw_tcp(self):
        from labbench.drivers.scpi import ScpiTransport

        dev = ScpiInstrument(DeviceDescriptor(id="d"), address="10.0.0.5:5025")
        assert isinstance(dev.transport, ScpiTransport)

    def test_visa_resource_string_is_detected(self):
        assert _looks_like_visa("GPIB0::12::INSTR")
        assert _looks_like_visa("USB0::0x0699::0x0363::C000001::INSTR")
        assert not _looks_like_visa("192.168.1.50:5025")

    def test_missing_address_is_rejected(self):
        with pytest.raises(ValidationError):
            ScpiInstrument(DeviceDescriptor(id="d"))

    def test_unknown_profile_is_rejected(self):
        with pytest.raises(ValidationError):
            ScpiInstrument(DeviceDescriptor(id="d"), address="x:5025", profile="not_a_profile")


class TestConnect:
    async def test_idn_populates_the_descriptor(self, dmm):
        assert dmm.descriptor.vendor == "Acme"
        assert dmm.descriptor.model == "DMM-1"
        assert dmm.descriptor.serial == "SN123"
        assert dmm.descriptor.firmware == "FW1.0"


class TestDataPlane:
    async def test_read_property_from_profile_query(self, dmm):
        value = await dmm.read("Measurement", "voltage_dc")
        assert value.value == pytest.approx(3.3)

    async def test_read_unknown_property_raises(self, dmm):
        with pytest.raises(ValidationError):
            await dmm._read("Measurement", "not_a_real_property")

    async def test_command_with_enum_parameter(self, dmm):
        result = await dmm.invoke("Measurement", "measure", {"function": "VOLT:DC"})
        assert result["value"] == pytest.approx(3.3)
        assert "MEAS:VOLT:DC?" in dmm.transport.written

    async def test_raw_query_bypasses_the_profile(self, dmm):
        dmm.transport.queries["SYST:VERS?"] = "1999.0"
        result = await dmm.invoke("Measurement", "raw_query", {"command": "SYST:VERS?"})
        assert result["reply"] == "1999.0"


class TestErrorQueue:
    async def test_instrument_error_raises_device_fault(self, dmm):
        dmm.transport.queries["SYST:ERR?"] = '-113,"Undefined header"'
        with pytest.raises(DeviceFault):
            await dmm.invoke("Measurement", "raw_write", {"command": "BOGUS"})


class TestSimulate:
    async def test_scpi_has_no_digital_twin(self, dmm):
        sim = await dmm.simulate("Measurement", "measure", {"function": "VOLT:DC"})
        assert sim.fidelity == "none"

    async def test_energising_tag_is_surfaced_as_a_warning(self):
        dev = ScpiInstrument(
            DeviceDescriptor(id="psu1"), address="10.0.0.6:5025", profile="power_supply",
        )
        dev.transport = FakeTransport()
        await dev.connect()
        try:
            sim = await dev.simulate("PowerOutput", "enable_output", {})
            assert any("energises" in w for w in sim.warnings)
        finally:
            await dev.disconnect()


class TestProfileOverrides:
    def test_profile_override_deep_merges(self):
        dev = ScpiInstrument(
            DeviceDescriptor(id="d"), address="x:5025", profile="dmm",
            profile_overrides={"properties": {"voltage_dc": {"unit": "mV"}}},
        )
        assert dev.profile["properties"]["voltage_dc"]["unit"] == "mV"
        # Untouched keys survive the merge.
        assert dev.profile["properties"]["voltage_dc"]["query"] == "MEAS:VOLT:DC?"
        assert "current_dc" in dev.profile["properties"]
