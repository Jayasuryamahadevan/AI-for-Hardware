"""SCPI instruments: oscilloscopes, DMMs, power supplies, function generators.

SCPI is the closest thing test-and-measurement has to a universal language, and
it is the driver that makes "any hardware" more than a slogan: an enormous
amount of real laboratory equipment already speaks it.

Two decisions shape this driver.

**Raw TCP is the default, not VISA.** Most modern instruments expose SCPI on a
socket (port 5025 by convention), and that path needs nothing installed at all.
PyVISA is used when it is present and when the address needs it -- GPIB, USBTMC
and serial all do -- but requiring a VISA runtime to talk to an oscilloscope
that is sitting on the network would be an artificial barrier.

**The capability model comes from a profile, not from code.** SCPI is a
grammar, not a schema: `MEAS:VOLT:DC?` means the same thing on every DMM, but
nothing in the protocol says an instrument has that command or what units it
answers in. So a profile maps LabBench properties and commands onto SCPI
strings, and profiles are data -- built in for common instrument classes,
overridable in the lab configuration. Adding support for a new instrument is
usually a YAML block, not a Python module.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from typing import Any

from ..core.capability import (
    Command,
    Constraint,
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
    DriverUnavailable,
    TransportError,
    ValidationError,
)

log = logging.getLogger("labbench.scpi")

#: The conventional SCPI-over-TCP port. Instruments that differ say so.
DEFAULT_SCPI_PORT = 5025
#: SCPI is line-oriented; instruments differ on whether they want \n or \r\n.
DEFAULT_TERMINATOR = "\n"


class ScpiTransport:
    """Raw SCPI over a TCP socket. No dependencies.

    Written against asyncio streams rather than PyVISA because a networked
    instrument needs nothing more, and because a blocking VISA call inside an
    event loop would stall every other device on the bench.
    """

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_SCPI_PORT,
        *,
        terminator: str = DEFAULT_TERMINATOR,
        timeout_s: float = 10.0,
        encoding: str = "ascii",
    ) -> None:
        self.host = host
        self.port = port
        self.terminator = terminator
        self.timeout_s = timeout_s
        self.encoding = encoding
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        # One command at a time. SCPI has no request ids, so a reply belongs to
        # whoever asked last; overlapping queries would silently cross wires.
        self._lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def open(self) -> None:
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), self.timeout_s
            )
        except (TimeoutError, OSError) as exc:
            raise TransportError(
                f"cannot reach the instrument at {self.host}:{self.port}: {exc}",
                host=self.host, port=self.port,
            ) from None

    async def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except (OSError, RuntimeError):
                pass
        self._reader = self._writer = None

    async def write(self, command: str) -> None:
        if self._writer is None:
            raise TransportError("not connected", host=self.host)
        async with self._lock:
            self._writer.write((command + self.terminator).encode(self.encoding))
            await self._writer.drain()

    async def query(self, command: str) -> str:
        if self._reader is None or self._writer is None:
            raise TransportError("not connected", host=self.host)
        async with self._lock:
            self._writer.write((command + self.terminator).encode(self.encoding))
            await self._writer.drain()
            try:
                raw = await asyncio.wait_for(self._reader.readline(), self.timeout_s)
            except TimeoutError:
                # A timed-out query is the dangerous case: the instrument may
                # still answer later, and that answer would be read as the
                # reply to the *next* query. The link is closed rather than
                # left desynchronised.
                await self.close()
                raise TransportError(
                    f"{command!r} did not answer within {self.timeout_s}s; the connection "
                    "has been closed because a late reply would be mistaken for the next "
                    "query's answer",
                    command=command, host=self.host,
                ) from None
        if not raw:
            raise TransportError(f"instrument closed the connection during {command!r}",
                                 command=command, host=self.host)
        return raw.decode(self.encoding, errors="replace").strip()


class VisaTransport:
    """SCPI through PyVISA, for GPIB, USBTMC and serial instruments.

    PyVISA is synchronous, so every call is pushed to a worker thread. A
    blocking GPIB read on the event loop would freeze the whole gateway,
    including the e-stop.
    """

    def __init__(self, resource: str, *, timeout_s: float = 10.0, **kwargs: Any) -> None:
        self.resource = resource
        self.timeout_s = timeout_s
        self.kwargs = kwargs
        self._instrument: Any = None
        self._lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._instrument is not None

    async def open(self) -> None:
        try:
            import pyvisa
        except ImportError:
            raise DriverUnavailable(
                f"the address {self.resource!r} needs a VISA backend. "
                "Install it with: pip install 'labbench[scpi]'. "
                "Networked instruments can use a plain host:port address instead, "
                "which needs nothing.",
                driver="scpi", resource=self.resource,
            ) from None

        def connect() -> Any:
            manager = pyvisa.ResourceManager(self.kwargs.pop("visa_library", ""))
            instrument = manager.open_resource(self.resource, **self.kwargs)
            instrument.timeout = int(self.timeout_s * 1000)
            return instrument

        try:
            self._instrument = await asyncio.to_thread(connect)
        except Exception as exc:  # noqa: BLE001 - pyvisa can raise almost anything here
            raise TransportError(
                f"cannot open VISA resource {self.resource!r}: {exc}",
                resource=self.resource,
            ) from None

    async def close(self) -> None:
        if self._instrument is not None:
            await asyncio.to_thread(self._instrument.close)
        self._instrument = None

    async def write(self, command: str) -> None:
        if self._instrument is None:
            raise TransportError("not connected", resource=self.resource)
        async with self._lock:
            await asyncio.to_thread(self._instrument.write, command)

    async def query(self, command: str) -> str:
        if self._instrument is None:
            raise TransportError("not connected", resource=self.resource)
        async with self._lock:
            try:
                reply = await asyncio.to_thread(self._instrument.query, command)
            except Exception as exc:  # noqa: BLE001 - pyvisa can raise almost anything here
                raise TransportError(
                    f"{command!r} failed on {self.resource}: {exc}",
                    command=command, resource=self.resource,
                ) from None
        return str(reply).strip()


# -- profiles --------------------------------------------------------------

#: Built-in instrument-class profiles.
#:
#: Each profile maps LabBench capabilities onto SCPI strings. `{}` in a command
#: template is substituted with the argument. Everything here is standard SCPI
#: from the IEEE 488.2 and SCPI-99 common command sets, so these work across
#: vendors far more often than not -- and where a vendor differs, the fix is a
#: profile override in YAML rather than a code change.
PROFILES: dict[str, dict[str, Any]] = {
    "dmm": {
        "display_name": "Digital Multimeter",
        "feature": "Measurement",
        "namespace": "org.labbench.measurement",
        "properties": {
            "voltage_dc": {"query": "MEAS:VOLT:DC?", "unit": "V", "type": "number"},
            "voltage_ac": {"query": "MEAS:VOLT:AC?", "unit": "V", "type": "number"},
            "current_dc": {"query": "MEAS:CURR:DC?", "unit": "A", "type": "number"},
            "resistance": {"query": "MEAS:RES?", "unit": "ohm", "type": "number"},
        },
        "commands": {
            "measure": {
                "description": "Take one reading of the selected function.",
                "write": "MEAS:{function}?",
                "query": True,
                "parameters": {
                    "function": {
                        "type": "string", "default": "VOLT:DC",
                        "enum": ["VOLT:DC", "VOLT:AC", "CURR:DC", "CURR:AC", "RES", "FREQ"],
                    }
                },
                "hazard": "none",
            },
        },
    },
    "power_supply": {
        "display_name": "DC Power Supply",
        "feature": "PowerOutput",
        "namespace": "org.labbench.power",
        "properties": {
            "voltage_setpoint": {"query": "SOUR:VOLT?", "write": "SOUR:VOLT {}",
                                 "unit": "V", "type": "number", "writable": True},
            "current_limit": {"query": "SOUR:CURR?", "write": "SOUR:CURR {}",
                              "unit": "A", "type": "number", "writable": True},
            "voltage_measured": {"query": "MEAS:VOLT?", "unit": "V", "type": "number"},
            "current_measured": {"query": "MEAS:CURR?", "unit": "A", "type": "number"},
            "output_enabled": {"query": "OUTP?", "type": "boolean"},
        },
        "commands": {
            "set_voltage": {
                "description": "Set the output voltage setpoint.",
                "write": "SOUR:VOLT {voltage}",
                "parameters": {"voltage": {"type": "number", "unit": "V",
                                           "minimum": 0.0, "maximum": 30.0}},
                # Energising a circuit can destroy a device under test.
                "hazard": "benign", "reversibility": "reversible",
            },
            "set_current_limit": {
                "description": "Set the compliance current.",
                "write": "SOUR:CURR {current}",
                "parameters": {"current": {"type": "number", "unit": "A",
                                           "minimum": 0.0, "maximum": 5.0}},
                "hazard": "benign", "reversibility": "reversible",
            },
            "enable_output": {
                "description": "Energise the output terminals.",
                "write": "OUTP ON",
                # Current starts flowing into whatever is connected.
                "hazard": "thermal", "reversibility": "reversible",
                "inverse": "disable_output", "tags": ["energises_circuit"],
            },
            "disable_output": {
                "description": "De-energise the output terminals.",
                "write": "OUTP OFF", "hazard": "benign",
                "reversibility": "reversible",
            },
        },
    },
    "oscilloscope": {
        "display_name": "Oscilloscope",
        "feature": "Acquisition",
        "namespace": "org.labbench.measurement",
        "properties": {
            "timebase_s_per_div": {"query": "TIM:SCAL?", "write": "TIM:SCAL {}",
                                   "unit": "s", "type": "number", "writable": True},
            "trigger_level": {"query": "TRIG:LEV?", "write": "TRIG:LEV {}",
                              "unit": "V", "type": "number", "writable": True},
        },
        "commands": {
            "autoscale": {
                "description": "Let the instrument choose timebase and vertical scale.",
                "write": "AUT", "hazard": "benign",
                "reversibility": "restorable",
            },
            "single": {
                "description": "Arm for a single acquisition.",
                "write": "SING", "hazard": "none", "reversibility": "reversible",
            },
            "measure": {
                "description": "Read one automatic measurement from a channel.",
                "write": "MEAS:{measurement}? CHAN{channel}", "query": True,
                "parameters": {
                    "measurement": {"type": "string", "default": "VPP",
                                    "enum": ["VPP", "VAVG", "VRMS", "FREQ", "PER",
                                             "VMAX", "VMIN", "RIS", "FALL"]},
                    "channel": {"type": "integer", "default": 1,
                                "minimum": 1, "maximum": 4},
                },
                "hazard": "none",
            },
        },
    },
    "function_generator": {
        "display_name": "Function Generator",
        "feature": "Waveform",
        "namespace": "org.labbench.power",
        "properties": {
            "frequency": {"query": "SOUR:FREQ?", "write": "SOUR:FREQ {}",
                          "unit": "Hz", "type": "number", "writable": True},
            "amplitude": {"query": "SOUR:VOLT?", "write": "SOUR:VOLT {}",
                          "unit": "Vpp", "type": "number", "writable": True},
            "output_enabled": {"query": "OUTP?", "type": "boolean"},
        },
        "commands": {
            "set_waveform": {
                "description": "Select the output waveform shape.",
                "write": "SOUR:FUNC {shape}",
                "parameters": {"shape": {"type": "string", "default": "SIN",
                                         "enum": ["SIN", "SQU", "RAMP", "PULS", "NOIS", "DC"]}},
                "hazard": "benign", "reversibility": "reversible",
            },
            "enable_output": {
                "description": "Enable the signal output.",
                "write": "OUTP ON", "hazard": "benign",
                "reversibility": "reversible", "inverse": "disable_output",
            },
            "disable_output": {
                "description": "Disable the signal output.",
                "write": "OUTP OFF", "hazard": "none",
                "reversibility": "reversible",
            },
        },
    },
    "generic": {
        "display_name": "SCPI Instrument",
        "feature": "Scpi",
        "namespace": "org.labbench.scpi",
        "properties": {},
        "commands": {},
    },
}

_HAZARDS = {h.value: h for h in Hazard}
_REVERSIBILITY = {r.value: r for r in Reversibility}


class ScpiInstrument(Device):
    """One SCPI instrument, described by a profile.

    Configuration:

        driver: scpi
        settings:
          address: 192.168.1.50:5025      # or a VISA resource string
          profile: oscilloscope           # a built-in class, or omit for generic
          terminator: "\\n"
          timeout_s: 10
          profile_overrides:              # merged over the built-in profile
            commands:
              measure:
                write: "MEASU:IMM:VAL? CH{channel}"
    """

    requires_package = None  # only VISA addresses need one, and they say so

    def __init__(self, descriptor: DeviceDescriptor, **config: Any) -> None:
        super().__init__(descriptor, **config)
        self.address = str(config.get("address", "")).strip()
        if not self.address:
            raise ValidationError(
                "an SCPI device needs an address: either 'host:port' for a networked "
                "instrument or a VISA resource string such as "
                "'GPIB0::12::INSTR' / 'USB0::0x0699::0x0363::C000001::INSTR'",
                device=descriptor.id,
            )
        self.profile_name = str(config.get("profile", "generic"))
        if self.profile_name not in PROFILES:
            raise ValidationError(
                f"unknown SCPI profile {self.profile_name!r}; built in: "
                f"{sorted(PROFILES)}. Use 'generic' plus profile_overrides for an "
                "instrument that does not fit a class.",
                profile=self.profile_name, available=sorted(PROFILES),
            )
        self.profile = _merge(
            PROFILES[self.profile_name], config.get("profile_overrides", {}) or {}
        )
        self.timeout_s = float(config.get("timeout_s", 10.0))
        self.transport = self._build_transport(config)
        self.identity: str = ""
        #: Reported by *IDN?, split into the four SCPI-mandated fields.
        self.manufacturer = self.model = self.serial = self.firmware = ""

    def _build_transport(self, config: dict[str, Any]) -> Any:
        """Choose a transport from the shape of the address."""
        if _looks_like_visa(self.address):
            return VisaTransport(self.address, timeout_s=self.timeout_s,
                                 **config.get("visa", {}))
        host, _, port = self.address.partition(":")
        return ScpiTransport(
            host, int(port) if port else DEFAULT_SCPI_PORT,
            terminator=config.get("terminator", DEFAULT_TERMINATOR),
            timeout_s=self.timeout_s,
            encoding=config.get("encoding", "ascii"),
        )

    # -- lifecycle --------------------------------------------------------

    async def _connect(self) -> None:
        await self.transport.open()
        try:
            self.identity = await self.transport.query("*IDN?")
        except TransportError:
            await self.transport.close()
            raise
        # *IDN? is mandated to return manufacturer,model,serial,firmware. Not
        # every instrument obeys; a short answer is recorded rather than
        # rejected, because refusing to talk to a slightly non-compliant box
        # would be worse than not knowing its serial number.
        fields = [f.strip() for f in self.identity.split(",")]
        self.manufacturer, self.model, self.serial, self.firmware = (
            fields + ["", "", "", ""]
        )[:4]
        self.descriptor.vendor = self.manufacturer or self.descriptor.vendor
        self.descriptor.model = self.model or self.descriptor.model
        self.descriptor.serial = self.serial or self.descriptor.serial
        self.descriptor.firmware = self.firmware or self.descriptor.firmware
        self.descriptor.protocol = "scpi"
        log.info("%s identified as %s", self.id, self.identity)
        await self._check_errors()

    async def _disconnect(self) -> None:
        await self.transport.close()

    async def _initialize(self, ctx: ExecutionContext) -> None:
        """*RST then *CLS: a known state, and an empty error queue."""
        await self.transport.write("*RST")
        await asyncio.sleep(0.5)
        await self.transport.write("*CLS")
        # *OPC? blocks until the reset has actually finished. Without it the
        # next command can arrive mid-reset and be discarded.
        try:
            await self.transport.query("*OPC?")
        except TransportError:
            log.warning("%s did not answer *OPC? after reset", self.id)

    async def _estop(self) -> None:
        """Make the instrument safe, then stop.

        For anything that sources power this means the output off *first*.
        There is no SCPI universal e-stop, so this is best-effort and says so:
        a failure here is logged loudly rather than swallowed.
        """
        for command in ("OUTP OFF", "SOUR:VOLT 0", "ABOR", "*CLS"):
            try:
                await self.transport.write(command)
            except Exception as exc:  # noqa: BLE001 - never block an e-stop
                log.warning("%s: e-stop command %r failed: %s", self.id, command, exc)

    async def _check_errors(self) -> None:
        """Drain the SCPI error queue.

        An instrument that rejected a command usually answers the *next* query
        perfectly well while quietly holding the error, so an agent sees
        plausible numbers from a machine that ignored its instruction. Draining
        the queue is what turns that into a visible failure.
        """
        for _ in range(10):
            try:
                reply = await self.transport.query("SYST:ERR?")
            except TransportError:
                return  # not every instrument implements it
            if not reply:
                return
            code = reply.split(",")[0].strip()
            try:
                if int(float(code)) == 0:
                    return
            except ValueError:
                return
            await self.emit("Scpi", "instrument_error", {"error": reply}, severity="error")
            raise DeviceFault(
                f"the instrument reported: {reply}", device=self.id, scpi_error=reply
            )

    # -- capability model -------------------------------------------------

    def _features(self) -> Sequence[Feature]:
        feature_id = self.profile.get("feature", "Scpi")
        namespace = self.profile.get("namespace", "org.labbench.scpi")

        properties = [
            Property(
                name=name,
                description=spec.get("description", f"SCPI: {spec.get('query', '')}"),
                schema=Parameter(
                    name=name, type=spec.get("type", "number"),
                    unit=spec.get("unit"),
                    constraint=_constraint(spec),
                ),
                writable=bool(spec.get("write")) and spec.get("writable", False),
            )
            for name, spec in self.profile.get("properties", {}).items()
        ]

        commands = [
            self._build_command(name, spec)
            for name, spec in self.profile.get("commands", {}).items()
        ]

        # Every SCPI instrument gets the raw escape hatch. It is the honest
        # answer to a protocol with no schema: no profile will ever cover every
        # instrument, and an agent that can read the programming manual should
        # not be blocked by a gap in ours.
        commands.append(
            Command(
                name="raw_query",
                description="Send an arbitrary SCPI query and return the reply verbatim. "
                            "Use when the profile does not cover what you need. The "
                            "instrument's own programming manual is the reference.",
                parameters=[
                    Parameter(name="command", type="string",
                              description="SCPI query, e.g. 'MEAS:VOLT:DC?'."),
                ],
                returns=[Parameter(name="reply", type="string")],
                duration_estimate_s=0.5,
                # A query is a query only by convention; the driver cannot know
                # what an arbitrary string does, so it is never treated as free.
                hazard=Hazard.BENIGN, reversibility=Reversibility.RESTORABLE,
                simulatable=False, tags={"raw", "unverifiable"},
            )
        )
        commands.append(
            Command(
                name="raw_write",
                description="Send an arbitrary SCPI command with no reply expected. "
                            "The driver cannot predict what this does to the instrument.",
                parameters=[Parameter(name="command", type="string")],
                duration_estimate_s=0.3,
                hazard=Hazard.THERMAL, reversibility=Reversibility.IRREVERSIBLE,
                simulatable=False, tags={"raw", "unverifiable"},
            )
        )

        return [
            Feature(
                identifier=feature_id,
                display_name=self.profile.get("display_name", "SCPI Instrument"),
                description=f"{self.identity or self.address} ({self.profile_name})",
                namespace=namespace,
                properties=properties,
                commands=commands,
                events=[
                    Event(name="instrument_error",
                          description="The SCPI error queue reported a fault.",
                          severity="error"),
                ],
            )
        ]

    def _build_command(self, name: str, spec: dict[str, Any]) -> Command:
        parameters = [
            Parameter(
                name=pname,
                type=pspec.get("type", "number"),
                unit=pspec.get("unit"),
                description=pspec.get("description", ""),
                default=pspec.get("default"),
                required=pspec.get("default") is None,
                constraint=_constraint(pspec),
            )
            for pname, pspec in spec.get("parameters", {}).items()
        ]
        return Command(
            name=name,
            description=spec.get("description", ""),
            parameters=parameters,
            returns=[Parameter(name="reply", type="string")] if spec.get("query") else [],
            duration_estimate_s=float(spec.get("duration_estimate_s", 0.5)),
            hazard=_HAZARDS.get(spec.get("hazard", "benign"), Hazard.BENIGN),
            reversibility=_REVERSIBILITY.get(
                spec.get("reversibility", "restorable"), Reversibility.RESTORABLE
            ),
            inverse=spec.get("inverse"),
            # Simulatable, but only in the sense that the driver can say
            # something useful about the command - never that it can predict
            # the outcome. `_simulate` answers fidelity "none", which the
            # safety kernel escalates exactly as it would a non-simulatable
            # command, while still delivering the warnings a profile *can*
            # justify. Marking it unsimulatable would make the base class
            # short-circuit and throw those away.
            simulatable=True,
            tags=set(spec.get("tags", [])),
        )

    # -- data plane -------------------------------------------------------

    async def _read(self, feature: str, name: str) -> Any:
        spec = self.profile.get("properties", {}).get(name)
        if spec is None or "query" not in spec:
            raise ValidationError(
                f"{name!r} has no SCPI query in the {self.profile_name!r} profile",
                property=name, profile=self.profile_name,
            )
        reply = await self.transport.query(spec["query"])
        return _coerce(reply, spec.get("type", "number"))

    async def _write(self, feature: str, name: str, value: Any) -> None:
        spec = self.profile.get("properties", {}).get(name)
        if spec is None or "write" not in spec:
            raise ValidationError(
                f"{name!r} is not writable in the {self.profile_name!r} profile",
                property=name,
            )
        await self.transport.write(spec["write"].format(value))
        await self._check_errors()

    async def _invoke(
        self, feature: str, command: str, args: dict[str, Any], ctx: ExecutionContext
    ) -> Any:
        if command == "raw_query":
            reply = await self.transport.query(args["command"])
            return {"reply": reply, "command": args["command"]}
        if command == "raw_write":
            await self.transport.write(args["command"])
            await self._check_errors()
            return {"sent": args["command"]}

        spec = self.profile.get("commands", {}).get(command)
        if spec is None:
            raise ValidationError(
                f"{command!r} is not in the {self.profile_name!r} profile",
                command=command, profile=self.profile_name,
            )
        template = spec.get("write", "")
        try:
            rendered = template.format(**args)
        except KeyError as exc:
            raise ValidationError(
                f"the {command!r} template needs {exc} but it was not supplied",
                command=command, template=template,
            ) from None

        if spec.get("query"):
            reply = await self.transport.query(rendered)
            await self._check_errors()
            return {"reply": reply, "value": _coerce(reply, "number"), "command": rendered}
        await self.transport.write(rendered)
        await self._check_errors()
        return {"sent": rendered}

    async def _simulate(
        self, feature: str, command: str, args: dict[str, Any]
    ) -> SimulationResult:
        """No digital twin, and saying so is the point.

        A profile maps names onto strings; it has no model of what the
        instrument will do with them. Returning fidelity "none" makes the
        safety kernel escalate anything hazardous to a human instead of
        accepting a fabricated prediction.
        """
        warnings = [
            (f"the SCPI driver has no model of {self.model or 'this instrument'}; "
             "the outcome of this command cannot be predicted before it is sent")
        ]
        spec = self.profile.get("commands", {}).get(command, {})
        if "energises_circuit" in set(spec.get("tags", [])):
            warnings.append(
                "this command energises the output; whatever is connected will draw current"
            )
        return SimulationResult(feasible=True, fidelity="none", warnings=warnings)


# -- helpers ---------------------------------------------------------------

_VISA_PATTERN = re.compile(
    r"^(GPIB|USB|ASRL|TCPIP|PXI|VXI|FIREWIRE)\d*::", re.IGNORECASE
)


def _looks_like_visa(address: str) -> bool:
    """True for a VISA resource string, false for host:port."""
    return bool(_VISA_PATTERN.match(address)) or address.upper().endswith("::INSTR")


def _coerce(reply: str, kind: str) -> Any:
    """Turn a SCPI reply into a Python value.

    SCPI answers booleans as "1"/"0"/"ON"/"OFF" depending on vendor and mood,
    and numbers in whatever notation the instrument prefers. Falling back to
    the raw string is deliberate: a value we cannot parse is still worth
    surfacing verbatim rather than replaced with a zero.
    """
    text = reply.strip()
    if kind == "boolean":
        return text.upper() in ("1", "ON", "TRUE", "YES")
    if kind == "integer":
        try:
            return int(float(text))
        except ValueError:
            return text
    if kind == "number":
        try:
            return float(text)
        except ValueError:
            return text
    return text


def _constraint(spec: dict[str, Any]) -> Constraint:
    return Constraint(
        minimum=spec.get("minimum"), maximum=spec.get("maximum"),
        enum=spec.get("enum"),
    )


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge a profile override over a built-in profile.

    Deep rather than shallow so a lab can retarget one command's SCPI string
    without restating the whole instrument class.
    """
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out
