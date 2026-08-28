"""SiLA 2 (Standardization in Lab Automation), through the `sila2` client library.

SiLA 2 is the one southbound protocol built on the *exact* triple LabBench
adopted -- Feature, Command, (Unobservable/Observable) Property -- so unlike
SCPI this driver invents almost nothing. `sila2.client.SilaClient` already
does the interesting part: on connect it calls the server's own
`SiLAService.GetFeatureDefinition` for every feature it implements, parses the
returned Feature Definition Language (FDL) XML, and builds one `ClientFeature`
object per feature with a Python attribute for every command and property,
named exactly as the server named them. This driver's job is projecting
*that* onto LabBench's model -- reading the same parsed FDL structures the
client already built, not re-discovering anything.

What FDL cannot express, and what this driver therefore cannot get from
introspection: hazard class and reversibility, corrected the same way as
every other self-describing protocol here (`drivers/http_wot.py`,
`drivers/opcua_lads.py`) -- `profile_overrides` in the lab configuration.

SiLA basic types (String, Integer, Real, Boolean, Date, Time, Timestamp) map
onto LabBench's scalar parameter types directly. A List, Structure,
Constrained or Any type is passed through as a LabBench `object` parameter
rather than translated field-by-field: SiLA's Structure can nest arbitrarily
and carry its own custom data type definitions, and pretending to flatten
that generically would produce schemas that looked precise and were not. The
server itself still validates the real value when the call is made; this
driver is honest about what it can check before that point.

`SilaClient` is entirely synchronous (built on grpc's sync channel), so every
call here runs via `asyncio.to_thread` -- the same reason `drivers/scpi.py`
does it for PyVISA.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from ..core.capability import Command, Feature, Hazard, Parameter, Property, Reversibility
from ..core.device import Device, DeviceDescriptor, ExecutionContext, SimulationResult
from ..core.errors import DeviceFault, DriverUnavailable, TransportError, ValidationError

_HAZARDS = {h.value: h for h in Hazard}
_REVERSIBILITY = {r.value: r for r in Reversibility}

#: SiLA Basic data type name -> LabBench Parameter type. Anything not listed
#: here (List, Structure, Constrained-of-those, Binary, Any) becomes "object".
_TYPE_MAP = {
    "Boolean": "boolean", "Integer": "integer", "Real": "number",
    "String": "string", "Date": "string", "Time": "string", "Timestamp": "string",
}


def _sila_type_name(data_type: Any) -> str:
    """Best-effort SiLA basic type name for one FDL data type node."""
    # Constrained types wrap a base type; unwrap one level, which covers the
    # overwhelming majority of real feature definitions (a constrained List
    # or Structure is rare and falls through to "object" below regardless).
    base = getattr(data_type, "base_type", data_type)
    return type(base).__name__


def _parameter_from(node: Any) -> Parameter:
    """Project one SiLA Parameter/Response (a NamedDataNode) into a LabBench Parameter."""
    kind = _TYPE_MAP.get(_sila_type_name(node.data_type), "object")
    return Parameter(
        name=node._identifier, description=node._description, type=kind,
        required=True,
    )


class Sila2Device(Device):
    """One SiLA 2 server, projected feature-for-feature onto the LabBench model.

    Configuration:

        driver: sila2
        settings:
          host: 192.168.1.80
          port: 50052
          insecure: true          # or root_certs / cert_chain / private_key (PEM bytes)
          profile_overrides:
            features:
              TemperatureController:
                commands:
                  SetTargetTemperature: {hazard: thermal, reversibility: reversible}
    """

    requires_package = "sila2"

    def __init__(self, descriptor: DeviceDescriptor, **config: Any) -> None:
        super().__init__(descriptor, **config)
        self.host = config.get("host")
        if not self.host:
            raise ValidationError("a sila2 device needs 'host'", device=descriptor.id)
        self.port = int(config.get("port", 50052))
        self.insecure = bool(config.get("insecure", False))
        self.root_certs = config.get("root_certs")
        self.private_key = config.get("private_key")
        self.cert_chain = config.get("cert_chain")
        self.overrides = config.get("profile_overrides", {}).get("features", {})
        self.client: Any = None

    async def _connect(self) -> None:
        try:
            from sila2.client import SilaClient
        except ImportError:
            raise DriverUnavailable(
                "the sila2 driver needs the sila2 package. Install it with: "
                "pip install 'labbench[sila2]'",
                driver="sila2",
            ) from None

        def build() -> Any:
            return SilaClient(
                self.host, self.port, insecure=self.insecure,
                root_certs=self.root_certs, private_key=self.private_key,
                cert_chain=self.cert_chain,
            )

        try:
            self.client = await asyncio.to_thread(build)
        except Exception as exc:  # noqa: BLE001 - grpc/sila2 can raise almost anything on connect
            raise TransportError(
                f"cannot connect to SiLA server at {self.host}:{self.port}: {exc}",
                host=self.host, port=self.port,
            ) from None

        info = self.client.SiLAService
        try:
            self.descriptor.serial = str(await asyncio.to_thread(info.ServerUUID.get))
        except Exception:  # noqa: BLE001, S110 - ServerUUID is mandatory but be defensive anyway
            pass
        self.descriptor.protocol = "sila2"

    async def _disconnect(self) -> None:
        self.client = None  # SilaClient has no explicit close; the channel is GC'd

    async def _estop(self) -> None:
        """Best-effort: SiLA 2 defines no universal e-stop.

        Some servers implement a custom `Stop`/`Abort` command per feature;
        this calls one wherever a feature happens to have it, and reports
        failures rather than swallowing them -- the same honesty every other
        driver's best-effort e-stop uses.
        """
        import logging

        log = logging.getLogger("labbench.sila2")
        for identifier, feature in self.client._features.items():
            for name in ("Stop", "Abort"):
                command = feature._client_commands.get(name)
                if command is None:
                    continue
                try:
                    await asyncio.to_thread(command)
                except Exception as exc:  # noqa: BLE001 - never block an e-stop
                    log.warning("%s: %s.%s failed during e-stop: %s", self.id, identifier, name, exc)

    # -- capability model -------------------------------------------------

    def _features(self) -> Sequence[Feature]:
        out = []
        for feature in self.client._features.values():
            if feature.fully_qualified_identifier.startswith("org.silastandard/core/"):
                continue  # SiLAService and friends: infrastructure, not a lab capability
            out.append(self._project(feature))
        return out

    def _project(self, feature: Any) -> Feature:
        overrides = self.overrides.get(feature._identifier, {})
        properties = [
            Property(
                name=name, description=prop._description,
                schema=Parameter(name=name, type=_TYPE_MAP.get(_sila_type_name(prop.data_type), "object")),
                writable=False,  # SiLA properties are always server-computed, never client-set
                observable=name in feature._observable_properties,
            )
            for name, prop in feature._unobservable_properties.items()
        ] + [
            Property(
                name=name, description=prop._description,
                schema=Parameter(name=name, type=_TYPE_MAP.get(_sila_type_name(prop.data_type), "object")),
                writable=False, observable=True,
            )
            for name, prop in feature._observable_properties.items()
        ]
        commands = [
            self._build_command(feature, name, cmd, overrides.get("commands", {}).get(name, {}))
            for name, cmd in feature._unobservable_commands.items()
        ]
        namespace, _, _rest = feature.fully_qualified_identifier.rpartition("/" + feature._identifier)
        version = feature.fully_qualified_identifier.rsplit("/v", 1)[-1]
        return Feature(
            identifier=feature._identifier, display_name=feature._display_name,
            description=feature._description, version=version,
            namespace=namespace.replace("/", "."), properties=properties, commands=commands,
        )

    def _build_command(self, feature: Any, name: str, cmd: Any, override: dict[str, Any]) -> Command:
        # `cmd` here is the framework's own Command node (from
        # `feature._unobservable_commands`), which already carries the parsed
        # FDL parameter/response lists directly -- distinct from
        # `feature._client_commands[name]`, the callable wrapper `_invoke`
        # uses via `getattr(sila_feature, name)`.
        parameters = [_parameter_from(p) for p in cmd.parameters.fields]
        returns = [_parameter_from(r) for r in cmd.responses.fields]
        return Command(
            name=name, description=cmd._description,
            parameters=parameters, returns=returns,
            duration_estimate_s=float(override.get("duration_estimate_s", 1.0)),
            hazard=_HAZARDS.get(override.get("hazard", "benign"), Hazard.BENIGN),
            reversibility=_REVERSIBILITY.get(
                override.get("reversibility", "restorable"), Reversibility.RESTORABLE
            ),
            # Observable SiLA commands (progress, intermediate responses) are
            # out of scope for this driver's first cut: LabBench's own
            # job/observable model already gives an agent progress and
            # cancellation, and bridging two different long-running-operation
            # designs onto each other is worth doing deliberately, not as a
            # side effect of a generic command loop.
            simulatable=True, tags=set(override.get("tags", [])),
        )

    def _find_feature(self, identifier: str) -> Any:
        for feature in self.client._features.values():
            if feature._identifier == identifier:
                return feature
        raise ValidationError(f"no SiLA feature {identifier!r} on this server", feature=identifier)

    # -- data plane -------------------------------------------------------

    async def _read(self, feature: str, name: str) -> Any:
        sila_feature = self._find_feature(feature)
        prop = getattr(sila_feature, name, None)
        if prop is None:
            raise ValidationError(f"{feature} has no property {name!r}")
        try:
            return await asyncio.to_thread(prop.get)
        except Exception as exc:  # noqa: BLE001 - a SiLA server can raise almost anything here
            raise DeviceFault(f"{feature}.{name}: {exc}", device=self.id) from None

    async def _invoke(
        self, feature: str, command: str, args: dict[str, Any], ctx: ExecutionContext
    ) -> Any:
        sila_feature = self._find_feature(feature)
        cmd = getattr(sila_feature, command, None)
        if cmd is None:
            raise ValidationError(f"{feature} has no command {command!r}")
        try:
            response = await asyncio.to_thread(lambda: cmd(**args))
        except Exception as exc:  # noqa: BLE001 - a SiLA server can raise almost anything here
            raise DeviceFault(f"{feature}.{command}: {exc}", device=self.id) from None
        if response is None:
            return {}
        return dict(response._asdict()) if hasattr(response, "_asdict") else {"value": response}

    async def _simulate(
        self, feature: str, command: str, args: dict[str, Any]
    ) -> SimulationResult:
        return SimulationResult(
            feasible=True, fidelity="none",
            warnings=[(f"the SiLA server exposes no digital twin for {feature}.{command}; "
                       "the outcome cannot be predicted before it runs")],
        )
