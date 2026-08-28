"""OPC UA for Laboratory Automation Devices (LADS, IEC 63339 / OPC 30500).

LADS servers standardise the top of the address space and nothing more:
`Objects -> DeviceSet -> <Device> -> FunctionalUnitSet -> <FunctionalUnit>`,
where each functional unit exposes OPC UA Variables (LADS "process values" and
parameters) and Methods (LADS "functions"). Everything below that shape is the
device's own information model -- which is the point of OPC UA: the server
*describes itself*, unlike SCPI, where `drivers/scpi.py` has to supply the
capability model from an external profile because the wire protocol carries
none.

So this driver does not ship a profile. It connects, browses, and turns
whatever Variables and Methods it finds under each functional unit into
LabBench Properties and Commands -- a projection of the address space, not a
hand-maintained map of one. What it cannot get from the server this way is
hazard class and reversibility, which OPC UA has no vocabulary for; those
default to `benign`/`restorable` and are corrected the same way `drivers/scpi.py`
and `drivers/http_wot.py` correct their own honest gaps: `profile_overrides`
in the lab configuration.

Depth is bounded (`max_browse_depth`) because an address space can nest
arbitrarily and a runaway browse at connect time would make a large server
unusable rather than merely slow.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from ..core.capability import (
    Command,
    Feature,
    Hazard,
    Parameter,
    Property,
    Reversibility,
)
from ..core.device import Device, DeviceDescriptor, ExecutionContext, SimulationResult
from ..core.errors import DeviceFault, DriverUnavailable, TransportError, ValidationError

log = logging.getLogger("labbench.opcua_lads")

_HAZARDS = {h.value: h for h in Hazard}
_REVERSIBILITY = {r.value: r for r in Reversibility}

#: OPC UA VariantType name -> LabBench Parameter type.
_TYPE_MAP = {
    "Boolean": "boolean",
    "SByte": "integer", "Byte": "integer", "Int16": "integer", "UInt16": "integer",
    "Int32": "integer", "UInt32": "integer", "Int64": "integer", "UInt64": "integer",
    "Float": "number", "Double": "number",
    "String": "string", "DateTime": "string", "Guid": "string", "XmlElement": "string",
}


class LadsDevice(Device):
    """One LADS functional unit set, projected onto the LabBench capability model.

    Configuration:

        driver: opcua_lads
        settings:
          endpoint_url: "opc.tcp://192.168.1.70:4840/lads"
          device_name: null       # optional; defaults to the first DeviceSet child
          username: null
          password: null
          max_browse_depth: 3
          profile_overrides:      # LADS's address space carries no hazard metadata
            functional_units:
              Dosing:
                commands:
                  Start: {hazard: sample, reversibility: irreversible}
    """

    requires_package = "asyncua"

    def __init__(self, descriptor: DeviceDescriptor, **config: Any) -> None:
        super().__init__(descriptor, **config)
        self.endpoint_url = config.get("endpoint_url")
        if not self.endpoint_url:
            raise ValidationError("an opcua_lads device needs 'endpoint_url'", device=descriptor.id)
        self.device_name = config.get("device_name")
        self.username = config.get("username")
        self.password = config.get("password")
        self.timeout_s = float(config.get("timeout_s", 10.0))
        self.max_browse_depth = int(config.get("max_browse_depth", 3))
        self.overrides = config.get("profile_overrides", {}).get("functional_units", {})
        self.client: Any = None
        self.device_node: Any = None
        #: functional unit identifier -> {"node": ..., "variables": {name: node}, "methods": {name: node}}
        self._units: dict[str, dict[str, Any]] = {}

    async def _connect(self) -> None:
        try:
            from asyncua import Client
        except ImportError:
            raise DriverUnavailable(
                "the opcua_lads driver needs asyncua. Install it with: "
                "pip install 'labbench[opcua]'",
                driver="opcua_lads",
            ) from None

        self.client = Client(self.endpoint_url, timeout=self.timeout_s)
        if self.username:
            self.client.set_user(self.username)
            if self.password:
                self.client.set_password(self.password)
        try:
            await self.client.connect()
        except Exception as exc:  # noqa: BLE001 - asyncua raises a plain Exception on failure
            raise TransportError(
                f"cannot connect to {self.endpoint_url}: {exc}", endpoint=self.endpoint_url,
            ) from None

        try:
            self.device_node = await self._find_device()
            await self._read_identification()
            await self._discover_functional_units()
        except Exception:
            await self.client.disconnect()
            raise
        self.descriptor.protocol = "opcua-lads"

    async def _disconnect(self) -> None:
        if self.client is not None:
            await self.client.disconnect()
            self.client = None

    async def _estop(self) -> None:
        """No universal LADS e-stop exists; call one on every functional unit that has it.

        LADS functional units commonly expose a `Stop` or `Abort` method; this
        tries both names on every unit and reports failures rather than
        swallowing them, because a silent no-op standing in for an e-stop is
        the one failure mode this project does not tolerate anywhere else.
        """
        for unit_id, unit in self._units.items():
            for name in ("Stop", "Abort"):
                node = unit["methods"].get(name)
                if node is None:
                    continue
                try:
                    await unit["node"].call_method(node.nodeid)
                except Exception as exc:  # noqa: BLE001 - never block an e-stop
                    log.warning("%s: %s.%s failed during e-stop: %s", self.id, unit_id, name, exc)

    # -- discovery ------------------------------------------------------------

    async def _child_named(self, node: Any, name: str) -> Any | None:
        """Find a child by BrowseName text, independent of namespace index.

        The DI/LADS companion specs fix the *names* `DeviceSet`,
        `Identification` and `FunctionalUnitSet`; the namespace index they
        land in depends on the order a server loaded its node sets in, which
        LabBench has no way to know in advance. Standard, base-namespace
        properties (`InputArguments`, `EngineeringUnits`) are the one
        exception -- ns=0 is fixed by the OPC UA specification itself.
        """
        for desc in await node.get_children_descriptions():
            if desc.BrowseName.Name == name:
                return self.client.get_node(desc.NodeId)
        return None

    async def _find_device(self) -> Any:
        objects = self.client.get_objects_node()
        device_set = await self._child_named(objects, "DeviceSet")
        if device_set is None:
            raise DeviceFault(
                f"{self.endpoint_url}: no DeviceSet under Objects; this does not look "
                "like a LADS or OPC UA DI server",
                device=self.id,
            )
        candidates = await device_set.get_children()
        if not candidates:
            raise DeviceFault(f"{self.endpoint_url}: DeviceSet has no devices", device=self.id)
        if self.device_name is None:
            return candidates[0]
        for node in candidates:
            name = (await node.read_browse_name()).Name
            if name == self.device_name:
                return node
        raise DeviceFault(
            f"no device named {self.device_name!r} under DeviceSet",
            device=self.id, available=[
                (await n.read_browse_name()).Name for n in candidates
            ],
        )

    async def _read_identification(self) -> None:
        ident = await self._child_named(self.device_node, "Identification")
        if ident is None:
            return  # recommended by DI, not guaranteed
        for name, field in (
            ("Manufacturer", "vendor"), ("Model", "model"),
            ("SerialNumber", "serial"), ("DeviceRevision", "firmware"),
        ):
            node = await self._child_named(ident, name)
            if node is None:
                continue
            try:
                setattr(self.descriptor, field, str(await node.read_value()))
            except Exception:  # noqa: BLE001, S112 - an optional identification field
                continue

    async def _discover_functional_units(self) -> None:
        unit_set = await self._child_named(self.device_node, "FunctionalUnitSet")
        if unit_set is None:
            raise DeviceFault(
                f"{self.device_name or 'device'} has no FunctionalUnitSet; "
                "this server does not look like a LADS device",
                device=self.id,
            )
        for unit_node in await unit_set.get_children():
            name = (await unit_node.read_browse_name()).Name
            variables, methods = {}, {}
            await self._walk(unit_node, variables, methods, depth=self.max_browse_depth)
            var_meta = {vname: await self._variable_meta(vnode) for vname, vnode in variables.items()}
            self._units[name] = {
                "node": unit_node, "variables": variables, "methods": methods, "var_meta": var_meta,
            }

    async def _variable_meta(self, node: Any) -> dict[str, Any]:
        """Type, unit and writability, read once at connect time.

        `_features()` is synchronous (every `Device` builds its capability
        model without awaiting anything, so the base class can cache it), so
        this cannot happen lazily when a `Property` object is built; it is
        done here, during the one async pass discovery already makes, instead.
        """
        from asyncua import ua

        kind = "string"
        try:
            variant = await node.read_data_type_as_variant_type()
            kind = _TYPE_MAP.get(variant.name, "string")
        except Exception:  # noqa: BLE001, S110 - fall back to "string" rather than fail discovery
            pass
        writable = False
        try:
            writable = ua.AccessLevel.CurrentWrite in await node.get_access_level()
        except Exception:  # noqa: BLE001, S110 - AccessLevel is unreadable on some servers; default to read-only
            pass
        return {"type": kind, "unit": await self._engineering_unit(node), "writable": writable}

    async def _walk(
        self, node: Any, variables: dict[str, Any], methods: dict[str, Any], *, depth: int
    ) -> None:
        from asyncua import ua

        if depth <= 0:
            return
        for child in await node.get_children():
            node_class = await child.read_node_class()
            if node_class == ua.NodeClass.Variable:
                name = (await child.read_browse_name()).Name
                variables.setdefault(name, child)
            elif node_class == ua.NodeClass.Method:
                name = (await child.read_browse_name()).Name
                methods.setdefault(name, child)
            elif node_class == ua.NodeClass.Object:
                await self._walk(child, variables, methods, depth=depth - 1)

    async def _unit_of(self, name: str) -> dict[str, Any]:
        if name not in self._units:
            raise ValidationError(
                f"no functional unit {name!r}; available: {sorted(self._units)}",
                available=sorted(self._units),
            )
        return self._units[name]

    async def _engineering_unit(self, node: Any) -> str | None:
        try:
            eu_node = await node.get_child(["0:EngineeringUnits"])
            eu = await eu_node.read_value()
            return getattr(getattr(eu, "DisplayName", None), "Text", None)
        except Exception:  # noqa: BLE001 - most variables carry no unit
            return None

    # -- capability model -------------------------------------------------

    def _features(self) -> Sequence[Feature]:
        return [self._project(name, unit) for name, unit in self._units.items()]

    def _project(self, name: str, unit: dict[str, Any]) -> Feature:
        overrides = self.overrides.get(name, {})
        properties = [
            Property(
                name=vname,
                schema=Parameter(name=vname, type=meta["type"], unit=meta["unit"]),
                writable=meta["writable"],
            )
            for vname, meta in unit["var_meta"].items()
        ]
        commands = [
            self._build_command(name, mname, mnode, overrides.get("commands", {}).get(mname, {}))
            for mname, mnode in unit["methods"].items()
        ]
        return Feature(
            identifier=name, display_name=name, namespace="org.opcfoundation.lads",
            description=f"LADS functional unit {name!r} on {self.endpoint_url}",
            properties=properties, commands=commands,
        )

    def _build_command(
        self, unit_name: str, name: str, node: Any, override: dict[str, Any]
    ) -> Command:
        return Command(
            name=name,
            description=override.get("description", f"OPC UA method {unit_name}/{name}."),
            # Arguments are validated against the server's declared InputArguments
            # at call time (see `_invoke`), not pre-declared here: reading them
            # requires an async round trip this synchronous capability builder
            # cannot make, so the schema is intentionally open here and enforced
            # where it is actually checkable.
            parameters=[Parameter(name="args", type="object", required=False,
                                   description="Positional arguments, by name, per the "
                                               "server's InputArguments.")],
            duration_estimate_s=float(override.get("duration_estimate_s", 1.0)),
            hazard=_HAZARDS.get(override.get("hazard", "benign"), Hazard.BENIGN),
            reversibility=_REVERSIBILITY.get(
                override.get("reversibility", "restorable"), Reversibility.RESTORABLE
            ),
            tags=set(override.get("tags", [])),
        )

    # -- data plane -------------------------------------------------------

    async def _read(self, feature: str, name: str) -> Any:
        unit = await self._unit_of(feature)
        node = unit["variables"].get(name)
        if node is None:
            raise ValidationError(f"{feature} has no variable {name!r}")
        return await node.read_value()

    async def _write(self, feature: str, name: str, value: Any) -> None:
        unit = await self._unit_of(feature)
        node = unit["variables"].get(name)
        if node is None:
            raise ValidationError(f"{feature} has no variable {name!r}")
        from asyncua import ua

        access = await node.get_access_level()
        if ua.AccessLevel.CurrentWrite not in access:
            raise ValidationError(f"{feature}.{name} is read-only on the server")
        await node.write_value(value)

    async def _invoke(
        self, feature: str, command: str, args: dict[str, Any], ctx: ExecutionContext
    ) -> Any:
        unit = await self._unit_of(feature)
        node = unit["methods"].get(command)
        if node is None:
            raise ValidationError(f"{feature} has no method {command!r}")
        ordered = await self._order_arguments(node, args.get("args", {}) or {})
        result = await unit["node"].call_method(node.nodeid, *ordered)
        if result is None:
            return {}
        if isinstance(result, (list, tuple)):
            return {"outputs": [_jsonable(v) for v in result]}
        return {"output": _jsonable(result)}

    async def _order_arguments(self, method_node: Any, args: dict[str, Any]) -> list[Any]:
        """Turn named arguments into the positional list OPC UA methods take.

        The server declares its own signature via the standard
        `InputArguments` property; using it (rather than trusting caller
        order) is what lets an agent pass arguments by name the way every
        other LabBench command does.
        """
        try:
            input_args_node = await method_node.get_child(["0:InputArguments"])
            input_args = await input_args_node.read_value()
        except Exception:  # noqa: BLE001 - a method with no arguments has none to read
            return []
        ordered = []
        for arg in input_args:
            if arg.Name not in args:
                raise ValidationError(f"missing required argument {arg.Name!r}")
            ordered.append(args[arg.Name])
        return ordered

    async def _simulate(
        self, feature: str, command: str, args: dict[str, Any]
    ) -> SimulationResult:
        return SimulationResult(
            feasible=True, fidelity="none",
            warnings=[(f"the OPC UA server exposes no digital twin for {feature}.{command}; "
                       "the outcome cannot be predicted before it runs")],
        )


def _jsonable(value: Any) -> Any:
    """Best-effort scalar coercion for an OPC UA method's output.

    Method outputs may be structures asyncua represents as generated classes
    rather than plain dicts; a value this driver cannot cleanly serialise is
    reported as its string form rather than dropped, so the caller still sees
    *something* happened.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)
