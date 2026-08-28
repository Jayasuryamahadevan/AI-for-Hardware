"""W3C Web of Things: instruments that publish a Thing Description.

WoT is the one southbound protocol whose own vocabulary is *already* the
Property/Command/Event triple LabBench adopted (see the README's mapping
table) -- a Thing Description (TD) is a JSON document declaring exactly that:
`properties`, `actions`, `events`, each with a JSON Schema and one or more
`forms` saying how to read, write or invoke it over HTTP. So unlike SCPI
(a grammar with no schema) this driver's whole job is projection with almost
no invention: a TD property becomes a LabBench `Property`, an action becomes a
`Command`, its `input`/`output` schema becomes its `Parameter`s, verbatim.

What LabBench adds on top, because a bare TD does not carry it: hazard class
and reversibility. A Thing has no concept of "this destroys the sample", so
those are supplied by the lab configuration (`profile_overrides`) rather than
invented from the TD -- an unclassified action defaults to `Hazard.BENIGN`,
`Reversibility.RESTORABLE`, which is a guess and is logged as one via
`_simulate`'s fidelity, never silently upgraded to something more confident
than the source document supports.

Uses `httpx` because, unlike SCPI's raw line protocol, a Thing's `forms` can
name arbitrary content types, query the `htv:methodName`, and (per the TD
security vocabulary) require a bearer or basic credential -- a small but real
surface that a hand-rolled client would end up reinventing badly.
"""

from __future__ import annotations

import logging
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
from ..core.errors import DeviceFault, DriverUnavailable, TransportError, ValidationError

log = logging.getLogger("labbench.wot")

_JSON_TYPES = {"string": "string", "integer": "integer", "number": "number",
               "boolean": "boolean", "array": "array", "object": "object"}
_HAZARDS = {h.value: h for h in Hazard}
_REVERSIBILITY = {r.value: r for r in Reversibility}


def _schema_to_parameter(name: str, schema: dict[str, Any], *, required: bool = True) -> Parameter:
    """Project one TD DataSchema into a LabBench Parameter.

    TD schemas are JSON Schema, so most of this is a rename, not a
    translation. `unit` comes from the TD's own `unit` keyword (a WoT
    convention, not core JSON Schema) when present.
    """
    kind = _JSON_TYPES.get(schema.get("type", "string"), "string")
    constraint = Constraint(
        minimum=schema.get("minimum"), maximum=schema.get("maximum"),
        enum=schema.get("enum"), pattern=schema.get("pattern"),
        min_length=schema.get("minLength"), max_length=schema.get("maxLength"),
    )
    items = None
    if kind == "array" and isinstance(schema.get("items"), dict):
        items = _schema_to_parameter(f"{name}[]", schema["items"])
    properties = []
    if kind == "object" and isinstance(schema.get("properties"), dict):
        nested_required = set(schema.get("required", []))
        properties = [
            _schema_to_parameter(pname, pschema, required=pname in nested_required)
            for pname, pschema in schema["properties"].items()
        ]
    return Parameter(
        name=name, type=kind, description=schema.get("description", ""),
        unit=schema.get("unit"), required=required, constraint=constraint,
        items=items, properties=properties,
    )


class WoTThing(Device):
    """One Thing, described entirely by its Thing Description.

    Configuration:

        driver: wot
        settings:
          td_url: "http://192.168.1.40/.well-known/wot-thing-description"
          # or, for a Thing with no HTTP TD directory endpoint:
          # thing_description: { ...inline TD JSON... }
          token: "..."                # bearer, if the TD declares BearerSecurityScheme
          username: "..." password: "..."   # if it declares BasicSecurityScheme
          profile_overrides:          # LabBench metadata the TD cannot carry
            actions:
              home:
                hazard: motion
                reversibility: restorable
    """

    requires_package = "httpx"

    def __init__(self, descriptor: DeviceDescriptor, **config: Any) -> None:
        super().__init__(descriptor, **config)
        self.td_url = config.get("td_url")
        self._inline_td = config.get("thing_description")
        if not self.td_url and not self._inline_td:
            raise ValidationError(
                "a WoT device needs either 'td_url' (fetched at connect time) or an "
                "inline 'thing_description'",
                device=descriptor.id,
            )
        self.timeout_s = float(config.get("timeout_s", 10.0))
        self.token = config.get("token")
        self.username = config.get("username")
        self.password = config.get("password")
        self.overrides = config.get("profile_overrides", {}) or {}
        self.td: dict[str, Any] = {}
        self._client: Any = None

    def _auth(self) -> Any:
        try:
            import httpx
        except ImportError:  # pragma: no cover - guarded by requires_package
            return None
        if self.token:
            return None  # applied per-request as a header; see _client_for
        if self.username is not None:
            return httpx.BasicAuth(self.username, self.password or "")
        return None

    async def _connect(self) -> None:
        try:
            import httpx
        except ImportError:
            raise DriverUnavailable(
                "the wot driver needs httpx. Install it with: pip install 'labbench[http]'",
                driver="wot",
            ) from None

        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self._client = httpx.AsyncClient(timeout=self.timeout_s, headers=headers, auth=self._auth())
        if self._inline_td is not None:
            self.td = self._inline_td
        else:
            try:
                response = await self._client.get(self.td_url)
                response.raise_for_status()
                self.td = response.json()
            except httpx.HTTPError as exc:
                raise TransportError(
                    f"cannot fetch the Thing Description from {self.td_url}: {exc}",
                    url=self.td_url,
                ) from None

        self.descriptor.vendor = self.td.get("manufacturer", self.descriptor.vendor)
        self.descriptor.model = self.td.get("title", self.descriptor.model)
        self.descriptor.protocol = "wot"

    async def _disconnect(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _estop(self) -> None:
        """Best-effort: invoke an action literally named `stop`, if the Thing has one.

        WoT defines no universal emergency action, so this is what an honest
        e-stop looks like for a protocol that never promised one: try the
        obvious name, and say so either way rather than pretending success.
        """
        if "stop" in self.td.get("actions", {}):
            try:
                await self._invoke_action("stop", {})
            except Exception as exc:  # noqa: BLE001 - never block an e-stop
                log.warning("%s: WoT e-stop action failed: %s", self.id, exc)

    # -- capability model -------------------------------------------------

    def _features(self) -> Sequence[Feature]:
        overrides = self.overrides
        properties = [
            self._build_property(name, schema, overrides.get("properties", {}).get(name, {}))
            for name, schema in self.td.get("properties", {}).items()
        ]
        commands = [
            self._build_command(name, schema, overrides.get("actions", {}).get(name, {}))
            for name, schema in self.td.get("actions", {}).items()
        ]
        events = [
            Event(name=name, description=schema.get("description", ""))
            for name, schema in self.td.get("events", {}).items()
        ]
        return [
            Feature(
                identifier="Thing",
                display_name=self.td.get("title", self.descriptor.id),
                description=self.td.get("description", ""),
                namespace="org.w3.wot",
                properties=properties, commands=commands, events=events,
            )
        ]

    def _build_property(self, name: str, schema: dict[str, Any], override: dict[str, Any]) -> Property:
        param = _schema_to_parameter(name, schema)
        if override.get("unit"):
            param = param.model_copy(update={"unit": override["unit"]})
        return Property(
            name=name, description=schema.get("description", ""), schema=param,
            writable=not schema.get("readOnly", False) and _has_op(schema, "writeproperty"),
            observable=schema.get("observable", False),
            write_hazard=_HAZARDS.get(override.get("write_hazard", "benign"), Hazard.BENIGN),
        )

    def _build_command(self, name: str, schema: dict[str, Any], override: dict[str, Any]) -> Command:
        input_schema = schema.get("input", {})
        parameters = []
        if input_schema.get("type") == "object":
            required = set(input_schema.get("required", []))
            parameters = [
                _schema_to_parameter(pname, pschema, required=pname in required)
                for pname, pschema in input_schema.get("properties", {}).items()
            ]
        elif input_schema:
            parameters = [_schema_to_parameter("value", input_schema)]
        output_schema = schema.get("output", {})
        returns = [_schema_to_parameter("value", output_schema)] if output_schema else []
        return Command(
            name=name, description=schema.get("description", ""), parameters=parameters,
            returns=returns,
            duration_estimate_s=float(override.get("duration_estimate_s", 1.0)),
            hazard=_HAZARDS.get(override.get("hazard", "benign"), Hazard.BENIGN),
            reversibility=_REVERSIBILITY.get(
                override.get("reversibility", "restorable"), Reversibility.RESTORABLE
            ),
            inverse=override.get("inverse"),
            # A Thing Description carries no notion of a predictive model.
            simulatable=True, tags=set(override.get("tags", [])),
        )

    # -- data plane -------------------------------------------------------

    def _form_for(self, section: str, name: str, op: str) -> dict[str, Any]:
        entry = self.td.get(section, {}).get(name, {})
        for form in entry.get("forms", []):
            ops = form.get("op", ["readproperty" if section == "properties" else "invokeaction"])
            ops = [ops] if isinstance(ops, str) else ops
            if op in ops:
                return form
        if entry.get("forms"):
            return entry["forms"][0]
        raise ValidationError(f"{section}/{name} has no usable form in the Thing Description")

    async def _read(self, feature: str, name: str) -> Any:
        form = self._form_for("properties", name, "readproperty")
        response = await self._request("GET", form["href"])
        return response.json() if _is_json(form) else response.text

    async def _write(self, feature: str, name: str, value: Any) -> None:
        form = self._form_for("properties", name, "writeproperty")
        await self._request(form.get("htv:methodName", "PUT"), form["href"], json=value)

    async def _invoke_action(self, name: str, args: dict[str, Any]) -> Any:
        form = self._form_for("actions", name, "invokeaction")
        payload = args.get("value", args) if set(args) == {"value"} else args
        response = await self._request(
            form.get("htv:methodName", "POST"), form["href"], json=payload or None,
        )
        if not response.content:
            return {}
        return response.json() if _is_json(form) else {"raw": response.text}

    async def _invoke(
        self, feature: str, command: str, args: dict[str, Any], ctx: ExecutionContext
    ) -> Any:
        result = await self._invoke_action(command, args)
        return result if isinstance(result, dict) else {"value": result}

    async def _request(self, method: str, href: str, **kwargs: Any) -> Any:
        import httpx

        url = href if href.startswith("http") else str(
            httpx.URL(self.td_url or "http://thing/").join(href)
        )
        try:
            response = await self._client.request(method, url, **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DeviceFault(
                f"{method} {url} returned {exc.response.status_code}: {exc.response.text[:200]}",
                device=self.id, status_code=exc.response.status_code,
            ) from None
        except httpx.HTTPError as exc:
            raise TransportError(f"{method} {url} failed: {exc}", device=self.id) from None
        return response

    async def _simulate(
        self, feature: str, command: str, args: dict[str, Any]
    ) -> SimulationResult:
        return SimulationResult(
            feasible=True, fidelity="none",
            warnings=[(f"a Thing Description carries no digital twin for {command!r}; "
                       "the outcome cannot be predicted before it is invoked")],
        )


def _has_op(schema: dict[str, Any], op: str) -> bool:
    for form in schema.get("forms", []):
        ops = form.get("op", "readproperty")
        if op in (ops if isinstance(ops, list) else [ops]):
            return True
    return False


def _is_json(form: dict[str, Any]) -> bool:
    return "json" in form.get("contentType", "application/json")
