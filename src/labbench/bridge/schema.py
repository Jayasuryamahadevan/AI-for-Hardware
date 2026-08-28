"""Tool-schema emitters, one per AI dialect.

A `ToolSpec` is the neutral form: a name, a description, a JSON Schema for the
arguments, and the safety metadata LabBench knows and the vendors mostly do
not. Each emitter projects that into one vendor's wire shape.

The projections are not cosmetic. They differ in ways that silently break tool
calling if you get them wrong:

* **Anthropic** puts the schema at `input_schema`, top level.
* **OpenAI chat completions** nests it at `function.parameters`; strict mode
  additionally requires `additionalProperties: false` *and* every property
  listed in `required`, so optional arguments must be expressed as a nullable
  union rather than by omission.
* **OpenAI responses** flattens that same object by one level.
* **Gemini** accepts an OpenAPI 3.0 subset and rejects requests carrying
  keywords outside it, so unsupported validation keywords must be dropped
  rather than passed through and hoped for.

Where a dialect cannot express a constraint, the constraint is folded into the
description instead of being discarded. A model that cannot see `maximum: 200`
in the schema can still read "must be 0-200" in the text, and for an instrument
that difference is a crashed objective.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Dialect(str, Enum):
    """Tool-schema formats this gateway can speak."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"                 # chat completions `tools`
    OPENAI_RESPONSES = "openai-responses"
    GEMINI = "gemini"
    JSON_SCHEMA = "jsonschema"        # raw, for a hand-rolled loop
    OPENAPI = "openapi"               # a whole OpenAPI 3.1 document


DIALECTS = tuple(d.value for d in Dialect)

#: Every major vendor converged on this constraint for tool names.
_NAME_OK = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_NAME_CLEAN = re.compile(r"[^a-zA-Z0-9_-]")


def sanitise_name(name: str) -> str:
    """Make a tool name acceptable to every dialect.

    Device ids and feature names contain dots and slashes in the natural
    LabBench spelling; those are legal internally and rejected on the wire.
    """
    cleaned = _NAME_CLEAN.sub("_", name).strip("_") or "tool"
    return cleaned[:64]


class ToolSpec(BaseModel):
    """One tool, before it is projected into any vendor's shape."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    #: JSON Schema for the arguments. Always an object schema.
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    #: Schema of the result, when the tool has a predictable shape. Advisory:
    #: no dialect enforces it, but it materially improves an agent's planning.
    returns: dict[str, Any] | None = None
    #: Pure query; changes no physical state. Maps to readOnlyHint.
    read_only: bool = False
    #: Consumes matter or time with no way back. Maps to destructiveHint.
    destructive: bool = False
    #: Same arguments, same effect. False for anything that moves an axis.
    idempotent: bool = False
    #: LabBench hazard class, passed through for clients that understand it.
    hazard: str | None = None
    #: Free-form routing/policy tags.
    tags: list[str] = Field(default_factory=list)

    def wire_name(self) -> str:
        return sanitise_name(self.name)

    def full_description(self) -> str:
        """Description plus the safety facts a vendor schema cannot carry.

        This is the honest fallback. Hazard class and irreversibility are the
        two things an agent most needs to know before calling an instrument,
        and no dialect has a field for them, so they go where the model will
        definitely read them.
        """
        parts = [self.description.strip()] if self.description.strip() else []
        notes = []
        if self.hazard and self.hazard != "none":
            notes.append(f"hazard: {self.hazard}")
        if self.destructive:
            notes.append("IRREVERSIBLE - consumes sample, reagent or time; there is no undo")
        if self.read_only:
            notes.append("read-only; cannot change physical state")
        if notes:
            parts.append("[" + "; ".join(notes) + "]")
        return " ".join(parts)


# -- JSON Schema helpers ---------------------------------------------------


def _describe_constraints(schema: dict[str, Any]) -> str:
    """Render validation keywords as prose, for dialects that drop them."""
    bits = []
    if "minimum" in schema and "maximum" in schema:
        bits.append(f"{schema['minimum']} to {schema['maximum']}")
    elif "minimum" in schema:
        bits.append(f"at least {schema['minimum']}")
    elif "maximum" in schema:
        bits.append(f"at most {schema['maximum']}")
    if "exclusiveMinimum" in schema:
        bits.append(f"greater than {schema['exclusiveMinimum']}")
    if "exclusiveMaximum" in schema:
        bits.append(f"less than {schema['exclusiveMaximum']}")
    if "multipleOf" in schema:
        bits.append(f"a multiple of {schema['multipleOf']}")
    if "pattern" in schema:
        bits.append(f"matching /{schema['pattern']}/")
    if "minLength" in schema or "maxLength" in schema:
        lo, hi = schema.get("minLength"), schema.get("maxLength")
        bits.append(f"length {lo if lo is not None else 0}-{hi if hi is not None else 'any'}")
    return "must be " + ", ".join(bits) if bits else ""


#: Keywords Gemini's OpenAPI 3.0 subset accepts. Anything else is dropped.
_GEMINI_KEYWORDS = {
    "type", "format", "description", "nullable", "enum",
    "maxItems", "minItems", "properties", "required", "items",
}


def _to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Reduce a JSON Schema to Gemini's accepted subset.

    Dropped constraints are appended to the description rather than lost --
    losing `maximum: 200` on a stage travel limit is not an acceptable
    degradation, even though the schema cannot carry it.
    """
    out: dict[str, Any] = {}
    prose = _describe_constraints(schema)
    for key, value in schema.items():
        if key not in _GEMINI_KEYWORDS:
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {k: _to_gemini_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            out[key] = _to_gemini_schema(value)
        else:
            out[key] = value
    if prose:
        existing = out.get("description", "")
        out["description"] = f"{existing} ({prose})".strip() if existing else prose
    out.setdefault("type", "object" if "properties" in out else "string")
    return out


def _to_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a schema for OpenAI strict mode.

    Strict mode requires every property to appear in `required`, so an optional
    argument becomes a nullable union instead of an absent key. This is the one
    projection that changes meaning, and it is the only way to get guaranteed
    schema adherence out of that API.
    """
    out = dict(schema)
    properties = out.get("properties")
    if not isinstance(properties, dict):
        return out
    was_required = set(out.get("required") or [])
    rewritten: dict[str, Any] = {}
    for key, prop in properties.items():
        prop = dict(prop) if isinstance(prop, dict) else {"type": "string"}
        if "properties" in prop:
            prop = _to_strict_schema(prop)
        if key not in was_required:
            # Listing an optional argument in `required` without also making it
            # nullable would force the model to invent a value for something it
            # was entitled to omit. On an instrument that is how a default
            # exposure becomes a guessed one.
            prop = _make_nullable(prop)
        rewritten[key] = prop
    out["properties"] = rewritten
    out["required"] = list(rewritten)
    out["additionalProperties"] = False
    return out


def _make_nullable(prop: dict[str, Any]) -> dict[str, Any]:
    """Widen a property's type to admit null, however its type is spelled."""
    out = dict(prop)
    kind = out.get("type")
    if kind is None:
        # A schema with no `type` (anyOf, $ref, bare enum) already admits
        # whatever it admits; forcing a type on it would narrow it instead.
        return out
    if isinstance(kind, list):
        if "null" not in kind:
            out["type"] = [*kind, "null"]
    elif kind != "null":
        out["type"] = [kind, "null"]
    note = "optional - pass null to leave it unset"
    out["description"] = f"{out['description']} ({note})" if out.get("description") else note
    return out


# -- emitters --------------------------------------------------------------


def _anthropic(tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.wire_name(),
        "description": tool.full_description(),
        "input_schema": tool.parameters,
    }


def _openai(tool: ToolSpec, *, strict: bool) -> dict[str, Any]:
    params = _to_strict_schema(tool.parameters) if strict else tool.parameters
    function: dict[str, Any] = {
        "name": tool.wire_name(),
        "description": tool.full_description(),
        "parameters": params,
    }
    if strict:
        function["strict"] = True
    return {"type": "function", "function": function}


def _openai_responses(tool: ToolSpec, *, strict: bool) -> dict[str, Any]:
    params = _to_strict_schema(tool.parameters) if strict else tool.parameters
    out: dict[str, Any] = {
        "type": "function",
        "name": tool.wire_name(),
        "description": tool.full_description(),
        "parameters": params,
    }
    if strict:
        out["strict"] = True
    return out


def _gemini(tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.wire_name(),
        "description": tool.full_description(),
        "parameters": _to_gemini_schema(tool.parameters),
    }


def _json_schema(tool: ToolSpec) -> dict[str, Any]:
    """The neutral form, with LabBench's own metadata kept intact.

    A hand-rolled agent loop gets more here than any vendor dialect carries:
    the hazard class and the irreversibility flag stay as structured fields
    rather than being flattened into prose.
    """
    out: dict[str, Any] = {
        "name": tool.wire_name(),
        "description": tool.description,
        "parameters": tool.parameters,
        "annotations": {
            "readOnly": tool.read_only,
            "destructive": tool.destructive,
            "idempotent": tool.idempotent,
        },
    }
    if tool.returns is not None:
        out["returns"] = tool.returns
    if tool.hazard:
        out["hazard"] = tool.hazard
    if tool.tags:
        out["tags"] = sorted(tool.tags)
    return out


def emit(
    tools: list[ToolSpec],
    dialect: Dialect | str = Dialect.JSON_SCHEMA,
    *,
    strict: bool = False,
    title: str = "LabBench",
    server_url: str = "http://127.0.0.1:8765",
) -> Any:
    """Project a tool list into one dialect.

    `strict` applies only to the OpenAI dialects, where it buys guaranteed
    schema adherence at the cost of rewriting optional arguments as nullable.
    """
    dialect = Dialect(dialect)
    if dialect is Dialect.ANTHROPIC:
        return [_anthropic(t) for t in tools]
    if dialect is Dialect.OPENAI:
        return [_openai(t, strict=strict) for t in tools]
    if dialect is Dialect.OPENAI_RESPONSES:
        return [_openai_responses(t, strict=strict) for t in tools]
    if dialect is Dialect.GEMINI:
        # Gemini takes one declaration list wrapped in a tool object.
        return [{"function_declarations": [_gemini(t) for t in tools]}]
    if dialect is Dialect.OPENAPI:
        return _openapi_document(tools, title=title, server_url=server_url)
    return [_json_schema(t) for t in tools]


def _openapi_document(
    tools: list[ToolSpec], *, title: str, server_url: str
) -> dict[str, Any]:
    """A full OpenAPI 3.1 document, one POST path per tool.

    This is the dialect for everything that is not a named vendor: an agent
    framework that ingests OpenAPI, an internal service, a code generator, or a
    person reading the docs. It is also what makes the gateway self-describing
    to tooling nobody has written yet.
    """
    paths: dict[str, Any] = {}
    for tool in tools:
        operation: dict[str, Any] = {
            "operationId": tool.wire_name(),
            "summary": tool.description.split("\n")[0][:120],
            "description": tool.full_description(),
            "requestBody": {
                "required": bool(tool.parameters.get("required")),
                "content": {"application/json": {"schema": tool.parameters}},
            },
            "responses": {
                "200": {
                    "description": "Success",
                    "content": {
                        "application/json": {
                            "schema": tool.returns or {"type": "object"}
                        }
                    },
                },
                "400": {"description": "Validation or constraint failure"},
                "403": {"description": "Blocked by the safety kernel"},
                "428": {"description": "Human approval required before this may run"},
            },
        }
        if tool.tags:
            operation["tags"] = sorted(tool.tags)
        if tool.read_only:
            # Advisory, but it is what lets a generator mark the safe subset.
            operation["x-read-only"] = True
        if tool.hazard:
            operation["x-hazard"] = tool.hazard
        if tool.destructive:
            operation["x-irreversible"] = True
        paths[f"/tools/{tool.wire_name()}"] = {"post": operation}

    return {
        "openapi": "3.1.0",
        "info": {
            "title": title,
            "version": "0.1.0",
            "description": "Laboratory instrument control. Commands may move "
                           "matter and consume samples; read the hazard "
                           "annotations before calling.",
        },
        "servers": [{"url": server_url}],
        "paths": paths,
    }
