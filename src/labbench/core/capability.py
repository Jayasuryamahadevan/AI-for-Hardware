"""The capability model: LabBench's lingua franca for instruments.

Every lab-automation standard converged on the same three nouns:

  =========  ==================  ==================  =====================
  LabBench   SiLA 2 (FDL)        W3C WoT TD          OPC UA LADS
  =========  ==================  ==================  =====================
  Property   Property            Property            Variable
  Command    Command             Action              Method
  Event      (observable prop.)  Event               Event / Notifier
  =========  ==================  ==================  =====================

So we adopt that triple as the internal model and treat every real protocol as
a *projection*. Northbound it projects to MCP tools; southbound each driver
projects it onto SiLA/LADS/SCPI/MMCore. Adding a protocol means writing one
projection, not touching the agent-facing surface.

What this model adds beyond the three standards is the metadata an *autonomous*
caller needs and a human operator normally supplies from tacit knowledge:
how long a command takes, whether it moves matter, whether it can be undone,
and what has to be true before it may run.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import ConstraintViolation, ValidationError

# --------------------------------------------------------------------------
# Hazard / reversibility metadata
# --------------------------------------------------------------------------


class Hazard(str, Enum):
    """Worst-case physical consequence class of a command.

    Ordered. The safety kernel compares against the session autonomy level, so
    the ordering is load-bearing, not decorative.
    """

    NONE = "none"              # pure query; cannot change physical state
    BENIGN = "benign"          # changes state, trivially reversible (LED, filter)
    MOTION = "motion"          # moves an axis or robot; collision risk
    SAMPLE = "sample"          # consumes/alters sample or reagent; irreversible
    THERMAL = "thermal"        # heating/cooling; runaway potential
    CHEMICAL = "chemical"      # reagent mixing, pressure, reactive chemistry
    BIOLOGICAL = "biological"  # live culture, containment relevant
    RADIOLOGICAL = "radiological"  # X-ray/laser shutter, ionising or class-3B+

    @property
    def rank(self) -> int:
        return _HAZARD_RANK[self]


_HAZARD_RANK = {
    Hazard.NONE: 0,
    Hazard.BENIGN: 1,
    Hazard.MOTION: 2,
    Hazard.SAMPLE: 3,
    Hazard.THERMAL: 4,
    Hazard.CHEMICAL: 5,
    Hazard.BIOLOGICAL: 6,
    Hazard.RADIOLOGICAL: 7,
}


class Reversibility(str, Enum):
    REVERSIBLE = "reversible"      # an exact inverse command exists
    RESTORABLE = "restorable"      # prior state can be re-reached, not undone
    IRREVERSIBLE = "irreversible"  # consumes matter or time; no way back


# --------------------------------------------------------------------------
# Typed parameters with physical units
# --------------------------------------------------------------------------

_SCALAR = Literal["number", "integer", "string", "boolean", "array", "object"]


class Constraint(BaseModel):
    """Value constraints, kept separate from the JSON Schema so that drivers can
    tighten them at runtime from the real device (a stage's true travel limits
    are not knowable until it is homed)."""

    model_config = ConfigDict(extra="forbid")

    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: float | None = None
    exclusive_maximum: float | None = None
    multiple_of: float | None = None
    enum: list[Any] | None = None
    pattern: str | None = None
    min_length: int | None = None
    max_length: int | None = None
    min_items: int | None = None
    max_items: int | None = None

    def check(self, value: Any, *, path: str) -> None:
        """Raise ConstraintViolation if `value` is out of envelope."""
        if self.enum is not None and value not in self.enum:
            raise ConstraintViolation(
                f"{path}: {value!r} is not one of {self.enum!r}",
                path=path, value=value, allowed=self.enum,
            )
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            for bound, op, sym in (
                (self.minimum, lambda v, b: v < b, ">="),
                (self.maximum, lambda v, b: v > b, "<="),
                (self.exclusive_minimum, lambda v, b: v <= b, ">"),
                (self.exclusive_maximum, lambda v, b: v >= b, "<"),
            ):
                if bound is not None and op(value, bound):
                    raise ConstraintViolation(
                        f"{path}: {value} violates {sym} {bound}",
                        path=path, value=value, bound=bound, operator=sym,
                    )
            if self.multiple_of is not None and self.multiple_of > 0:
                q = value / self.multiple_of
                if abs(q - round(q)) > 1e-9:
                    raise ConstraintViolation(
                        f"{path}: {value} is not a multiple of {self.multiple_of}",
                        path=path, value=value, multiple_of=self.multiple_of,
                    )
        if isinstance(value, str):
            if self.pattern is not None and not re.fullmatch(self.pattern, value):
                raise ConstraintViolation(
                    f"{path}: {value!r} does not match /{self.pattern}/",
                    path=path, value=value, pattern=self.pattern,
                )
            if self.min_length is not None and len(value) < self.min_length:
                raise ConstraintViolation(f"{path}: shorter than {self.min_length}", path=path)
            if self.max_length is not None and len(value) > self.max_length:
                raise ConstraintViolation(f"{path}: longer than {self.max_length}", path=path)
        if isinstance(value, (list, tuple)):
            if self.min_items is not None and len(value) < self.min_items:
                raise ConstraintViolation(f"{path}: fewer than {self.min_items} items", path=path)
            if self.max_items is not None and len(value) > self.max_items:
                raise ConstraintViolation(f"{path}: more than {self.max_items} items", path=path)

    def to_json_schema(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        mapping = {
            "minimum": self.minimum, "maximum": self.maximum,
            "exclusiveMinimum": self.exclusive_minimum,
            "exclusiveMaximum": self.exclusive_maximum,
            "multipleOf": self.multiple_of, "enum": self.enum,
            "pattern": self.pattern, "minLength": self.min_length,
            "maxLength": self.max_length, "minItems": self.min_items,
            "maxItems": self.max_items,
        }
        for k, v in mapping.items():
            if v is not None:
                out[k] = v
        return out


class Parameter(BaseModel):
    """One command argument or property value.

    `unit` is mandatory for physical quantities and is *not* cosmetic: unit
    mismatch is the single most common class of automation error, and an LLM
    reading `"unit": "um"` is far less likely to send millimetres than one
    reading a bare `float`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    type: _SCALAR = "number"
    description: str = ""
    unit: str | None = None
    default: Any = None
    required: bool = True
    constraint: Constraint = Field(default_factory=Constraint)
    #: For type="array": the element definition.
    items: "Parameter | None" = None
    #: For type="object": named fields.
    properties: list["Parameter"] = Field(default_factory=list)

    def to_json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self.type}
        desc = self.description
        if self.unit:
            desc = f"{desc} [unit: {self.unit}]".strip()
        if desc:
            schema["description"] = desc
        schema.update(self.constraint.to_json_schema())
        if self.type == "array" and self.items is not None:
            schema["items"] = self.items.to_json_schema()
        if self.type == "object" and self.properties:
            schema["properties"] = {p.name: p.to_json_schema() for p in self.properties}
            req = [p.name for p in self.properties if p.required]
            if req:
                schema["required"] = req
        if self.default is not None:
            schema["default"] = self.default
        return schema

    def validate_value(self, value: Any, *, path: str | None = None) -> Any:
        """Type-coerce then constraint-check. Returns the coerced value."""
        path = path or self.name
        expected = {
            "number": (int, float), "integer": (int,), "string": (str,),
            "boolean": (bool,), "array": (list, tuple), "object": (dict,),
        }[self.type]
        # bool is a subclass of int in Python; keep them distinct for instruments.
        if self.type in ("number", "integer") and isinstance(value, bool):
            raise ValidationError(f"{path}: expected {self.type}, got boolean", path=path)
        if not isinstance(value, expected):
            if self.type == "number" and isinstance(value, int):
                value = float(value)
            elif self.type == "integer" and isinstance(value, float) and value.is_integer():
                value = int(value)
            else:
                raise ValidationError(
                    f"{path}: expected {self.type}, got {type(value).__name__}",
                    path=path, expected=self.type, got=type(value).__name__,
                )
        self.constraint.check(value, path=path)
        if self.type == "array" and self.items is not None:
            value = [self.items.validate_value(v, path=f"{path}[{i}]")
                     for i, v in enumerate(value)]
        if self.type == "object" and self.properties:
            out: dict[str, Any] = {}
            for p in self.properties:
                if p.name in value:
                    out[p.name] = p.validate_value(value[p.name], path=f"{path}.{p.name}")
                elif p.required and p.default is None:
                    raise ValidationError(f"{path}.{p.name}: required", path=f"{path}.{p.name}")
                elif p.default is not None:
                    out[p.name] = p.default
            value = out
        return value


Parameter.model_rebuild()


# --------------------------------------------------------------------------
# Preconditions
# --------------------------------------------------------------------------


class Precondition(BaseModel):
    """A machine-checkable statement that must hold before a command runs.

    Expressed against device properties so the safety kernel can evaluate it
    without knowing anything about the specific instrument. This is what turns
    "you must home the stage first" from tribal knowledge into an enforced gate.
    """

    model_config = ConfigDict(extra="forbid")

    property: str
    operator: Literal["==", "!=", "<", "<=", ">", ">=", "in", "not_in", "is_true", "is_false"]
    value: Any = None
    message: str = ""

    def evaluate(self, properties: dict[str, Any]) -> tuple[bool, str]:
        if self.property not in properties:
            return False, f"property {self.property!r} is unknown"
        actual = properties[self.property]
        ok = {
            "==": lambda: actual == self.value,
            "!=": lambda: actual != self.value,
            "<": lambda: actual < self.value,
            "<=": lambda: actual <= self.value,
            ">": lambda: actual > self.value,
            ">=": lambda: actual >= self.value,
            "in": lambda: actual in (self.value or []),
            "not_in": lambda: actual not in (self.value or []),
            "is_true": lambda: bool(actual),
            "is_false": lambda: not bool(actual),
        }[self.operator]()
        if ok:
            return True, ""
        reason = self.message or (
            f"requires {self.property} {self.operator} {self.value!r}, but it is {actual!r}"
        )
        return False, reason


# --------------------------------------------------------------------------
# The three capability kinds
# --------------------------------------------------------------------------


class Property(BaseModel):
    """Readable (and optionally writable/observable) instrument state."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    schema_: Parameter = Field(alias="schema")
    writable: bool = False
    observable: bool = False
    #: Hazard incurred by *writing*. Reads are always Hazard.NONE.
    write_hazard: Hazard = Hazard.BENIGN
    #: Suggested minimum polling interval when observing, in seconds.
    poll_interval_s: float = 1.0

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Command(BaseModel):
    """An invocable operation.

    `duration_estimate_s` and `observable` together decide whether the MCP layer
    answers inline or hands back a job handle. Anything that can outlive a
    request timeout must be observable, or agents will simply lose track of it.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    parameters: list[Parameter] = Field(default_factory=list)
    returns: list[Parameter] = Field(default_factory=list)
    #: Long-running: executes as a job with progress, not a blocking call.
    observable: bool = False
    duration_estimate_s: float = 0.1
    hazard: Hazard = Hazard.BENIGN
    reversibility: Reversibility = Reversibility.RESTORABLE
    #: Name of the command that undoes this one, when one exists.
    inverse: str | None = None
    preconditions: list[Precondition] = Field(default_factory=list)
    #: Exclusive use of the device for the duration (blocks concurrent commands).
    exclusive: bool = True
    #: True when a faithful simulation exists, enabling dry-run verification.
    simulatable: bool = True
    #: Free-form tags used by policy rules, e.g. {"consumes_reagent", "moves_z"}.
    tags: set[str] = Field(default_factory=set)

    @field_validator("parameters", "returns")
    @classmethod
    def _unique_names(cls, v: list[Parameter]) -> list[Parameter]:
        names = [p.name for p in v]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate parameter names: {names}")
        return v

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {p.name: p.to_json_schema() for p in self.parameters},
            "required": [p.name for p in self.parameters if p.required and p.default is None],
            "additionalProperties": False,
        }

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        by_name = {p.name: p for p in self.parameters}
        unknown = set(args) - set(by_name)
        if unknown:
            raise ValidationError(
                f"unknown parameter(s): {sorted(unknown)}; expected {sorted(by_name)}",
                unknown=sorted(unknown), expected=sorted(by_name),
            )
        out: dict[str, Any] = {}
        for name, p in by_name.items():
            if name in args and args[name] is not None:
                out[name] = p.validate_value(args[name])
            elif p.default is not None:
                out[name] = p.default
            elif p.required:
                raise ValidationError(
                    f"missing required parameter {name!r}"
                    + (f" [unit: {p.unit}]" if p.unit else ""),
                    parameter=name,
                )
        return out


class Event(BaseModel):
    """An asynchronous notification the device may emit."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    payload: list[Parameter] = Field(default_factory=list)
    #: Events at or above this severity are mirrored into the audit log.
    severity: Literal["debug", "info", "warning", "error", "critical"] = "info"


class Feature(BaseModel):
    """A cohesive, versioned group of capabilities — SiLA 2's key idea.

    Grouping by *function* rather than by device model is what makes the system
    substitutable: an agent that can drive `MotionControl/move_absolute` drives
    every stage that implements it, regardless of vendor.
    """

    model_config = ConfigDict(extra="forbid")

    identifier: str
    display_name: str = ""
    description: str = ""
    version: str = "1.0"
    #: Reverse-DNS namespace, mirroring SiLA 2 originators.
    namespace: str = "org.labbench.core"
    properties: list[Property] = Field(default_factory=list)
    commands: list[Command] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)

    @property
    def fqid(self) -> str:
        return f"{self.namespace}/{self.identifier}/v{self.version}"

    def command(self, name: str) -> Command | None:
        return next((c for c in self.commands if c.name == name), None)

    def property(self, name: str) -> Property | None:  # noqa: A003
        return next((p for p in self.properties if p.name == name), None)
