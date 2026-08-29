"""The search space: a campaign's operating design domain.

A `ParameterSpace` is the optimiser's equivalent of the safety kernel's ODD.
The planner may propose any point inside it and no point outside it, and that
containment is checked *before* a proposal can become a run — because an
optimiser is exactly the kind of caller that discovers, empirically, that the
interesting region is just past the edge of the envelope.

Dimensions deliberately reuse `core.capability.Parameter` and `Constraint` for
their validation rather than re-implementing bounds checking. The same code
that decides whether an agent's hand-written argument is legal decides whether
the optimiser's proposal is, so there is no second envelope to keep in sync.

Encoding. Planners work in the unit cube: numeric dimensions map to [0, 1]
(through a log transform when the dimension is declared logarithmic, because
an exposure search from 0.5 ms to 5000 ms is a search over decades, not over
milliseconds), and a categorical dimension occupies a one-hot block, decoded by
argmax. A Gaussian process over that cube then treats every dimension as
comparably scaled without the caller having to think about units.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.capability import Constraint, Parameter
from ..core.errors import ValidationError

DimensionType = Literal["continuous", "integer", "categorical"]


class Dimension(BaseModel):
    """One axis of the search.

    `name` is not free-form: it must be the name of a variable the campaign's
    protocol declares, because binding a proposal to a run means overriding
    that variable. A dimension whose name matches nothing would search a knob
    that is not connected to anything, which `CampaignSpec.validate_against`
    reports rather than silently tolerating.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    type: DimensionType = "continuous"
    low: float | None = None
    high: float | None = None
    #: For type="categorical".
    choices: list[Any] | None = None
    unit: str | None = None
    #: Search on a log scale. Right whenever the plausible range spans decades.
    log: bool = False
    #: Round proposals to this grid, in the dimension's own units.
    step: float | None = None
    description: str = ""

    @model_validator(mode="after")
    def _coherent(self) -> Dimension:
        if self.type == "categorical":
            if not self.choices or len(self.choices) < 2:
                raise ValueError(f"dimension {self.name!r}: categorical needs >= 2 choices")
            if self.log:
                raise ValueError(f"dimension {self.name!r}: a categorical axis cannot be log")
            return self
        if self.low is None or self.high is None:
            raise ValueError(f"dimension {self.name!r}: numeric axes need both low and high")
        if not self.high > self.low:
            raise ValueError(
                f"dimension {self.name!r}: high ({self.high}) must exceed low ({self.low})"
            )
        if self.log and self.low <= 0:
            raise ValueError(
                f"dimension {self.name!r}: a log axis needs low > 0, got {self.low}"
            )
        return self

    # -- shape ------------------------------------------------------------

    @property
    def width(self) -> int:
        """How many columns this dimension occupies in the encoded vector."""
        return len(self.choices or ()) if self.type == "categorical" else 1

    def to_parameter(self) -> Parameter:
        """The capability-model view of this axis, for validation and schemas."""
        if self.type == "categorical":
            kind = "string" if all(isinstance(c, str) for c in self.choices or ()) else "number"
            return Parameter(
                name=self.name, type=kind, description=self.description,
                constraint=Constraint(enum=list(self.choices or ())),
            )
        return Parameter(
            name=self.name,
            type="integer" if self.type == "integer" else "number",
            description=self.description, unit=self.unit,
            constraint=Constraint(minimum=self.low, maximum=self.high),
        )

    def extremes(self) -> list[Any]:
        """The values worth checking against the device envelope.

        For a numeric axis the two ends and the midpoint; for a categorical
        axis every choice. Checking these is what lets `campaign.validate`
        prove, before a single trial runs, that no proposal the planner is
        *able* to make can leave the declared operating domain — the corners
        are where an optimiser ends up, so the corners are what get checked.
        """
        if self.type == "categorical":
            return list(self.choices or ())
        assert self.low is not None and self.high is not None
        mid = self.quantize(
            math.sqrt(self.low * self.high) if self.log else (self.low + self.high) / 2
        )
        return [self.quantize(self.low), mid, self.quantize(self.high)]

    # -- values -----------------------------------------------------------

    def quantize(self, value: float) -> Any:
        if self.type == "categorical":
            return value
        if self.step:
            assert self.low is not None
            value = self.low + round((value - self.low) / self.step) * self.step
        assert self.low is not None and self.high is not None
        value = min(max(value, self.low), self.high)
        if self.type == "integer":
            return round(value)
        return float(round(value, 10))

    def check(self, value: Any) -> None:
        """Raise if `value` is outside this axis. Reuses `Constraint.check`."""
        self.to_parameter().validate_value(value, path=self.name)

    def encode(self, value: Any) -> list[float]:
        if self.type == "categorical":
            choices = list(self.choices or ())
            if value not in choices:
                raise ValidationError(
                    f"{self.name}: {value!r} is not one of {choices!r}", path=self.name
                )
            return [1.0 if c == value else 0.0 for c in choices]
        assert self.low is not None and self.high is not None
        v = float(value)
        if self.log:
            lo, hi, v = math.log(self.low), math.log(self.high), math.log(max(v, 1e-300))
        else:
            lo, hi = self.low, self.high
        return [min(max((v - lo) / (hi - lo), 0.0), 1.0)]

    def decode(self, block: list[float]) -> Any:
        if self.type == "categorical":
            choices = list(self.choices or ())
            return choices[int(np.argmax(block))]
        assert self.low is not None and self.high is not None
        u = min(max(float(block[0]), 0.0), 1.0)
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return self.quantize(math.exp(lo + u * (hi - lo)))
        return self.quantize(self.low + u * (self.high - self.low))


class ParameterSpace(BaseModel):
    """The set of points a planner may propose.

    Ordered, because the encoded vector's column layout must be stable for as
    long as a campaign's surrogate model is being fitted to it.
    """

    model_config = ConfigDict(extra="forbid")

    dimensions: list[Dimension] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> ParameterSpace:
        names = [d.name for d in self.dimensions]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate dimension names: {names}")
        if not names:
            raise ValueError("a search space needs at least one dimension")
        return self

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.dimensions]

    @property
    def width(self) -> int:
        return sum(d.width for d in self.dimensions)

    def dimension(self, name: str) -> Dimension | None:
        return next((d for d in self.dimensions if d.name == name), None)

    # -- containment ------------------------------------------------------

    def validate_point(self, point: dict[str, Any]) -> dict[str, Any]:
        """Type-check, bounds-check and quantise a proposal.

        Raises rather than clipping. A planner that proposes out of bounds has
        a bug, and silently clipping it would hide the bug behind an
        experiment that ran at a different point than the model believes.
        """
        unknown = set(point) - set(self.names)
        if unknown:
            raise ValidationError(
                f"point has unknown dimension(s) {sorted(unknown)}; "
                f"space declares {self.names}",
                unknown=sorted(unknown), expected=self.names,
            )
        missing = set(self.names) - set(point)
        if missing:
            raise ValidationError(
                f"point is missing dimension(s) {sorted(missing)}", missing=sorted(missing)
            )
        out: dict[str, Any] = {}
        for dim in self.dimensions:
            value = point[dim.name]
            dim.check(value)
            out[dim.name] = dim.quantize(value) if dim.type != "categorical" else value
        return out

    def contains(self, point: dict[str, Any]) -> bool:
        try:
            self.validate_point(point)
        except ValidationError:
            return False
        return True

    # -- encoding ---------------------------------------------------------

    def encode(self, point: dict[str, Any]) -> np.ndarray:
        row: list[float] = []
        for dim in self.dimensions:
            row.extend(dim.encode(point[dim.name]))
        return np.asarray(row, dtype=float)

    def encode_many(self, points: list[dict[str, Any]]) -> np.ndarray:
        if not points:
            return np.zeros((0, self.width))
        return np.vstack([self.encode(p) for p in points])

    def decode(self, vector: np.ndarray) -> dict[str, Any]:
        point: dict[str, Any] = {}
        offset = 0
        for dim in self.dimensions:
            block = [float(v) for v in vector[offset : offset + dim.width]]
            point[dim.name] = dim.decode(block)
            offset += dim.width
        return point

    # -- designs ----------------------------------------------------------

    def random(self, rng: np.random.Generator, n: int = 1) -> list[dict[str, Any]]:
        return [self.decode(rng.random(self.width)) for _ in range(n)]

    def latin_hypercube(self, rng: np.random.Generator, n: int) -> list[dict[str, Any]]:
        """A space-filling initial design.

        Better than n independent uniform draws for the same n: each axis is
        stratified into n bins with one sample per bin, so a small initial
        budget — and an initial budget is always small, because every point
        costs a real experiment — still covers every axis evenly.
        """
        n = max(1, n)
        cube = np.empty((n, self.width))
        for col in range(self.width):
            bins = (rng.permutation(n) + rng.random(n)) / n
            cube[:, col] = bins
        return [self.decode(cube[i]) for i in range(n)]

    def grid(self, per_dim: int = 3) -> list[dict[str, Any]]:
        """A full factorial design. Grows as `per_dim ** len(dimensions)`; the
        caller is expected to know that and to keep the space small."""
        axes: list[list[Any]] = []
        for dim in self.dimensions:
            if dim.type == "categorical":
                axes.append(list(dim.choices or ()))
            else:
                k = max(2, per_dim)
                axes.append([dim.decode([i / (k - 1)]) for i in range(k)])
        points: list[dict[str, Any]] = [{}]
        for dim, values in zip(self.dimensions, axes, strict=True):
            points = [{**p, dim.name: v} for p in points for v in values]
        return points

    def to_json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {d.name: d.to_parameter().to_json_schema() for d in self.dimensions},
            "required": self.names,
            "additionalProperties": False,
        }
