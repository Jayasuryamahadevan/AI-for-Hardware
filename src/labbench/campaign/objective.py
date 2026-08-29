"""Objectives: what a campaign is trying to make true, read out of a run.

The gateway strips every `truth_*` key at the boundary (see
`Gateway._strip_ground_truth`), so an objective can only ever be computed from
what an instrument actually *measured*. That is not an inconvenience to work
around — it is the whole reason a simulated campaign proves anything. An
optimiser that could read the true focal plane out of the digital twin would
solve autofocus in one trial and demonstrate nothing about optimising a real
microscope.

Three kinds of objective:

  maximize / minimize   contribute to the scalar the planner improves
  constrain             do not contribute; a trial that violates one is
                        recorded as infeasible and can never be reported best

Constraints are separate from objectives on purpose. "Maximise signal-to-noise
but keep saturation under 2%" is not a weighted sum — no amount of extra SNR
makes a saturated frame acceptable — and folding it into one would let the
optimiser buy its way past a limit that exists because the measurement stops
being valid beyond it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.errors import ValidationError

Direction = Literal["maximize", "minimize", "constrain"]
Aggregate = Literal["last", "first", "mean", "min", "max", "sum", "count"]


class Objective(BaseModel):
    """One measured quantity extracted from a finished run.

    `path` is a dotted path into the run's results, rooted the same way a
    protocol's `${...}` references are: `steps.<label>.result.<key>`. Reusing
    that rooting means an objective is written the same way as the step
    reference an operator has already learned, and points at the same data.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    path: str
    direction: Direction = "maximize"
    #: Relative importance when several objectives are optimised at once.
    weight: float = 1.0
    #: How to reduce a path that resolves to a list.
    aggregate: Aggregate = "last"
    unit: str | None = None
    description: str = ""
    #: For direction="constrain": the admissible band.
    minimum: float | None = None
    maximum: float | None = None
    #: Stop the campaign once this objective reaches this value. Interpreted in
    #: the objective's own direction: >= for maximize, <= for minimize.
    target: float | None = None

    @model_validator(mode="after")
    def _coherent(self) -> Objective:
        if self.direction == "constrain" and self.minimum is None and self.maximum is None:
            raise ValueError(
                f"objective {self.name!r}: a constraint needs a minimum, a maximum, or both"
            )
        if self.direction != "constrain" and self.weight <= 0:
            raise ValueError(f"objective {self.name!r}: weight must be positive")
        return self

    @property
    def optimised(self) -> bool:
        return self.direction in ("maximize", "minimize")

    @property
    def sign(self) -> float:
        """+1 when larger is better; -1 when smaller is. Used to fold both
        directions into a single "higher is better" scalar for the planner."""
        return -1.0 if self.direction == "minimize" else 1.0

    def extract(self, scope: dict[str, Any]) -> float:
        """Resolve `path` against a run's results and reduce it to a number."""
        node: Any = scope
        walked: list[str] = []
        for part in self.path.split("."):
            walked.append(part)
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            else:
                available = sorted(node) if isinstance(node, dict) else type(node).__name__
                raise ValidationError(
                    f"objective {self.name!r}: path {self.path!r} does not resolve; "
                    f"stuck at {'.'.join(walked)!r} (available: {available})",
                    objective=self.name, path=self.path,
                )
        return self._reduce(node)

    def _reduce(self, node: Any) -> float:
        if isinstance(node, bool):
            return float(node)
        if isinstance(node, (int, float)):
            return float(node)
        if isinstance(node, list):
            values = [self._reduce(v) for v in node]
            if self.aggregate == "count":
                return float(len(values))
            if not values:
                raise ValidationError(
                    f"objective {self.name!r}: path {self.path!r} resolved to an empty list",
                    objective=self.name,
                )
            return float({
                "last": lambda: values[-1],
                "first": lambda: values[0],
                "mean": lambda: sum(values) / len(values),
                "min": lambda: min(values),
                "max": lambda: max(values),
                "sum": lambda: sum(values),
            }[self.aggregate]())
        raise ValidationError(
            f"objective {self.name!r}: path {self.path!r} resolved to "
            f"{type(node).__name__}, which is not a number",
            objective=self.name, path=self.path,
        )

    def satisfied(self, value: float) -> tuple[bool, str]:
        """Constraint check. Always True for an optimised objective."""
        if self.direction != "constrain":
            return True, ""
        if self.minimum is not None and value < self.minimum:
            return False, f"{self.name}={value:.6g} is below its minimum {self.minimum:.6g}"
        if self.maximum is not None and value > self.maximum:
            return False, f"{self.name}={value:.6g} is above its maximum {self.maximum:.6g}"
        return True, ""

    def reached(self, value: float) -> bool:
        if self.target is None:
            return False
        return value >= self.target if self.sign > 0 else value <= self.target


class Observation(BaseModel):
    """One evaluated point: what was tried, what came back, and whether it counts."""

    model_config = ConfigDict(extra="forbid")

    trial: int
    point: dict[str, Any]
    values: dict[str, float] = Field(default_factory=dict)
    feasible: bool = True
    violations: list[str] = Field(default_factory=list)
    #: True only for a trial whose run finished and whose objectives resolved.
    evaluated: bool = True
    run_id: str | None = None
    source: str = "campaign"


def scalarize(
    objectives: list[Objective], observations: list[Observation]
) -> dict[int, float]:
    """Fold each observation's objectives into one "higher is better" number.

    A single objective is passed through untouched apart from its sign, so the
    reported scalar is the measured quantity itself and needs no explaining.

    Several objectives are min-max normalised across the observations seen so
    far and then weighted — which means the scalar of an *old* trial changes as
    later trials widen the observed range. That is correct for the surrogate
    model (which is refitted from scratch each round anyway) and would be
    dishonest to hide, so the raw per-objective values are stored immutably on
    the observation and only the derived scalar moves.

    Infeasible observations are placed strictly below every feasible one rather
    than being dropped: the planner should learn that a region is barred, not
    merely be left with a hole in its data where the barrier is.
    """
    optimised = [o for o in objectives if o.optimised]
    usable = [o for o in observations if o.evaluated]
    if not optimised or not usable:
        return {}

    ranges: dict[str, tuple[float, float]] = {}
    for obj in optimised:
        seen = [o.values[obj.name] for o in usable if obj.name in o.values]
        if seen:
            ranges[obj.name] = (min(seen), max(seen))

    total_weight = sum(o.weight for o in optimised) or 1.0
    raw: dict[int, float] = {}
    for obs in usable:
        if len(optimised) == 1:
            obj = optimised[0]
            if obj.name not in obs.values:
                continue
            raw[obs.trial] = obj.sign * obs.values[obj.name]
            continue
        acc, used = 0.0, 0.0
        for obj in optimised:
            if obj.name not in obs.values or obj.name not in ranges:
                continue
            low, high = ranges[obj.name]
            span = high - low
            unit = 0.5 if span <= 0 else (obs.values[obj.name] - low) / span
            acc += obj.weight * (unit if obj.sign > 0 else 1.0 - unit)
            used += obj.weight
        if used:
            raw[obs.trial] = acc / total_weight

    feasible = [v for t, v in raw.items() if _feasible(usable, t)]
    if feasible:
        floor = min(feasible)
        spread = (max(feasible) - floor) or abs(floor) or 1.0
        penalty = floor - spread
        for trial, value in list(raw.items()):
            if not _feasible(usable, trial):
                raw[trial] = min(value, penalty)
    return raw


def _feasible(observations: list[Observation], trial: int) -> bool:
    return next((o.feasible for o in observations if o.trial == trial), True)


def pareto_front(
    objectives: list[Objective], observations: list[Observation]
) -> list[int]:
    """Trial indices that no other feasible trial dominates on every objective.

    Reported alongside the scalar best because a weighted sum picks one point
    off a trade-off curve and then forgets the curve existed. With one
    objective this degenerates to "the best trial", which is the right answer.
    """
    optimised = [o for o in objectives if o.optimised]
    pool = [o for o in observations if o.evaluated and o.feasible
            and all(obj.name in o.values for obj in optimised)]
    if not optimised or not pool:
        return []

    def better_or_equal(a: Observation, b: Observation, obj: Objective) -> bool:
        return obj.sign * a.values[obj.name] >= obj.sign * b.values[obj.name]

    front: list[int] = []
    for candidate in pool:
        dominated = any(
            other is not candidate
            and all(better_or_equal(other, candidate, obj) for obj in optimised)
            and any(
                obj.sign * other.values[obj.name] > obj.sign * candidate.values[obj.name]
                for obj in optimised
            )
            for other in pool
        )
        if not dominated:
            front.append(candidate.trial)
    return sorted(front)
