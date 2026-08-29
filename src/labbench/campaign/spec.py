"""`CampaignSpec`: a protocol, a search space over its variables, and what to
optimise -- the one document a closed-loop campaign is defined by.

A campaign does not invent a new way to talk to hardware. Every trial is an
ordinary `Protocol` run, exactly as `experiment.start` would run it, with one
dimension's proposed value bound to each protocol variable the same way an
operator's `var=value` override at the CLI is. This is deliberate: a person
reading a campaign's ledger sees the same `run_start`/`run_step`/`run_end`
records a hand-run protocol produces, just with `campaign_trial_*` bookends
around them, because a self-driving loop is a disciplined, repeated caller of
the same front door -- not a second one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from ..core.errors import ValidationError
from ..experiment.protocol import Protocol, direct_variable_ref
from .objective import Objective
from .space import ParameterSpace

DesignKind = str  # "latin_hypercube" | "random" | "grid"


class CampaignSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "campaign"
    description: str = ""
    protocol: Protocol
    space: ParameterSpace
    objectives: list[Objective]
    #: Maximum number of trials (protocol runs) this campaign may spend.
    budget: int = 20
    #: How the first `initial_design_size` points are chosen, before there is
    #: enough data to fit a surrogate. See `planner.py`'s module docstring for
    #: why the planner never sees these itself.
    initial_design: DesignKind = "latin_hypercube"
    initial_design_size: int = 5
    seed: int | None = None

    @classmethod
    def load(cls, path: str | Path) -> CampaignSpec:
        text = Path(path).expanduser().read_text(encoding="utf-8")
        return cls.model_validate(yaml.safe_load(text) or {})

    @model_validator(mode="after")
    def _coherent(self) -> CampaignSpec:
        if self.budget < 1:
            raise ValueError("a campaign needs a budget of at least one trial")
        names = [o.name for o in self.objectives]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate objective names: {names}")
        if not any(o.optimised for o in self.objectives):
            raise ValueError("a campaign needs at least one maximize/minimize objective")
        if self.initial_design not in ("latin_hypercube", "random", "grid"):
            raise ValueError(
                f"initial_design must be latin_hypercube, random or grid; "
                f"got {self.initial_design!r}"
            )
        return self

    def validate_against(self, gateway: Any) -> list[str]:
        """As much static checking as `Protocol.validate_against` plus what a
        search space adds: every dimension must actually drive something, and
        the extremes of every dimension bound directly to a step argument must
        sit inside that argument's declared envelope.

        The last part is the search-specific promise: an optimiser is exactly
        the kind of caller that discovers the interesting region is just past
        the edge, and this is what proves, before a single trial runs, that no
        point the planner is *able* to propose can leave it.
        """
        problems = self.protocol.validate_against(gateway)

        labels = {step.resolved_label(i) for i, step in enumerate(self.protocol.steps)}
        for objective in self.objectives:
            head = objective.path.split(".", 2)
            if len(head) < 2 or head[0] != "steps" or head[1] not in labels:
                problems.append(
                    f"objective {objective.name!r}: path {objective.path!r} does not "
                    f"reference a step result (expected 'steps.<label>.result....')"
                )

        bound_dimensions: set[str] = set()
        for index, step in enumerate(self.protocol.steps):
            try:
                device = gateway.device(step.device)
                _, command = device.resolve(step.feature, step.command)
            except Exception:  # noqa: BLE001, S112 - already reported by protocol.validate_against
                continue
            by_name = {p.name: p for p in command.parameters}
            for arg_name, arg_value in step.args.items():
                var_name = direct_variable_ref(arg_value)
                dimension = self.space.dimension(var_name) if var_name else None
                if dimension is None or arg_name not in by_name:
                    continue
                bound_dimensions.add(dimension.name)
                for extreme in dimension.extremes():
                    try:
                        by_name[arg_name].validate_value(extreme)
                    except ValidationError as exc:
                        problems.append(
                            f"step {index + 1} ({step.resolved_label(index)!r}): dimension "
                            f"{dimension.name!r} can propose {extreme!r} for {arg_name!r}, "
                            f"which is out of envelope: {exc.message}"
                        )

        unbound = set(self.space.names) - bound_dimensions
        if unbound:
            problems.append(
                f"dimension(s) {sorted(unbound)} are not bound to any step argument as "
                f"'${{name}}'; the planner would be searching a knob connected to nothing"
            )
        return problems

