"""Closed-loop autonomous experimentation: search space, objectives, planner, runner.

A `CampaignSpec` binds a `Protocol` to a `ParameterSpace` over its variables
and a list of `Objective`s; `CampaignManager` runs it as a sequence of
ordinary `experiment` runs -- propose a point, run it through the same
ledger/safety/approval front door any `device.invoke` uses, extract the
declared objectives from what the instrument actually measured, replan.

See `spec.py` for why a campaign is not a second way to reach hardware, and
`planner.py` for why the search itself needs no dependency beyond `numpy`.
"""

from __future__ import annotations

from .objective import Objective, Observation, pareto_front, scalarize
from .planner import BayesianPlanner, GaussianProcess, expected_improvement
from .runner import CampaignManager, CampaignState, CampaignStatus
from .space import Dimension, ParameterSpace
from .spec import CampaignSpec

__all__ = [
    "BayesianPlanner",
    "CampaignManager",
    "CampaignSpec",
    "CampaignState",
    "CampaignStatus",
    "Dimension",
    "GaussianProcess",
    "Objective",
    "Observation",
    "ParameterSpace",
    "expected_improvement",
    "pareto_front",
    "scalarize",
]
