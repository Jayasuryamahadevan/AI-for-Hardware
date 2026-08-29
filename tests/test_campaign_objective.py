"""Objectives: extraction from a run's results, constraint satisfaction,
scalarisation, and the Pareto front."""

from __future__ import annotations

import pytest

from labbench.campaign.objective import Objective, Observation, pareto_front, scalarize
from labbench.core.errors import ValidationError


def make_objective(**overrides):
    defaults = {"name": "signal", "path": "steps.snap.result.value", "direction": "maximize"}
    defaults.update(overrides)
    return Objective(**defaults)


class TestObjectiveValidation:
    def test_constraint_needs_a_bound(self):
        with pytest.raises(ValueError, match="needs a minimum, a maximum, or both"):
            Objective(name="x", path="steps.a.result.x", direction="constrain")

    def test_optimised_objective_needs_positive_weight(self):
        with pytest.raises(ValueError, match="weight must be positive"):
            Objective(name="x", path="steps.a.result.x", direction="maximize", weight=0.0)

    def test_optimised_property_true_for_maximize_and_minimize(self):
        assert make_objective(direction="maximize").optimised
        assert make_objective(direction="minimize").optimised
        assert not make_objective(direction="constrain", maximum=1.0).optimised

    def test_sign_flips_for_minimize(self):
        assert make_objective(direction="maximize").sign == 1.0
        assert make_objective(direction="minimize").sign == -1.0


class TestExtract:
    def test_extracts_a_scalar(self):
        obj = make_objective()
        assert obj.extract({"steps": {"snap": {"result": {"value": 4.2}}}}) == 4.2

    def test_missing_path_raises_with_the_stuck_point(self):
        obj = make_objective()
        with pytest.raises(ValidationError, match="does not resolve"):
            obj.extract({"steps": {"snap": {"result": {}}}})

    def test_bool_coerces_to_float(self):
        obj = make_objective(path="steps.snap.result.ok")
        assert obj.extract({"steps": {"snap": {"result": {"ok": True}}}}) == 1.0

    def test_non_numeric_leaf_raises(self):
        obj = make_objective(path="steps.snap.result.uri")
        with pytest.raises(ValidationError, match="is not a number"):
            obj.extract({"steps": {"snap": {"result": {"uri": "file:///x.png"}}}})

    @pytest.mark.parametrize("aggregate,expected", [
        ("last", 3.0), ("first", 1.0), ("mean", 2.0), ("min", 1.0), ("max", 3.0),
        ("sum", 6.0), ("count", 3.0),
    ])
    def test_list_aggregation(self, aggregate, expected):
        obj = make_objective(aggregate=aggregate)
        scope = {"steps": {"snap": {"result": {"value": [1.0, 2.0, 3.0]}}}}
        assert obj.extract(scope) == expected

    def test_empty_list_raises(self):
        obj = make_objective(aggregate="mean")
        with pytest.raises(ValidationError, match="empty list"):
            obj.extract({"steps": {"snap": {"result": {"value": []}}}})

    def test_index_into_list_by_position(self):
        obj = make_objective(path="steps.snap.result.value.1")
        scope = {"steps": {"snap": {"result": {"value": [10.0, 20.0]}}}}
        assert obj.extract(scope) == 20.0


class TestSatisfiedAndReached:
    def test_optimised_objective_always_satisfied(self):
        obj = make_objective()
        ok, reason = obj.satisfied(-1e9)
        assert ok and reason == ""

    def test_constraint_minimum(self):
        obj = make_objective(direction="constrain", minimum=1.0)
        assert obj.satisfied(2.0) == (True, "")
        ok, reason = obj.satisfied(0.5)
        assert not ok and "below its minimum" in reason

    def test_constraint_maximum(self):
        obj = make_objective(direction="constrain", maximum=1.0)
        ok, reason = obj.satisfied(2.0)
        assert not ok and "above its maximum" in reason

    def test_reached_respects_direction(self):
        maximize = make_objective(direction="maximize", target=10.0)
        assert maximize.reached(10.0) and maximize.reached(11.0)
        assert not maximize.reached(9.9)
        minimize = make_objective(direction="minimize", target=1.0)
        assert minimize.reached(1.0) and minimize.reached(0.5)
        assert not minimize.reached(1.1)

    def test_reached_is_false_without_a_target(self):
        assert not make_objective().reached(1e9)


def observation(trial, **values) -> Observation:
    return Observation(trial=trial, point={"x": trial}, values=values)


class TestScalarize:
    def test_single_objective_passes_through_with_sign(self):
        maximize = [make_objective(direction="maximize")]
        obs = [observation(0, signal=1.0), observation(1, signal=5.0)]
        assert scalarize(maximize, obs) == {0: 1.0, 1: 5.0}

        minimize = [make_objective(direction="minimize")]
        assert scalarize(minimize, obs) == {0: -1.0, 1: -5.0}

    def test_no_optimised_objectives_returns_empty(self):
        constraint_only = [make_objective(direction="constrain", minimum=0.0)]
        assert scalarize(constraint_only, [observation(0, signal=1.0)]) == {}

    def test_infeasible_trials_are_penalised_below_every_feasible_one(self):
        objs = [make_objective(direction="maximize")]
        obs = [
            Observation(trial=0, point={}, values={"signal": 10.0}, feasible=True),
            Observation(trial=1, point={}, values={"signal": 999.0}, feasible=False),
        ]
        scalars = scalarize(objs, obs)
        assert scalars[1] < scalars[0]

    def test_unevaluated_trials_are_excluded(self):
        objs = [make_objective(direction="maximize")]
        obs = [
            observation(0, signal=1.0),
            Observation(trial=1, point={}, evaluated=False),
        ]
        assert 1 not in scalarize(objs, obs)

    def test_multi_objective_weighted_normalisation(self):
        a = make_objective(name="a", path="steps.x.result.a", direction="maximize", weight=1.0)
        b = make_objective(name="b", path="steps.x.result.b", direction="minimize", weight=1.0)
        obs = [
            Observation(trial=0, point={}, values={"a": 0.0, "b": 10.0}),
            Observation(trial=1, point={}, values={"a": 10.0, "b": 0.0}),
        ]
        scalars = scalarize([a, b], obs)
        # trial 1 maximises a and minimises b -- unambiguously the better point.
        assert scalars[1] > scalars[0]

    def test_constant_objective_does_not_divide_by_zero(self):
        objs = [make_objective(direction="maximize")]
        obs = [observation(0, signal=3.0), observation(1, signal=3.0)]
        scalars = scalarize(objs, obs)
        assert scalars[0] == scalars[1] == 3.0


class TestParetoFront:
    def test_single_objective_front_is_the_best_trial(self):
        objs = [make_objective(direction="maximize")]
        obs = [observation(0, signal=1.0), observation(1, signal=5.0), observation(2, signal=2.0)]
        assert pareto_front(objs, obs) == [1]

    def test_two_objective_tradeoff_keeps_both_extremes(self):
        a = make_objective(name="a", path="steps.x.result.a", direction="maximize")
        b = make_objective(name="b", path="steps.x.result.b", direction="maximize")
        obs = [
            Observation(trial=0, point={}, values={"a": 1.0, "b": 0.0}),
            Observation(trial=1, point={}, values={"a": 0.0, "b": 1.0}),
            # Worse than trial 0 on both objectives -- dominated, on nobody's frontier.
            Observation(trial=2, point={}, values={"a": 0.5, "b": -1.0}),
        ]
        assert pareto_front([a, b], obs) == [0, 1]

    def test_infeasible_trials_never_appear(self):
        objs = [make_objective(direction="maximize")]
        obs = [Observation(trial=0, point={}, values={"signal": 100.0}, feasible=False)]
        assert pareto_front(objs, obs) == []

    def test_no_optimised_objectives_returns_empty(self):
        constraint_only = [make_objective(direction="constrain", minimum=0.0)]
        assert pareto_front(constraint_only, [observation(0, signal=1.0)]) == []
