"""The planner: the from-scratch GP, its acquisition function, and the
planner's fallback-to-random behaviour when there is nothing yet to fit."""

from __future__ import annotations

import numpy as np
import pytest

from labbench.campaign.objective import Objective, Observation
from labbench.campaign.planner import (
    BayesianPlanner,
    GaussianProcess,
    expected_improvement,
)
from labbench.campaign.space import Dimension, ParameterSpace


class TestExpectedImprovement:
    def test_zero_everywhere_certainty_meets_the_incumbent(self):
        mu = np.array([5.0])
        sigma = np.array([1e-12])
        ei = expected_improvement(mu, sigma, best=5.0, xi=0.0)
        assert ei[0] == pytest.approx(0.0, abs=1e-6)

    def test_higher_mean_scores_higher_at_equal_uncertainty(self):
        mu = np.array([1.0, 5.0])
        sigma = np.array([1.0, 1.0])
        ei = expected_improvement(mu, sigma, best=0.0)
        assert ei[1] > ei[0]

    def test_more_uncertainty_scores_higher_at_equal_mean(self):
        mu = np.array([0.0, 0.0])
        sigma = np.array([0.1, 2.0])
        ei = expected_improvement(mu, sigma, best=0.0)
        assert ei[1] > ei[0]


class TestGaussianProcess:
    def test_predicts_training_points_almost_exactly(self):
        rng = np.random.default_rng(0)
        x = rng.random((10, 2))
        y = np.sin(x[:, 0] * 3) + x[:, 1]
        gp = GaussianProcess(length_scale=0.3, noise=1e-6)
        gp.fit(x, y)
        mu, sigma = gp.predict(x)
        assert mu == pytest.approx(y, abs=1e-2)
        assert np.all(sigma < 0.05)

    def test_uncertainty_grows_away_from_training_data(self):
        x = np.array([[0.5, 0.5]])
        y = np.array([1.0])
        gp = GaussianProcess(length_scale=0.2, noise=1e-6)
        gp.fit(x, y)
        _, sigma_near = gp.predict(np.array([[0.51, 0.51]]))
        _, sigma_far = gp.predict(np.array([[0.99, 0.99]]))
        assert sigma_far[0] > sigma_near[0]

    def test_survives_duplicate_points_via_jitter(self):
        x = np.array([[0.5, 0.5], [0.5, 0.5], [0.2, 0.8]])
        y = np.array([1.0, 1.0, 0.0])
        gp = GaussianProcess(length_scale=0.3, noise=1e-8)
        gp.fit(x, y)  # must not raise LinAlgError
        mu, _ = gp.predict(np.array([[0.5, 0.5]]))
        assert mu[0] == pytest.approx(1.0, abs=0.2)


def make_space() -> ParameterSpace:
    return ParameterSpace(dimensions=[Dimension(name="x", low=0.0, high=1.0)])


def make_objective() -> Objective:
    return Objective(name="score", path="steps.a.result.value", direction="maximize")


class TestBayesianPlanner:
    def test_falls_back_to_random_with_fewer_than_two_scored_points(self):
        space, objective = make_space(), make_objective()
        rng = np.random.default_rng(0)
        obs = [Observation(trial=0, point={"x": 0.5}, values={"score": 1.0})]
        point = BayesianPlanner().suggest(space, [objective], obs, rng)
        assert space.contains(point)

    def test_falls_back_to_random_when_every_score_ties(self):
        space, objective = make_space(), make_objective()
        rng = np.random.default_rng(0)
        obs = [
            Observation(trial=0, point={"x": 0.1}, values={"score": 3.0}),
            Observation(trial=1, point={"x": 0.9}, values={"score": 3.0}),
        ]
        point = BayesianPlanner().suggest(space, [objective], obs, rng)
        assert space.contains(point)

    def test_ignores_infeasible_and_unevaluated_trials(self):
        space, objective = make_space(), make_objective()
        rng = np.random.default_rng(0)
        obs = [
            Observation(trial=0, point={"x": 0.5}, values={"score": 1.0}),
            Observation(trial=1, point={"x": 0.6}, evaluated=False),
            Observation(trial=2, point={"x": 0.7}, values={"score": 2.0}, feasible=False),
        ]
        # Only one usable scored point remains -> falls back to random rather
        # than fitting a GP to a single point.
        point = BayesianPlanner().suggest(space, [objective], obs, rng)
        assert space.contains(point)

    def test_converges_towards_the_optimum_of_a_simple_quadratic(self):
        """The realistic check: given a handful of scored points sampled off a
        smooth 1D function, does the proposed point move meaningfully closer
        to the true optimum than an arbitrary guess would? This is the whole
        point of shipping GP-EI instead of pure random search."""
        space, objective = make_space(), make_objective()
        rng = np.random.default_rng(7)

        def f(x: float) -> float:
            return -((x - 0.7) ** 2)  # optimum at x=0.7, value 0.0

        planner = BayesianPlanner(candidates=4000)
        xs = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
        obs = [
            Observation(trial=i, point={"x": x}, values={"score": f(x)})
            for i, x in enumerate(xs)
        ]
        proposal = planner.suggest(space, [objective], obs, rng)
        assert abs(proposal["x"] - 0.7) < 0.25

    def test_proposals_stay_inside_the_space(self):
        space, objective = make_space(), make_objective()
        rng = np.random.default_rng(1)
        obs = [
            Observation(trial=i, point={"x": x}, values={"score": -(x - 0.3) ** 2})
            for i, x in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])
        ]
        for _ in range(5):
            point = BayesianPlanner().suggest(space, [objective], obs, rng)
            assert space.contains(point)
