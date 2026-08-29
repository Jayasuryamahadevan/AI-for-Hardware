"""The planner: proposes the next point to try, once the initial design is spent.

A from-scratch Gaussian process surrogate over the unit cube, fitted to the
scalarised score `objective.scalarize` produces, with Expected Improvement as
the acquisition function -- the same GP-EI loop every self-driving-lab paper
runs, minus the dependency. This project takes zero heavy ML dependencies
(see the README's three runtime dependencies), so the GP is ~60 lines of numpy
rather than a `gpytorch`/`botorch` import, and the acquisition is optimised by
random search over the encoded cube rather than L-BFGS -- adequate at the
scale a physical instrument can actually generate data (tens to low hundreds
of trials), and a great deal easier to audit than a black-box library call in
a system that already insists on auditing everything else.

`CampaignRunner` handles the part this module deliberately does not: the
initial space-filling design. A GP fitted to fewer than a handful of points is
fitting noise, so the first `initial_design_size` trials come from
`ParameterSpace.latin_hypercube` and never reach this module at all.
"""

from __future__ import annotations

import math

import numpy as np

from .objective import Objective, Observation, scalarize
from .space import ParameterSpace

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)
_erf = np.vectorize(math.erf, otypes=[float])


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(z / _SQRT2))


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / _SQRT2PI


def expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, *, best: float, xi: float = 0.01
) -> np.ndarray:
    """Classic maximising EI. `xi` trades exploitation for exploration: a
    candidate must beat the incumbent by more than `xi` to be rewarded for
    certainty alone, which is what keeps the search from clustering forever
    on the first good point."""
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - best - xi) / sigma
    return (mu - best - xi) * _norm_cdf(z) + sigma * _norm_pdf(z)


class GaussianProcess:
    """Zero-mean-after-standardisation GP with a squared-exponential kernel.

    Deliberately minimal: one length scale shared across every encoded
    dimension (the unit-cube encoding in `space.py` already puts every axis on
    a comparable scale, which is what makes a single length scale defensible
    rather than a modelling shortcut).
    """

    def __init__(
        self, *, length_scale: float = 0.2, noise: float = 1e-6, signal_var: float = 1.0
    ) -> None:
        self.length_scale = length_scale
        self.noise = noise
        self.signal_var = signal_var
        self._X: np.ndarray | None = None
        self._L: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._y_mean = 0.0
        self._y_std = 1.0

    def _kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        sq_dist = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return self.signal_var * np.exp(-0.5 * sq_dist / (self.length_scale**2))

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._X = x
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) or 1.0
        y_norm = (y - self._y_mean) / self._y_std
        jitter = self.noise
        last_error: np.linalg.LinAlgError | None = None
        for _ in range(5):
            try:
                k = self._kernel(x, x) + jitter * np.eye(len(x))
                self._L = np.linalg.cholesky(k)
                break
            except np.linalg.LinAlgError as exc:
                last_error = exc
                jitter *= 10
        else:  # pragma: no cover - defensive; five 10x jitter steps never fail in practice
            raise last_error  # type: ignore[misc]
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y_norm))

    def predict(self, x_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._X is not None and self._L is not None and self._alpha is not None
        k_star = self._kernel(x_star, self._X)
        mu = k_star @ self._alpha
        v = np.linalg.solve(self._L, k_star.T)
        var = np.clip(self.signal_var - np.sum(v**2, axis=0), 1e-12, None)
        return mu * self._y_std + self._y_mean, np.sqrt(var) * self._y_std


class BayesianPlanner:
    """GP-EI over a `ParameterSpace`, scored on `objective.scalarize`'s output.

    Falls back to a random draw whenever there is not yet enough signal to fit
    honestly -- fewer than two scored points, or every scored point tied (a
    constant target, or every trial so far infeasible) -- rather than handing
    a Cholesky factorisation a degenerate problem and calling the result a
    prediction.
    """

    def __init__(
        self,
        *,
        xi: float = 0.01,
        candidates: int = 2000,
        length_scale: float = 0.2,
        noise: float = 1e-6,
    ) -> None:
        self.xi = xi
        self.candidates = candidates
        self.length_scale = length_scale
        self.noise = noise

    def suggest(
        self,
        space: ParameterSpace,
        objectives: list[Objective],
        observations: list[Observation],
        rng: np.random.Generator,
    ) -> dict:
        scalars = scalarize(objectives, observations)
        by_trial = {o.trial: o for o in observations}
        scored = [(by_trial[t].point, v) for t, v in scalars.items() if t in by_trial]
        if len(scored) < 2 or np.allclose([v for _, v in scored], scored[0][1]):
            return space.random(rng, 1)[0]

        x = space.encode_many([point for point, _ in scored])
        y = np.array([value for _, value in scored])
        gp = GaussianProcess(length_scale=self.length_scale, noise=self.noise)
        gp.fit(x, y)

        candidates = rng.random((self.candidates, space.width))
        mu, sigma = gp.predict(candidates)
        acquisition = expected_improvement(mu, sigma, best=float(y.max()), xi=self.xi)
        return space.decode(candidates[int(np.argmax(acquisition))])
