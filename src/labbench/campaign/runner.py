"""Closed-loop autonomous experimentation: propose, run, observe, replan.

`CampaignManager` is `ExperimentManager`'s sibling, at one remove: it owns
campaigns the way `ExperimentManager` owns runs, and every trial it drives
*is* a run, executed the ordinary way. Nothing here talks to a device
directly, calls the safety kernel, or writes to the ledger except through
`ExperimentManager.start`/`resume`/`cancel` -- the same discipline
`ExperimentManager` itself observes one layer down, so a campaign gets the
safety kernel, the approval broker and the provenance ledger for every trial
for free, rather than as a second implementation to keep in sync.

The one behaviour a campaign adds on top of a run: when a trial's hazard
needs a human signature, the *campaign* parks -- not just that trial's run --
exactly as a run parks around a step. A human answers with `approval.grant`,
then `campaign.resume` continues the trial that asked, and the loop carries
on from there.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import JobNotFound, ValidationError
from ..experiment.run import ExperimentManager, Run, RunStatus
from .objective import Observation, pareto_front, scalarize
from .planner import BayesianPlanner
from .spec import CampaignSpec

log = logging.getLogger("labbench.campaign")


class CampaignStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (
            CampaignStatus.SUCCEEDED, CampaignStatus.FAILED, CampaignStatus.CANCELLED
        )


class CampaignState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"campaign_{uuid.uuid4().hex[:12]}")
    name: str
    actor: str = "agent"
    status: CampaignStatus = CampaignStatus.PENDING
    budget: int
    #: Index of the next trial to run; also "trials completed" while running.
    trial: int = 0
    observations: list[Observation] = Field(default_factory=list)
    #: Set while a trial's run is in flight, including while parked.
    current_run_id: str | None = None
    created: float = Field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    message: str = ""

    @property
    def elapsed_s(self) -> float:
        if self.started is None:
            return 0.0
        return (self.finished or time.time()) - self.started

    def summary(self, *, include_observations: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "campaign_id": self.id, "name": self.name, "status": self.status.value,
            "trial": self.trial, "budget": self.budget,
            "elapsed_s": round(self.elapsed_s, 2), "message": self.message,
        }
        feasible = [o for o in self.observations if o.evaluated and o.feasible]
        out["feasible_trials"] = len(feasible)
        if self.current_run_id:
            out["current_run_id"] = self.current_run_id
        if include_observations:
            out["observations"] = [o.model_dump(mode="json") for o in self.observations]
        return out


class CampaignManager:
    """Owns every campaign, the way `ExperimentManager` owns every run."""

    retention_s: float = 3600.0

    def __init__(self, *, experiments: ExperimentManager, ledger: Any | None = None) -> None:
        self._experiments = experiments
        self._ledger = ledger
        self._specs: dict[str, CampaignSpec] = {}
        self._states: dict[str, CampaignState] = {}
        self._rngs: dict[str, np.random.Generator] = {}
        self._designs: dict[str, list[dict[str, Any]]] = {}
        self._planners: dict[str, BayesianPlanner] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancels: dict[str, asyncio.Event] = {}
        self._watchers: list[Callable[[CampaignState], Awaitable[None] | None]] = []

    def watch(self, fn: Callable[[CampaignState], Awaitable[None] | None]) -> None:
        self._watchers.append(fn)

    async def _notify(self, state: CampaignState) -> None:
        for fn in list(self._watchers):
            res = fn(state)
            if asyncio.iscoroutine(res):
                await res

    def _log(self, kind: str, state: CampaignState, **fields: Any) -> None:
        if self._ledger is not None:
            self._ledger.log(kind, actor=state.actor, run_id=state.current_run_id, **fields)

    # -- registry -----------------------------------------------------------

    def get(self, campaign_id: str) -> CampaignState:
        if campaign_id not in self._states:
            raise JobNotFound(
                f"no campaign {campaign_id!r}", campaign_id=campaign_id,
                known=sorted(self._states)[-10:],
            )
        return self._states[campaign_id]

    def spec(self, campaign_id: str) -> CampaignSpec:
        self.get(campaign_id)  # raises if unknown, with the same message
        return self._specs[campaign_id]

    def list(
        self, *, status: CampaignStatus | None = None, limit: int = 50
    ) -> list[CampaignState]:
        states = sorted(self._states.values(), key=lambda s: s.created, reverse=True)
        if status is not None:
            states = [s for s in states if s.status is status]
        return states[:limit]

    # -- lifecycle ------------------------------------------------------------

    def define(self, spec: CampaignSpec) -> str:
        campaign_id = f"campaign_{uuid.uuid4().hex[:12]}"
        self._specs[campaign_id] = spec
        return campaign_id

    def start(self, campaign_id: str, *, actor: str = "agent") -> CampaignState:
        spec = self._specs.get(campaign_id)
        if spec is None:
            raise ValidationError(
                f"no campaign {campaign_id!r}; call CampaignManager.define first"
            )
        state = CampaignState(id=campaign_id, name=spec.name, actor=actor, budget=spec.budget)
        self._states[campaign_id] = state
        rng = np.random.default_rng(spec.seed)
        self._rngs[campaign_id] = rng
        design_size = min(max(1, spec.initial_design_size), spec.budget)
        self._designs[campaign_id] = getattr(spec.space, spec.initial_design)(rng, design_size)
        self._planners[campaign_id] = BayesianPlanner()
        self._cancels[campaign_id] = asyncio.Event()
        self._log("campaign_start", state, payload={
            "name": spec.name, "budget": spec.budget,
            "objectives": [o.name for o in spec.objectives], "space": spec.space.names,
        })
        task = asyncio.create_task(self._drive(campaign_id), name=f"labbench:campaign:{campaign_id}")
        self._tasks[campaign_id] = task
        return state

    def resume(self, campaign_id: str) -> CampaignState:
        """Continue a campaign parked on `AWAITING_APPROVAL`, mirroring
        `ExperimentManager.resume`: call after a human has answered via
        `approval.grant` / `approval.deny`."""
        state = self.get(campaign_id)
        if state.status is not CampaignStatus.AWAITING_APPROVAL:
            raise ValidationError(
                f"campaign {campaign_id!r} is {state.status.value}, not awaiting approval",
                campaign_id=campaign_id, status=state.status.value,
            )
        self._cancels[campaign_id] = asyncio.Event()
        task = asyncio.create_task(
            self._drive(campaign_id), name=f"labbench:campaign:{campaign_id}"
        )
        self._tasks[campaign_id] = task
        return state

    async def cancel(self, campaign_id: str, *, reason: str = "agent requested") -> CampaignState:
        state = self.get(campaign_id)
        if state.status.terminal:
            return state
        event = self._cancels.get(campaign_id)
        if event is not None:
            event.set()  # cooperative: let the in-flight trial finish or park
        if state.status in (CampaignStatus.PENDING, CampaignStatus.AWAITING_APPROVAL):
            state.status = CampaignStatus.CANCELLED
            state.message = f"cancelled: {reason}"
            state.finished = time.time()
            self._log("campaign_end", state, payload={"status": "cancelled", "reason": reason})
            await self._notify(state)
        return state

    # -- results --------------------------------------------------------------

    def best(self, campaign_id: str) -> dict[str, Any]:
        """Best trial so far by the scalarised score, plus the Pareto front.

        Available at any point in a campaign's life, not only once it is
        terminal: a long campaign is worth checking on.
        """
        spec = self.spec(campaign_id)
        state = self.get(campaign_id)
        scalars = scalarize(spec.objectives, state.observations)
        by_trial = {o.trial: o for o in state.observations}
        best_trial = max(scalars, key=lambda t: scalars[t]) if scalars else None
        return {
            "campaign_id": campaign_id,
            "trials_evaluated": len(state.observations),
            "best_trial": best_trial,
            "best_point": by_trial[best_trial].point if best_trial is not None else None,
            "best_values": by_trial[best_trial].values if best_trial is not None else None,
            "best_score": scalars.get(best_trial) if best_trial is not None else None,
            "pareto_front": pareto_front(spec.objectives, state.observations),
        }

    # -- execution --------------------------------------------------------

    async def _drive(self, campaign_id: str) -> None:
        state = self._states[campaign_id]
        spec = self._specs[campaign_id]
        state.status = CampaignStatus.RUNNING
        state.started = state.started or time.time()
        await self._notify(state)
        cancel = self._cancels[campaign_id]

        try:
            while state.trial < spec.budget:
                if cancel.is_set():
                    state.status = CampaignStatus.CANCELLED
                    state.message = "cancelled"
                    break
                point = self._propose(campaign_id)
                outcome = await self._run_trial(campaign_id, point, cancel)
                if outcome == "parked":
                    return  # AWAITING_APPROVAL; a later resume() re-enters here
                if outcome == "stopped":
                    break
                if self._target_reached(spec, state.observations):
                    state.status = CampaignStatus.SUCCEEDED
                    state.message = "an objective's target was reached"
                    break
            else:
                state.status = CampaignStatus.SUCCEEDED
                state.message = "budget exhausted"
        except asyncio.CancelledError:
            state.status = CampaignStatus.CANCELLED
            state.message = "cancelled"
        finally:
            if state.status.terminal:
                state.finished = time.time()
                self._log("campaign_end", state, payload={
                    "status": state.status.value, "trials": state.trial,
                    "elapsed_s": round(state.elapsed_s, 3),
                })
                self._tasks.pop(campaign_id, None)
            await self._notify(state)

    def _propose(self, campaign_id: str) -> dict[str, Any]:
        spec = self._specs[campaign_id]
        state = self._states[campaign_id]
        design = self._designs[campaign_id]
        completed = len(state.observations)
        if completed < len(design):
            return design[completed]
        rng = self._rngs[campaign_id]
        planner = self._planners[campaign_id]
        return planner.suggest(spec.space, spec.objectives, state.observations, rng)

    @staticmethod
    def _target_reached(spec: CampaignSpec, observations: list[Observation]) -> bool:
        if not observations:
            return False
        latest = observations[-1]
        return any(
            o.target is not None and o.name in latest.values and o.reached(latest.values[o.name])
            for o in spec.objectives
        )

    async def _run_trial(self, campaign_id: str, point: dict[str, Any], cancel: asyncio.Event) -> str:
        """Returns 'continue', 'parked', or 'stopped'. Re-entrant across a
        park/resume the same way `ExperimentManager._run_step` is.

        A trial whose run ends `FAILED` -- a denied approval, a transient
        driver fault -- falls through to `_evaluate`, which records it as
        unevaluated and infeasible and lets the campaign carry on to the next
        proposal. A long unattended campaign should not forfeit its whole
        remaining budget to one refused or flaky trial; only `CANCELLED`
        (an explicit stop) and `AWAITING_APPROVAL` (nothing to evaluate yet)
        interrupt the loop itself.
        """
        state = self._states[campaign_id]
        spec = self._specs[campaign_id]

        if state.current_run_id is not None:
            run = self._experiments.get(state.current_run_id)
            if run.status is RunStatus.AWAITING_APPROVAL:
                run = self._experiments.resume(run.id, spec.protocol)
        else:
            protocol_id = self._experiments.define(spec.protocol)
            run = self._experiments.start(
                spec.protocol, protocol_id=protocol_id, variables=point, actor=state.actor,
            )
            state.current_run_id = run.id
            self._log("campaign_trial_start", state, payload={"trial": state.trial, "point": point})
            await self._notify(state)

        run = await self._await_run(run.id, cancel)

        if run.status is RunStatus.AWAITING_APPROVAL:
            state.status = CampaignStatus.AWAITING_APPROVAL
            state.message = run.message
            await self._notify(state)
            return "parked"

        if run.status is RunStatus.CANCELLED:
            state.status = CampaignStatus.CANCELLED
            state.message = "trial cancelled"
            return "stopped"

        observation = self._evaluate(spec, state.trial, point, run)
        state.observations.append(observation)
        state.trial += 1
        state.current_run_id = None
        self._log("campaign_trial_end", state, payload={
            "trial": observation.trial, "feasible": observation.feasible,
            "values": observation.values, "violations": observation.violations,
        })
        await self._notify(state)
        return "continue"

    async def _await_run(self, run_id: str, cancel: asyncio.Event) -> Run:
        """Poll a run to its next quiet point: terminal, or parked on approval.

        A poll rather than a callback for the same reason `labbench experiment
        run` polls at the CLI: it is simple, it is obviously correct, and a
        campaign trial is not on a latency budget an instrument would notice.

        The sleep comes *before* the first check, not after: a caller reaches
        here immediately after `experiments.start`/`resume`, which only
        schedules that run's driving task rather than running any of it
        synchronously. Checking before yielding once would read the run's
        status from before the call was made -- for a resumed run that status
        is `AWAITING_APPROVAL` again, which this loop also treats as a
        legitimate place to stop, so without the yield a resume can never be
        observed to have happened at all.
        """
        while True:
            await asyncio.sleep(0.05)
            if cancel.is_set():
                await self._experiments.cancel(run_id, reason="campaign cancelled")
            run = self._experiments.get(run_id)
            if run.status.terminal or run.status is RunStatus.AWAITING_APPROVAL:
                return run

    def _evaluate(
        self, spec: CampaignSpec, trial: int, point: dict[str, Any], run: Run
    ) -> Observation:
        if run.status is not RunStatus.SUCCEEDED:
            return Observation(
                trial=trial, point=point, evaluated=False, feasible=False,
                violations=[run.message or f"run ended {run.status.value}"], run_id=run.id,
            )
        scope = {"steps": {r.label: {"result": r.result or {}} for r in run.results}}
        values: dict[str, float] = {}
        violations: list[str] = []
        feasible = True
        for objective in spec.objectives:
            try:
                value = objective.extract(scope)
            except ValidationError as exc:
                violations.append(f"{objective.name}: {exc.message}")
                feasible = False
                continue
            values[objective.name] = value
            ok, reason = objective.satisfied(value)
            if not ok:
                feasible = False
                violations.append(reason)
        return Observation(
            trial=trial, point=point, values=values, feasible=feasible,
            violations=violations, run_id=run.id, evaluated=True,
        )

    async def shutdown(self) -> None:
        for campaign_id in list(self._cancels):
            await self.cancel(campaign_id, reason="server shutting down")
        tasks = list(self._tasks.values())
        if tasks:
            await asyncio.wait(tasks, timeout=10)
        for task in tasks:
            if not task.done():
                task.cancel()
