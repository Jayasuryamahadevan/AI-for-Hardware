"""CampaignSpec validation and CampaignManager execution: propose, run,
observe, replan, through the real gateway/safety-kernel/ledger front door."""

from __future__ import annotations

import asyncio

import pytest

from labbench.campaign import CampaignSpec, CampaignStatus
from labbench.protocol.jsonrpc import Request
from labbench.protocol.router import RpcContext

FOCUS_SPEC = {
    "name": "autofocus-bo",
    "protocol": {
        "name": "focus-trial",
        "steps": [
            {"label": "focus", "device": "scope1", "feature": "FocusControl", "command": "move_z",
             "args": {"z_um": "${z_um}"}},
            {"label": "snap", "device": "scope1", "feature": "Camera", "command": "snap",
             "args": {"exposure_ms": "${exposure_ms}"}},
        ],
    },
    "space": {
        "dimensions": [
            {"name": "z_um", "type": "continuous", "low": 0.0, "high": 190.0, "unit": "um"},
            {"name": "exposure_ms", "type": "continuous", "low": 5.0, "high": 200.0,
             "unit": "ms", "log": True},
        ],
    },
    "objectives": [
        {"name": "focus", "path": "steps.snap.result.focus_score", "direction": "maximize"},
        {"name": "saturation", "path": "steps.snap.result.saturated_fraction",
         "direction": "constrain", "maximum": 0.5},
    ],
    "budget": 6,
    "initial_design_size": 3,
    "seed": 42,
}

GAS_SPEC = {
    "name": "gas-sweep",
    "protocol": {
        "name": "gas-trial",
        "steps": [
            {"label": "set_gas", "device": "incubator1", "feature": "GasControl", "command": "set_co2",
             "args": {"co2_pct": "${co2_pct}"}},
        ],
    },
    "space": {"dimensions": [{"name": "co2_pct", "type": "continuous", "low": 3.0, "high": 8.0}]},
    "objectives": [{"name": "co2", "path": "steps.set_gas.result.co2_pct", "direction": "maximize"}],
    "budget": 2,
    "initial_design_size": 2,
    "seed": 1,
}


async def wait_terminal(manager, campaign_id, *, timeout: float = 20.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        state = manager.get(campaign_id)
        if state.status.terminal:
            return state
        if asyncio.get_event_loop().time() >= deadline:
            raise AssertionError(f"campaign {campaign_id} did not finish within {timeout}s")
        await asyncio.sleep(0.02)


async def wait_for_status(manager, campaign_id, status, *, timeout: float = 20.0):
    """Sleeps before every check, including the first.

    A caller of this helper right after `resume()` is in exactly the race
    `CampaignManager._await_run` documents: `resume()` only *schedules* the
    campaign's driving task, so checking before the first yield would read
    the status from before resume was even called -- `AWAITING_APPROVAL`
    again, for a campaign that has in fact moved on.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        await asyncio.sleep(0.02)
        state = manager.get(campaign_id)
        if state.status is status or state.status.terminal:
            return state
        if asyncio.get_event_loop().time() >= deadline:
            raise AssertionError(f"campaign {campaign_id} never reached {status} within {timeout}s")


class TestCampaignSpecValidation:
    def test_valid_spec_has_no_problems(self, gateway):
        spec = CampaignSpec.model_validate(FOCUS_SPEC)
        assert spec.validate_against(gateway) == []

    def test_dimension_not_bound_to_any_step_is_reported(self, gateway):
        bad = {**FOCUS_SPEC, "space": {"dimensions": [
            *FOCUS_SPEC["space"]["dimensions"],
            {"name": "unused", "low": 0.0, "high": 1.0},
        ]}}
        problems = CampaignSpec.model_validate(bad).validate_against(gateway)
        assert any("unused" in p and "not bound" in p for p in problems)

    def test_dimension_extreme_outside_envelope_is_reported(self, gateway):
        bad = {**FOCUS_SPEC, "space": {"dimensions": [
            {"name": "z_um", "type": "continuous", "low": -50.0, "high": 99999.0},
            FOCUS_SPEC["space"]["dimensions"][1],
        ]}}
        problems = CampaignSpec.model_validate(bad).validate_against(gateway)
        assert any("out of envelope" in p for p in problems)

    def test_objective_path_not_referencing_a_step_is_reported(self, gateway):
        bad = {**FOCUS_SPEC, "objectives": [
            {"name": "focus", "path": "steps.nope.result.focus_score", "direction": "maximize"},
        ]}
        problems = CampaignSpec.model_validate(bad).validate_against(gateway)
        assert any("does not reference a step result" in p for p in problems)

    def test_unknown_device_is_still_reported_via_protocol_validation(self, gateway):
        bad = {**FOCUS_SPEC, "protocol": {
            "name": "x",
            "steps": [{"device": "nope", "feature": "F", "command": "c", "args": {"z_um": "${z_um}"}}],
        }}
        problems = CampaignSpec.model_validate(bad).validate_against(gateway)
        assert any("nope" in p for p in problems)

    def test_needs_at_least_one_optimised_objective(self):
        bad = {**FOCUS_SPEC, "objectives": [
            {"name": "saturation", "path": "steps.snap.result.saturated_fraction",
             "direction": "constrain", "maximum": 0.5},
        ]}
        with pytest.raises(ValueError, match="at least one maximize/minimize"):
            CampaignSpec.model_validate(bad)

    def test_duplicate_objective_names_rejected(self):
        bad = {**FOCUS_SPEC, "objectives": [
            FOCUS_SPEC["objectives"][0], FOCUS_SPEC["objectives"][0],
        ]}
        with pytest.raises(ValueError, match="duplicate objective"):
            CampaignSpec.model_validate(bad)


class TestCampaignExecution:
    async def test_runs_to_completion_and_records_every_trial(self, gateway):
        spec = CampaignSpec.model_validate(FOCUS_SPEC)
        cid = gateway.campaigns.define(spec)
        gateway.campaigns.start(cid, actor="agent:test")
        state = await wait_terminal(gateway.campaigns, cid)

        assert state.status is CampaignStatus.SUCCEEDED
        assert state.message == "budget exhausted"
        assert state.trial == spec.budget
        assert len(state.observations) == spec.budget
        assert all(o.evaluated and o.feasible for o in state.observations)
        # Every point actually sits inside the declared space.
        assert all(spec.space.contains(o.point) for o in state.observations)

    async def test_ledger_records_every_trial_boundary(self, gateway):
        spec = CampaignSpec.model_validate(FOCUS_SPEC)
        cid = gateway.campaigns.define(spec)
        gateway.campaigns.start(cid, actor="agent:test")
        await wait_terminal(gateway.campaigns, cid)

        kinds = [r.kind for r in gateway.ledger.query(limit=500)]
        assert kinds.count("campaign_start") == 1
        assert kinds.count("campaign_trial_start") == spec.budget
        assert kinds.count("campaign_trial_end") == spec.budget
        assert kinds.count("campaign_end") == 1
        # And every trial's own protocol run is a real, independently visible run.
        assert kinds.count("run_start") == spec.budget

    async def test_best_is_available_mid_campaign_and_improves_monotonically(self, gateway):
        spec = CampaignSpec.model_validate(FOCUS_SPEC)
        cid = gateway.campaigns.define(spec)
        gateway.campaigns.start(cid, actor="agent:test")

        seen_scores = []
        for _ in range(400):
            state = gateway.campaigns.get(cid)
            if state.observations:
                seen_scores.append(gateway.campaigns.best(cid)["best_score"])
            if state.status.terminal:
                break
            await asyncio.sleep(0.02)

        # The running best can only ever go up (or stay flat), never regress.
        assert seen_scores == sorted(seen_scores)
        final_best = gateway.campaigns.best(cid)
        assert final_best["best_trial"] is not None
        assert spec.space.contains(final_best["best_point"])
        assert final_best["pareto_front"] == [final_best["best_trial"]]  # single objective

    async def test_infeasible_trials_are_recorded_but_do_not_stop_the_campaign(self, gateway):
        # Force saturation over the whole exposure range so every trial is infeasible.
        impossible = {**FOCUS_SPEC, "objectives": [
            FOCUS_SPEC["objectives"][0],
            {"name": "saturation", "path": "steps.snap.result.saturated_fraction",
             "direction": "constrain", "maximum": -1.0},
        ]}
        spec = CampaignSpec.model_validate(impossible)
        cid = gateway.campaigns.define(spec)
        gateway.campaigns.start(cid, actor="agent:test")
        state = await wait_terminal(gateway.campaigns, cid)

        assert state.status is CampaignStatus.SUCCEEDED  # budget still exhausts cleanly
        assert state.trial == spec.budget
        assert all(not o.feasible for o in state.observations)
        best = gateway.campaigns.best(cid)
        assert best["pareto_front"] == []  # no feasible trial to report

    async def test_target_reached_stops_the_campaign_early(self, gateway):
        # A target no real focus_score could fail to clear: reached on trial 0,
        # without entangling this with the incubator's require_approval rule.
        spec = CampaignSpec.model_validate({
            **FOCUS_SPEC,
            "objectives": [{"name": "focus", "path": "steps.snap.result.focus_score",
                            "direction": "maximize", "target": -1e9}],
            "budget": 50, "initial_design_size": 1,
        })
        cid = gateway.campaigns.define(spec)
        gateway.campaigns.start(cid, actor="agent:test")
        state = await wait_terminal(gateway.campaigns, cid)
        assert state.status is CampaignStatus.SUCCEEDED
        assert state.message == "an objective's target was reached"
        assert state.trial < 50


class TestCampaignApprovalParkAndResume:
    async def test_campaign_parks_on_awaiting_approval(self, gateway):
        spec = CampaignSpec.model_validate(GAS_SPEC)
        cid = gateway.campaigns.define(spec)
        gateway.campaigns.start(cid, actor="agent:test")
        state = await wait_for_status(gateway.campaigns, cid, CampaignStatus.AWAITING_APPROVAL)
        assert state.status is CampaignStatus.AWAITING_APPROVAL
        assert gateway.approvals.pending()

    async def test_resume_after_grant_advances_every_trial(self, gateway):
        spec = CampaignSpec.model_validate(GAS_SPEC)
        cid = gateway.campaigns.define(spec)
        gateway.campaigns.start(cid, actor="agent:test")

        for _ in range(spec.budget):
            await wait_for_status(gateway.campaigns, cid, CampaignStatus.AWAITING_APPROVAL)
            state = gateway.campaigns.get(cid)
            if state.status.terminal:
                break
            [pending] = gateway.approvals.pending()
            await gateway.approvals.grant(pending.id, approver="human:alice")
            gateway.campaigns.resume(cid)

        finished = await wait_terminal(gateway.campaigns, cid)
        assert finished.status is CampaignStatus.SUCCEEDED
        assert finished.trial == spec.budget
        assert all(o.evaluated for o in finished.observations)

    async def test_resume_after_deny_records_an_infeasible_trial_and_moves_on(self, gateway):
        """A denied trial does not end the campaign -- see `_run_trial`'s
        module docstring: a campaign is resilient to one bad trial (a denial,
        a transient fault) the same way `continue_on_error` is for one step,
        because a long unattended campaign should not lose its whole budget
        to a single refused action."""
        spec = CampaignSpec.model_validate(GAS_SPEC)  # budget=2, every trial needs approval
        cid = gateway.campaigns.define(spec)
        gateway.campaigns.start(cid, actor="agent:test")

        for _ in range(spec.budget):
            await wait_for_status(gateway.campaigns, cid, CampaignStatus.AWAITING_APPROVAL)
            state = gateway.campaigns.get(cid)
            if state.status.terminal:
                break
            [pending] = gateway.approvals.pending()
            await gateway.approvals.deny(pending.id, approver="human:bob", reason="not now")
            gateway.campaigns.resume(cid)

        finished = await wait_terminal(gateway.campaigns, cid)
        assert finished.status is CampaignStatus.SUCCEEDED  # budget still exhausts cleanly
        assert finished.trial == spec.budget
        assert all(not o.evaluated and not o.feasible for o in finished.observations)

    async def test_cancel_while_parked(self, gateway):
        spec = CampaignSpec.model_validate(GAS_SPEC)
        cid = gateway.campaigns.define(spec)
        gateway.campaigns.start(cid, actor="agent:test")
        await wait_for_status(gateway.campaigns, cid, CampaignStatus.AWAITING_APPROVAL)
        cancelled = await gateway.campaigns.cancel(cid, reason="changed my mind")
        assert cancelled.status is CampaignStatus.CANCELLED

    async def test_cancel_mid_flight_stops_before_the_budget_is_spent(self, gateway):
        spec = CampaignSpec.model_validate(FOCUS_SPEC)
        cid = gateway.campaigns.define(spec)
        gateway.campaigns.start(cid, actor="agent:test")
        await asyncio.sleep(0.05)
        await gateway.campaigns.cancel(cid, reason="stop early")
        state = await wait_terminal(gateway.campaigns, cid)
        assert state.status is CampaignStatus.CANCELLED
        assert state.trial < spec.budget


class TestCampaignRpc:
    async def test_validate_start_status_best_over_rpc(self, gateway):
        ctx = RpcContext(actor="rpc:test")
        resp = await gateway.router.dispatch(Request("campaign.validate", {"spec": GAS_SPEC}, id=1), ctx)
        assert resp is not None and resp.result["valid"] is True

        resp = await gateway.router.dispatch(Request("campaign.start", {"spec": GAS_SPEC}, id=2), ctx)
        campaign_id = resp.result["campaign"]["campaign_id"]

        for _ in range(GAS_SPEC["budget"]):
            state = await wait_for_status(gateway.campaigns, campaign_id, CampaignStatus.AWAITING_APPROVAL)
            if state.status.terminal:
                break
            [pending] = gateway.approvals.pending()
            await gateway.approvals.grant(pending.id, approver="human:rpc")
            resp = await gateway.router.dispatch(
                Request("campaign.resume", {"campaign_id": campaign_id}, id=3), ctx
            )
            assert resp is not None and "error" not in resp.to_dict()

        await wait_terminal(gateway.campaigns, campaign_id)
        resp = await gateway.router.dispatch(
            Request("campaign.status", {"campaign_id": campaign_id}, id=4), ctx
        )
        assert resp.result["status"] == "succeeded"

        resp = await gateway.router.dispatch(
            Request("campaign.best", {"campaign_id": campaign_id}, id=5), ctx
        )
        assert resp.result["best_trial"] is not None

    async def test_invalid_spec_is_rejected_before_starting(self, gateway):
        bad = {**GAS_SPEC, "space": {"dimensions": [
            {"name": "co2_pct", "type": "continuous", "low": -50.0, "high": 99999.0},
        ]}}
        ctx = RpcContext(actor="rpc:test")
        resp = await gateway.router.dispatch(Request("campaign.start", {"spec": bad}, id=1), ctx)
        assert resp is not None
        payload = resp.to_dict()
        assert "error" in payload
        assert payload["error"]["code"] < 0

    def test_campaign_not_found_raises(self, gateway):
        from labbench.core.errors import JobNotFound

        with pytest.raises(JobNotFound):
            gateway.campaigns.get("campaign_does_not_exist")

    async def test_gateway_describe_reports_running_campaigns(self, gateway):
        spec = CampaignSpec.model_validate(FOCUS_SPEC)
        cid = gateway.campaigns.define(spec)
        gateway.campaigns.start(cid, actor="agent:test")
        await asyncio.sleep(0.02)
        assert gateway.describe()["campaigns_running"] == 1
        await wait_terminal(gateway.campaigns, cid)
        assert gateway.describe()["campaigns_running"] == 0
