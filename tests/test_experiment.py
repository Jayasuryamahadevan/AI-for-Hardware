"""Protocols and runs: validation, execution, the approval park/resume dance, replay."""

from __future__ import annotations

import asyncio

import pytest

from labbench.experiment import Protocol, RunStatus, dry_run_protocol, replay_run


async def wait_terminal(manager, run_id, *, timeout: float = 10.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        run = manager.get(run_id)
        if run.status.terminal:
            return run
        if asyncio.get_event_loop().time() >= deadline:
            raise AssertionError(f"run {run_id} did not finish within {timeout}s")
        await asyncio.sleep(0.02)


IMAGE_PROTOCOL = {
    "name": "quick-image",
    "steps": [
        {"label": "home", "device": "scope1", "feature": "MotionControl", "command": "home"},
        {"label": "focus", "device": "scope1", "feature": "FocusControl", "command": "move_z",
         "args": {"z_um": 120.0}},
        {"label": "snap", "device": "scope1", "feature": "Camera", "command": "snap",
         "args": {"exposure_ms": "${exposure}"}},
    ],
    "variables": {"exposure": 40.0},
}

GAS_PROTOCOL = {
    "name": "gas-change",
    "steps": [
        {"label": "set_gas", "device": "incubator1", "feature": "GasControl", "command": "set_co2",
         "args": {"co2_pct": 5.0}},
    ],
}


class TestProtocolValidation:
    def test_valid_protocol_has_no_problems(self, gateway):
        problems = Protocol.model_validate(IMAGE_PROTOCOL).validate_against(gateway)
        assert problems == []

    def test_unknown_device_is_reported(self, gateway):
        bad = {"name": "x", "steps": [{"device": "nope", "feature": "F", "command": "c"}]}
        problems = Protocol.model_validate(bad).validate_against(gateway)
        assert any("nope" in p for p in problems)

    def test_unknown_parameter_is_reported(self, gateway):
        bad = {
            "name": "x",
            "steps": [{"device": "scope1", "feature": "FocusControl", "command": "move_z",
                       "args": {"not_a_real_arg": 1}}],
        }
        problems = Protocol.model_validate(bad).validate_against(gateway)
        assert any("unknown parameter" in p for p in problems)

    def test_out_of_envelope_literal_is_reported(self, gateway):
        bad = {
            "name": "x",
            "steps": [{"device": "scope1", "feature": "FocusControl", "command": "move_z",
                       "args": {"z_um": 99999.0}}],
        }
        problems = Protocol.model_validate(bad).validate_against(gateway)
        assert problems  # exceeds the 0-300um travel

    def test_templated_reference_to_a_future_step_is_reported(self, gateway):
        bad = {
            "name": "x",
            "steps": [{"device": "scope1", "feature": "FocusControl", "command": "move_z",
                       "args": {"z_um": "${steps.later.result.z_um}"}}],
        }
        problems = Protocol.model_validate(bad).validate_against(gateway)
        assert any("later" in p for p in problems)

    def test_duplicate_labels_are_reported(self, gateway):
        bad = {
            "name": "x",
            "steps": [
                {"label": "dup", "device": "scope1", "feature": "MotionControl", "command": "home"},
                {"label": "dup", "device": "scope1", "feature": "MotionControl", "command": "home"},
            ],
        }
        problems = Protocol.model_validate(bad).validate_against(gateway)
        assert any("duplicate" in p for p in problems)


class TestReferenceResolution:
    def test_resolves_run_variable(self):
        protocol = Protocol.model_validate(IMAGE_PROTOCOL)
        step = protocol.steps[2]
        args = protocol.resolve_args(step, {}, {"exposure": 55.0})
        assert args == {"exposure_ms": 55.0}

    def test_resolves_prior_step_result(self):
        protocol = Protocol.model_validate({
            "name": "x",
            "steps": [{"label": "a", "device": "d", "feature": "F", "command": "c",
                       "args": {"z": "${steps.b.result.value}"}}],
        })
        args = protocol.resolve_args(
            protocol.steps[0], {"b": {"result": {"value": 42}}}, {},
        )
        assert args == {"z": 42}

    def test_unresolved_reference_raises(self):
        from labbench.core.errors import ValidationError

        protocol = Protocol.model_validate({
            "name": "x",
            "steps": [{"device": "d", "feature": "F", "command": "c", "args": {"z": "${nope}"}}],
        })
        with pytest.raises(ValidationError):
            protocol.resolve_args(protocol.steps[0], {}, {})


class TestExecution:
    async def test_successful_run(self, gateway):
        protocol = Protocol.model_validate(IMAGE_PROTOCOL)
        pid = gateway.experiments.define(protocol)
        run = gateway.experiments.start(protocol, protocol_id=pid, actor="agent:test")
        finished = await wait_terminal(gateway.experiments, run.id)
        assert finished.status is RunStatus.SUCCEEDED
        assert all(r.status == "succeeded" for r in finished.results)

    async def test_run_is_recorded_in_the_ledger(self, gateway):
        protocol = Protocol.model_validate(IMAGE_PROTOCOL)
        pid = gateway.experiments.define(protocol)
        run = gateway.experiments.start(protocol, protocol_id=pid)
        await wait_terminal(gateway.experiments, run.id)
        kinds = [r.kind for r in gateway.ledger.query(run_id=run.id)]
        assert "run_start" in kinds
        assert "run_step" in kinds
        assert "run_end" in kinds

    async def test_failed_step_stops_the_run_by_default(self, gateway):
        bad = {
            "name": "x",
            "steps": [
                {"label": "a", "device": "scope1", "feature": "FocusControl", "command": "move_z",
                 "args": {"z_um": 99999.0}},
                {"label": "b", "device": "scope1", "feature": "MotionControl", "command": "home"},
            ],
        }
        protocol = Protocol.model_validate(bad)
        run = gateway.experiments.start(protocol, actor="agent:test")
        finished = await wait_terminal(gateway.experiments, run.id)
        assert finished.status is RunStatus.FAILED
        assert finished.results[1].status == "pending"

    async def test_continue_on_error_keeps_going(self, gateway):
        bad = {
            "name": "x",
            "steps": [
                {"label": "a", "device": "scope1", "feature": "FocusControl", "command": "move_z",
                 "args": {"z_um": 99999.0}, "continue_on_error": True},
                {"label": "b", "device": "scope1", "feature": "MotionControl", "command": "home"},
            ],
        }
        protocol = Protocol.model_validate(bad)
        run = gateway.experiments.start(protocol, actor="agent:test")
        finished = await wait_terminal(gateway.experiments, run.id)
        assert finished.status is RunStatus.SUCCEEDED
        assert finished.results[0].status == "failed"
        assert finished.results[1].status == "succeeded"


class TestApprovalParkAndResume:
    async def test_run_parks_on_awaiting_approval(self, gateway):
        protocol = Protocol.model_validate(GAS_PROTOCOL)
        pid = gateway.experiments.define(protocol)
        run = gateway.experiments.start(protocol, protocol_id=pid, actor="agent:test")

        async def is_parked():
            return gateway.experiments.get(run.id).status is RunStatus.AWAITING_APPROVAL

        for _ in range(200):
            if await is_parked():
                break
            await asyncio.sleep(0.02)
        parked = gateway.experiments.get(run.id)
        assert parked.status is RunStatus.AWAITING_APPROVAL
        assert gateway.approvals.pending()

    async def test_resume_after_grant_completes_the_run(self, gateway):
        protocol = Protocol.model_validate(GAS_PROTOCOL)
        pid = gateway.experiments.define(protocol)
        run = gateway.experiments.start(protocol, protocol_id=pid, actor="agent:test")
        for _ in range(200):
            if gateway.experiments.get(run.id).status is RunStatus.AWAITING_APPROVAL:
                break
            await asyncio.sleep(0.02)
        [pending] = gateway.approvals.pending()
        await gateway.approvals.grant(pending.id, approver="human:alice", reason="ok")
        gateway.experiments.resume(run.id, protocol)
        finished = await wait_terminal(gateway.experiments, run.id)
        assert finished.status is RunStatus.SUCCEEDED
        # the retry actually happened, not just fell through
        assert finished.results[0].status == "succeeded"
        assert finished.results[0].result is not None

    async def test_resume_after_deny_fails_the_run(self, gateway):
        protocol = Protocol.model_validate(GAS_PROTOCOL)
        pid = gateway.experiments.define(protocol)
        run = gateway.experiments.start(protocol, protocol_id=pid, actor="agent:test")
        for _ in range(200):
            if gateway.experiments.get(run.id).status is RunStatus.AWAITING_APPROVAL:
                break
            await asyncio.sleep(0.02)
        [pending] = gateway.approvals.pending()
        await gateway.approvals.deny(pending.id, approver="human:bob", reason="not now")
        gateway.experiments.resume(run.id, protocol)
        finished = await wait_terminal(gateway.experiments, run.id)
        assert finished.status is RunStatus.FAILED

    async def test_cancel_while_parked(self, gateway):
        protocol = Protocol.model_validate(GAS_PROTOCOL)
        pid = gateway.experiments.define(protocol)
        run = gateway.experiments.start(protocol, protocol_id=pid, actor="agent:test")
        for _ in range(200):
            if gateway.experiments.get(run.id).status is RunStatus.AWAITING_APPROVAL:
                break
            await asyncio.sleep(0.02)
        cancelled = await gateway.experiments.cancel(run.id, reason="changed my mind")
        assert cancelled.status is RunStatus.CANCELLED


class TestReplay:
    async def test_replay_reconstructs_a_finished_run(self, gateway):
        protocol = Protocol.model_validate(IMAGE_PROTOCOL)
        run = gateway.experiments.start(protocol, actor="agent:test")
        await wait_terminal(gateway.experiments, run.id)
        summary = replay_run(gateway.ledger, run.id)
        assert summary.found
        assert summary.outcome == "succeeded"
        assert len(summary.steps) > 0

    def test_replay_unknown_run(self, gateway):
        summary = replay_run(gateway.ledger, "run_does_not_exist")
        assert not summary.found


class TestDryRun:
    async def test_dry_run_never_touches_hardware(self, gateway):
        before = gateway.device("scope1").x_um
        report = await dry_run_protocol(Protocol.model_validate(IMAGE_PROTOCOL), gateway)
        assert gateway.device("scope1").x_um == before
        assert isinstance(report["feasible"], bool)
        assert len(report["steps"]) == 3

    async def test_dry_run_stops_at_first_infeasible_step(self, gateway):
        bad = {
            "name": "x",
            "steps": [
                {"label": "bad_move", "device": "scope1", "feature": "MotionControl",
                 "command": "move_absolute", "args": {"x_um": -1.0, "y_um": 0.0}},
                {"label": "never", "device": "scope1", "feature": "MotionControl", "command": "home"},
            ],
        }
        report = await dry_run_protocol(Protocol.model_validate(bad), gateway)
        assert report["feasible"] is False
        assert len(report["steps"]) == 1
