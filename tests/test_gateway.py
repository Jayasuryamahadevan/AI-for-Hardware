"""End-to-end: request -> ledger -> safety kernel -> [approval] -> execute -> ledger.

Drives the real shipped simulated lab (`configs/simulated-lab.yaml`) through
`Gateway.invoke`, the one path an agent actually uses -- these tests are the
closest thing this suite has to "does the product work".
"""

from __future__ import annotations

import asyncio

import pytest

from labbench.core.errors import ApprovalDenied, ApprovalRequired, SafetyViolation


async def _wait_for_job(gateway, job_id: str, *, timeout: float = 10.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        job = gateway.jobs.get(job_id)
        if job.status.terminal:
            return job
        if asyncio.get_event_loop().time() >= deadline:
            raise AssertionError(f"job {job_id} did not finish within {timeout}s")
        await asyncio.sleep(0.02)


class TestInlineCommands:
    async def test_reads_are_never_gated(self, gateway):
        # A read never crosses the safety kernel at all: no gate to pass.
        sample = await gateway.device("scope1").read("MotionControl", "x_um")
        assert sample.value is not None

    async def test_home_runs_as_an_observable_job(self, gateway):
        # home() is declared observable, so invoke() returns a handle, not
        # the result inline -- the same shape a forty-minute tile scan uses.
        started = await gateway.invoke("scope1", "MotionControl", "home", reason="test")
        assert started["accepted"] is True
        job = await _wait_for_job(gateway, started["job"]["job_id"])
        assert job.status.value == "succeeded"
        assert job.result["homed"] is True

    async def test_job_completion_is_logged_to_the_ledger(self, gateway):
        started = await gateway.invoke("scope1", "MotionControl", "home", reason="setup")
        await _wait_for_job(gateway, started["job"]["job_id"])
        records = gateway.ledger.query(device_id="scope1")
        kinds = [r.kind for r in records]
        assert "command_request" in kinds
        assert "safety_decision" in kinds
        assert "command_result" in kinds

    async def test_reason_is_recorded(self, gateway):
        await gateway.invoke("scope1", "MotionControl", "home", reason="prepare for imaging")
        records = gateway.ledger.query(kind="command_request", device_id="scope1")
        assert records[-1].reason == "prepare for imaging"

    async def test_non_observable_command_returns_inline(self, gateway):
        started = await gateway.invoke("scope1", "MotionControl", "home", reason="setup")
        await _wait_for_job(gateway, started["job"]["job_id"])
        result = await gateway.invoke(
            "scope1", "Illumination", "set_intensity", {"intensity_pct": 40.0}, reason="test",
        )
        assert result["intensity_pct"] == 40.0


class TestObservableCommands:
    async def test_observable_command_returns_a_job_handle(self, gateway):
        started = await gateway.invoke("scope1", "MotionControl", "home", reason="setup")
        await _wait_for_job(gateway, started["job"]["job_id"])
        result = await gateway.invoke(
            "scope1", "Camera", "acquire_tile", {"columns": 1, "rows": 1}, reason="one field",
        )
        assert result["accepted"] is True
        job = await _wait_for_job(gateway, result["job"]["job_id"])
        assert job.status.value == "succeeded"
        assert job.result["fields"] == 1


class TestSafetyGating:
    async def test_hazard_above_ceiling_needs_approval(self, gateway):
        # incubator-gas-needs-a-human forces approval regardless of autonomy.
        with pytest.raises(ApprovalRequired) as excinfo:
            await gateway.invoke(
                "incubator1", "GasControl", "set_co2", {"co2_pct": 5.0}, reason="test",
            )
        assert excinfo.value.detail["approval_id"]

    async def test_rate_limited_command_is_denied(self, gateway):
        await gateway.invoke("scope1", "MotionControl", "home", reason="setup")
        for _ in range(20):
            await gateway.invoke(
                "scope1", "Camera", "acquire_tile", {"columns": 1, "rows": 1}, reason="fill the window",
            )
        with pytest.raises(SafetyViolation):
            await gateway.invoke(
                "scope1", "Camera", "acquire_tile", {"columns": 1, "rows": 1}, reason="one too many",
            )

    async def test_site_tightened_z_limit(self, gateway):
        # protect-the-63x-objective caps FocusControl.move_z at 190 um, tighter
        # than the driver's own 0-300 um envelope.
        with pytest.raises(SafetyViolation):
            await gateway.invoke(
                "scope1", "FocusControl", "move_z", {"z_um": 250.0}, reason="test",
            )


class TestApprovalFlow:
    async def test_wait_for_approval_denied_raises_approval_denied(self, gateway):
        async def deny_soon():
            await asyncio.sleep(0.05)
            [pending] = gateway.approvals.pending()
            await gateway.approvals.deny(pending.id, approver="human:bob", reason="not today")

        denier = asyncio.ensure_future(deny_soon())
        with pytest.raises(ApprovalDenied):
            await gateway.invoke(
                "incubator1", "GasControl", "set_co2", {"co2_pct": 5.0},
                reason="test", wait_for_approval=2.0,
            )
        await denier

    async def test_granted_approval_replayed_succeeds(self, gateway):
        try:
            await gateway.invoke(
                "incubator1", "GasControl", "set_co2", {"co2_pct": 5.0}, reason="test",
            )
        except ApprovalRequired as exc:
            approval_id = exc.detail["approval_id"]
        await gateway.approvals.grant(approval_id, approver="human:alice", reason="ok")
        result = await gateway.invoke(
            "incubator1", "GasControl", "set_co2", {"co2_pct": 5.0},
            reason="test", approval_id=approval_id,
        )
        assert result["target_co2_pct"] == 5.0

    async def test_approval_is_bound_to_the_exact_call(self, gateway):
        try:
            await gateway.invoke(
                "incubator1", "GasControl", "set_co2", {"co2_pct": 5.0}, reason="test",
            )
        except ApprovalRequired as exc:
            approval_id = exc.detail["approval_id"]
        await gateway.approvals.grant(approval_id, approver="human:alice")
        with pytest.raises(ApprovalDenied):
            await gateway.invoke(
                "incubator1", "GasControl", "set_co2", {"co2_pct": 9.9},
                reason="test", approval_id=approval_id,
            )


class TestEstop:
    async def test_estop_cancels_running_jobs_and_stops_devices(self, gateway):
        started_home = await gateway.invoke("scope1", "MotionControl", "home", reason="setup")
        await _wait_for_job(gateway, started_home["job"]["job_id"])
        started = await gateway.invoke(
            "scope1", "Camera", "acquire_tile", {"columns": 5, "rows": 5}, reason="long scan",
        )
        job_id = started["job"]["job_id"]
        await asyncio.sleep(0.1)  # let the job actually start iterating fields
        result = await gateway.estop("something looked wrong")
        assert result["stopped"]["scope1"] == "stopped"
        job = await _wait_for_job(gateway, job_id)
        # A job already inside its work loop is cancelled cooperatively; the
        # device itself always ends up in FAULT either way, which is the
        # property that actually matters after an e-stop.
        assert job.status.value == "cancelled"
        assert gateway.device("scope1").state.value == "fault"

    async def test_estop_after_leaves_device_unhomed(self, gateway):
        # SimulatedMicroscope forfeits its datum on e-stop: position mid-stop
        # is not knowable, and claiming otherwise would be the unsafe answer.
        started_home = await gateway.invoke("scope1", "MotionControl", "home", reason="setup")
        await _wait_for_job(gateway, started_home["job"]["job_id"])
        await gateway.estop("test")
        assert gateway.device("scope1").homed is False

    async def test_estop_is_never_gated_even_at_manual_autonomy(self, gateway):
        gateway.safety.policy.autonomy = gateway.safety.policy.autonomy.__class__.MANUAL
        result = await gateway.estop("panic")
        assert result["all_stopped"] or result["failures"] == {}


class TestDescribe:
    async def test_describe_lists_every_configured_device(self, gateway):
        description = gateway.describe()
        ids = {d["id"] for d in description["devices"]}
        assert ids == {"scope1", "reader1", "handler1", "incubator1"}

    async def test_describe_reports_autonomy(self, gateway):
        description = gateway.describe()
        assert description["autonomy"]["name"] == "BOUNDED"
        assert description["autonomy"]["hazard_ceiling"] == "sample"


class TestGroundTruthStripping:
    async def test_truth_prefixed_keys_never_reach_the_caller(self, gateway):
        started_home = await gateway.invoke("scope1", "MotionControl", "home", reason="setup")
        await _wait_for_job(gateway, started_home["job"]["job_id"])
        # autofocus has no digital twin, so it needs a human signature at
        # this lab's autonomy level; grant it up front to reach the job.
        try:
            await gateway.invoke(
                "scope1", "FocusControl", "autofocus", {"steps": 3}, reason="focus",
            )
        except ApprovalRequired as exc:
            approval_id = exc.detail["approval_id"]
            await gateway.approvals.grant(approval_id, approver="human:alice", reason="ok")
            result = await gateway.invoke(
                "scope1", "FocusControl", "autofocus", {"steps": 3},
                reason="focus", approval_id=approval_id,
            )
        job = await _wait_for_job(gateway, result["job"]["job_id"])
        assert job.result is not None
        assert not any(k.startswith("truth_") for k in job.result)


class TestSnapshot:
    """`lab.snapshot`: every instrument's readable state in one call, instead
    of `device.read` once per instrument."""

    async def _call(self, gateway):
        from labbench.protocol.jsonrpc import Request
        from labbench.protocol.router import RpcContext

        response = await gateway.router.dispatch(
            Request("lab.snapshot", {}, id=1), RpcContext(actor="test"),
        )
        assert response is not None and response.error is None
        return response.result

    async def test_reports_every_configured_device(self, gateway):
        result = await self._call(gateway)
        assert set(result["devices"]) == {"scope1", "reader1", "handler1", "incubator1"}

    async def test_each_device_carries_its_properties_and_state(self, gateway):
        result = await self._call(gateway)
        scope = result["devices"]["scope1"]
        assert scope["state"] == "idle"
        assert "MotionControl.x_um" in scope["properties"]
        assert "FocusControl.focus_score" in scope["properties"]

    async def test_labware_state_is_visible_as_ordinary_properties(self, gateway):
        result = await self._call(gateway)
        # A plate's location and a slot's occupant are properties like any
        # other -- lab.snapshot needs no special labware-aware code path.
        assert "Labware.plates" in result["devices"]["handler1"]["properties"]
        assert "PlateStorage.contents" in result["devices"]["incubator1"]["properties"]
        assert "PlateTransport.loaded_barcode" in result["devices"]["reader1"]["properties"]

    async def test_reflects_a_change_made_since_the_last_snapshot(self, gateway):
        before = await self._call(gateway)
        assert before["devices"]["handler1"]["properties"]["Labware.plates"] == []

        await gateway.invoke(
            "handler1", "Labware", "create_plate", {"barcode": "SNAP1"}, actor="test",
        )
        after = await self._call(gateway)
        assert after["devices"]["handler1"]["properties"]["Labware.plates"] == ["SNAP1"]

    async def test_is_never_gated(self, gateway):
        # No autonomy level, rule or approval requirement should ever touch
        # a call that only reads.
        from labbench.core.safety import AutonomyLevel

        gateway.safety.policy.autonomy = AutonomyLevel.MANUAL
        result = await self._call(gateway)
        assert set(result["devices"]) == {"scope1", "reader1", "handler1", "incubator1"}
