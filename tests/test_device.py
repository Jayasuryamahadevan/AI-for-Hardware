"""Device lifecycle, state gating and precondition checking, via the ABC.

Uses `SimulatedMicroscope` as the exemplar `Device` implementation rather than
a hand-rolled fake, since it already exercises the interesting cases: a
homing precondition, exclusive-command locking, hazard-gated state
requirements and a real `simulate()`.
"""

from __future__ import annotations

import pytest

from labbench.core.device import DeviceDescriptor, DeviceState
from labbench.core.errors import ConstraintViolation, DeviceNotReady
from labbench.drivers.simulated.microscope import SimulatedMicroscope


@pytest.fixture
async def scope(tmp_path):
    dev = SimulatedMicroscope(
        DeviceDescriptor(id="scope1", simulated=True), seed=1, data_dir=str(tmp_path)
    )
    await dev.connect()
    try:
        yield dev
    finally:
        await dev.disconnect()


class TestLifecycle:
    async def test_connect_sets_idle(self, scope):
        assert scope.state is DeviceState.IDLE

    async def test_disconnect_sets_offline(self, scope):
        await scope.disconnect()
        assert scope.state is DeviceState.OFFLINE
        await scope.connect()  # restore for the fixture's own teardown

    async def test_fail_and_clear_fault(self, scope):
        await scope.fail("simulated hardware fault")
        assert scope.state is DeviceState.FAULT
        assert scope.fault == "simulated hardware fault"
        await scope.clear_fault()
        assert scope.state is DeviceState.IDLE
        assert scope.fault is None


class TestPreconditions:
    async def test_move_before_home_is_refused(self, scope):
        with pytest.raises(DeviceNotReady):
            await scope.invoke("MotionControl", "move_absolute", {"x_um": 100.0, "y_um": 100.0})

    async def test_move_after_home_succeeds(self, scope):
        await scope.invoke("MotionControl", "home", {})
        result = await scope.invoke("MotionControl", "move_absolute", {"x_um": 500.0, "y_um": 500.0})
        assert result["x_um"] == 500.0


class TestEnvelope:
    async def test_out_of_travel_move_raises(self, scope):
        await scope.invoke("MotionControl", "home", {})
        with pytest.raises(ConstraintViolation):
            await scope.invoke("MotionControl", "move_absolute", {"x_um": -5.0, "y_um": 0.0})

    async def test_argument_validation_before_hardware(self, scope):
        from labbench.core.errors import ValidationError

        with pytest.raises(ValidationError):
            await scope.invoke("MotionControl", "move_absolute", {"x_um": "not a number", "y_um": 1.0})


class TestSimulate:
    async def test_simulate_does_not_move_the_stage(self, scope):
        await scope.invoke("MotionControl", "home", {})
        before = (scope.x_um, scope.y_um)
        sim = await scope.simulate("MotionControl", "move_absolute", {"x_um": 999.0, "y_um": 999.0})
        assert sim.feasible
        assert (scope.x_um, scope.y_um) == before

    async def test_simulate_flags_out_of_envelope(self, scope):
        sim = await scope.simulate("MotionControl", "move_absolute", {"x_um": -100.0, "y_um": 0.0})
        assert not sim.feasible
        assert sim.violations

    async def test_dry_run_via_invoke_never_touches_hardware(self, scope):
        from labbench.core.device import ExecutionContext

        await scope.invoke("MotionControl", "home", {})
        before = (scope.x_um, scope.y_um)
        result = await scope.invoke(
            "MotionControl", "move_absolute", {"x_um": 42.0, "y_um": 42.0},
            ExecutionContext(dry_run=True),
        )
        assert result["dry_run"] is True
        assert (scope.x_um, scope.y_um) == before


class TestEstop:
    async def test_estop_forfeits_homing(self, scope):
        await scope.invoke("MotionControl", "home", {})
        assert scope.homed
        await scope.estop("test")
        assert not scope.homed
        assert scope.state is DeviceState.FAULT


class TestReadAll:
    async def test_read_all_flattens_by_feature(self, scope):
        snapshot = await scope.read_all()
        assert "MotionControl.x_um" in snapshot
        assert "Camera.exposure_ms" in snapshot

    async def test_read_all_scoped_to_one_feature(self, scope):
        snapshot = await scope.read_all("Camera")
        assert all(k.startswith("Camera.") for k in snapshot)
