"""MicroManager driver, against the real MMCore demo device adapters.

Skipped rather than mocked when the adapters are not present: `pymmcore-plus`
being importable does not mean `MMConfig_demo.cfg` can load, since the actual
device adapter binaries are a separate download (`mmcore install`). This is
the same "no hardware, no test" honesty the rest of the suite applies to
optional protocols -- a mocked MMCore would test this driver against a fiction
of the API rather than the API itself.
"""

from __future__ import annotations

import pytest

pymmcore_plus = pytest.importorskip("pymmcore_plus")


def _demo_config_available() -> bool:
    try:
        core = pymmcore_plus.CMMCorePlus()
        core.loadSystemConfiguration()
        core.unloadAllDevices()
        return True
    except Exception:  # noqa: BLE001 - any failure means "skip this test"
        return False


pytestmark = pytest.mark.skipif(
    not _demo_config_available(),
    reason="Micro-Manager demo device adapters not installed; run `mmcore install`",
)

from labbench.core.device import DeviceDescriptor, ExecutionContext
from labbench.drivers.micromanager import MicroManagerMicroscope


@pytest.fixture
async def scope(tmp_path):
    dev = MicroManagerMicroscope(DeviceDescriptor(id="mm1"), data_dir=str(tmp_path))
    await dev.connect()
    await dev.initialize()
    try:
        yield dev
    finally:
        await dev.disconnect()


class TestCapabilityModel:
    async def test_expected_features_present(self, scope):
        features = scope.features()
        assert {"MotionControl", "FocusControl", "Camera"}.issubset(features)

    async def test_channel_group_discovered_from_the_config(self, scope):
        assert "ChannelControl" in scope.features()
        channel_prop = scope.features()["ChannelControl"].property("channel")
        assert "DAPI" in channel_prop.schema_.constraint.enum


class TestMotion:
    async def test_home_then_move(self, scope):
        assert (await scope.read("MotionControl", "homed")).value is True
        result = await scope.invoke(
            "MotionControl", "move_absolute", {"x_um": 500.0, "y_um": 300.0}, ExecutionContext(),
        )
        assert result["x_um"] == pytest.approx(500.0, abs=0.1)

    async def test_move_relative(self, scope):
        ctx = ExecutionContext()
        await scope.invoke("MotionControl", "move_absolute", {"x_um": 0.0, "y_um": 0.0}, ctx)
        result = await scope.invoke("MotionControl", "move_relative", {"dx_um": 10.0, "dy_um": -10.0}, ctx)
        assert result["x_um"] == pytest.approx(10.0, abs=0.1)
        assert result["y_um"] == pytest.approx(-10.0, abs=0.1)


class TestChannelAndCamera:
    async def test_set_channel_round_trip(self, scope):
        ctx = ExecutionContext()
        await scope.invoke("ChannelControl", "set_channel", {"channel": "FITC"}, ctx)
        assert (await scope.read("ChannelControl", "channel")).value == "FITC"

    async def test_snap_writes_a_real_png(self, scope):
        await scope.write("Camera", "exposure_ms", 20.0)
        result = await scope.invoke("Camera", "snap", {}, ExecutionContext())
        path = result["artifact_uri"].replace("file://", "")
        import os

        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


class TestSimulate:
    async def test_kinematic_fidelity_for_motion(self, scope):
        sim = await scope.simulate("MotionControl", "move_absolute", {"x_um": 1.0, "y_um": 1.0})
        assert sim.fidelity == "kinematic"

    async def test_no_twin_for_snap(self, scope):
        sim = await scope.simulate("Camera", "snap", {})
        assert sim.fidelity == "none"
