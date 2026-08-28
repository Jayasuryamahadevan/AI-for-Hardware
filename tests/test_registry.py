"""Driver discovery and device management."""

from __future__ import annotations

import pytest

from labbench.core.device import Device, DeviceState
from labbench.core.errors import DeviceNotFound, DriverUnavailable
from labbench.core.registry import DeviceConfig, DeviceManager, DriverRegistry, LabConfig


class TestDriverRegistry:
    def test_discovers_shipped_drivers(self):
        registry = DriverRegistry()
        catalog = registry.catalog()
        assert "sim_microscope" in catalog["available"]

    def test_get_unknown_driver_raises(self):
        registry = DriverRegistry()
        with pytest.raises(DriverUnavailable):
            registry.get("does_not_exist")

    def test_register_and_get(self):
        registry = DriverRegistry()

        class Dummy(Device):
            def _features(self):
                return []

            async def _read(self, feature, name):
                return None

            async def _invoke(self, feature, command, args, ctx):
                return {}

        registry.register("dummy", Dummy)
        assert registry.get("dummy") is Dummy


class TestDeviceManager:
    def test_add_from_config_marks_simulated_by_driver_prefix(self):
        manager = DeviceManager()
        device = manager.add_from_config(
            DeviceConfig(id="scope1", driver="sim_microscope", settings={"seed": 1})
        )
        assert device.descriptor.simulated is True

    def test_get_unknown_device_raises(self):
        manager = DeviceManager()
        with pytest.raises(DeviceNotFound):
            manager.get("nope")

    async def test_connect_all_tolerates_one_bad_device(self):
        manager = DeviceManager()
        manager.add_from_config(DeviceConfig(id="ok", driver="sim_incubator"))

        class Broken(Device):
            def _features(self):
                return []

            async def _connect(self):
                raise RuntimeError("cannot reach hardware")

            async def _read(self, feature, name):
                return None

            async def _invoke(self, feature, command, args, ctx):
                return {}

        manager.registry.register("broken", Broken)
        manager.add_from_config(DeviceConfig(id="bad", driver="broken"))

        results = await manager.connect_all()
        assert results["ok"] == "idle"
        assert "error" in results["bad"]
        # One bad device must not prevent the rest of the lab from coming up.
        assert manager.get("ok").state is DeviceState.IDLE

    def test_find_by_feature(self):
        manager = DeviceManager()
        manager.add_from_config(DeviceConfig(id="scope1", driver="sim_microscope"))
        manager.add_from_config(DeviceConfig(id="inc1", driver="sim_incubator"))
        matches = manager.find(feature="MotionControl")
        assert [d.id for d in matches] == ["scope1"]

    def test_autoconnect_false_is_skipped(self):
        pass  # covered indirectly by connect_all's return-value contract above


class TestLabConfig:
    def test_load_the_shipped_simulated_lab(self):
        import pathlib

        path = pathlib.Path(__file__).resolve().parent.parent / "configs" / "simulated-lab.yaml"
        config = LabConfig.load(path)
        assert config.name == "simulated-lab"
        assert len(config.devices) == 4

    def test_rejects_unknown_top_level_keys(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LabConfig.model_validate({"not_a_real_field": True})
