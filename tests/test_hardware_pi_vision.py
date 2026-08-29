"""The Raspberry Pi vision station example, proven against the real driver.

`examples/hardware/raspberry_pi/pi_vision_thing.py` is the first example in
this project that talks to real (if here, absent) hardware rather than the
simulator. This test loads that script's actual module -- not a hand-copied
description of its API, which could silently drift from what the script
really does -- serves it from a real socket, and drives it through the real
`labbench.drivers.http_wot.WoTThing` driver and a real `Gateway`: ledger,
safety kernel, and all. A fake camera and a fake Edge TPU interpreter stand
in for hardware this suite cannot depend on; everything downstream of that is
the genuine code path, the same standard `tests/test_driver_wot.py` and
`tests/test_examples.py` hold every other transport-facing script to.
"""

from __future__ import annotations

import importlib.util
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

httpx = pytest.importorskip("httpx")

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "examples" / "hardware" / "raspberry_pi" / "pi_vision_thing.py"


def _load_pi_module():
    spec = importlib.util.spec_from_file_location("pi_vision_thing", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pi_module():
    return _load_pi_module()


class _FakeCamera:
    """Returns a fixed grey frame; real enough to prove the JSON shapes,
    which is all this test is responsible for -- not real optics."""

    def capture_array(self) -> np.ndarray:
        return np.full((480, 640, 3), 100, dtype=np.uint8)

    def capture_file(self, path: str) -> None:
        Path(path).write_bytes(b"not a real jpeg, just needs to exist")


class _FakeClass:
    def __init__(self, id: int, score: float) -> None:
        self.id = id
        self.score = score


class _FakeCommonAdapter:
    @staticmethod
    def input_size(interpreter) -> tuple[int, int]:
        return (224, 224)

    @staticmethod
    def set_input(interpreter, image) -> None:
        pass


class _FakeClassifyAdapter:
    @staticmethod
    def get_classes(interpreter, top_k: int = 3) -> list[_FakeClass]:
        return [_FakeClass(1, 0.91), _FakeClass(2, 0.05), _FakeClass(3, 0.01)][:top_k]


class _FakeInterpreter:
    def invoke(self) -> None:
        pass


@pytest.fixture
def bare_station(pi_module, tmp_path):
    """A station with no camera and no TPU -- exactly what pi_vision_thing.py
    itself falls back to when picamera2/pycoral aren't installed, which is
    also true of whatever machine runs this test suite."""
    return pi_module.VisionStation(model_path=None, labels_path=None, image_dir=tmp_path)


@pytest.fixture
def equipped_station(pi_module, tmp_path):
    station = pi_module.VisionStation(model_path=None, labels_path=None, image_dir=tmp_path)
    station.camera = _FakeCamera()
    station.resolution = (640, 480)
    station.interpreter = _FakeInterpreter()
    station.labels = {1: "coffee_mug", 2: "cup", 3: "pitcher"}
    return station


@pytest.fixture
def server(pi_module, equipped_station):
    handler = pi_module.make_handler(equipped_station, "http://127.0.0.1:0")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()


class TestStationLogicWithoutHardware:
    """What actually happens today, on a machine with neither library
    installed -- the honest default this script promises in its own README."""

    def test_no_camera_detected(self, bare_station):
        assert bare_station.camera is None
        assert bare_station.tpu_present is False

    def test_snap_without_a_camera_raises_a_clear_error(self, bare_station):
        with pytest.raises(RuntimeError, match="no camera"):
            bare_station.snap()

    def test_classify_without_a_tpu_raises_a_clear_error(self, bare_station):
        with pytest.raises(RuntimeError, match="no Edge TPU"):
            bare_station.classify()

    def test_telemetry_works_with_no_hardware_at_all(self, bare_station):
        assert bare_station.uptime_s() >= 0
        assert 0.0 <= bare_station.cpu_load_pct()


class TestStationLogicWithFakeHardware:
    def test_snap_reports_real_frame_shape(self, equipped_station):
        result = equipped_station.snap()
        assert result["width"] == 640
        assert result["height"] == 480
        assert result["mean_brightness"] == pytest.approx(100.0, abs=0.5)
        assert result["artifact_uri"].endswith(".jpg")

    def test_classify_reports_labelled_predictions(self, pi_module, equipped_station, monkeypatch):
        monkeypatch.setattr(pi_module, "common", _FakeCommonAdapter)
        monkeypatch.setattr(pi_module, "classify", _FakeClassifyAdapter)
        result = equipped_station.classify(top_k=2)
        assert result["predictions"] == [
            {"label": "coffee_mug", "score": 0.91}, {"label": "cup", "score": 0.05},
        ]
        assert result["inference_ms"] >= 0.0


class TestThroughTheRealWotDriverAndGateway:
    """The point of the exercise: everything above, reached the way LabBench
    actually reaches it -- device.describe, device.read, device.invoke,
    through a real Gateway, with the ledger and safety kernel in the loop."""

    @pytest.fixture
    async def device(self, server):
        from labbench.core.device import DeviceDescriptor
        from labbench.drivers.http_wot import WoTThing

        device = WoTThing(
            DeviceDescriptor(id="pi1"),
            td_url=f"{server}/.well-known/wot-thing-description",
            profile_overrides={"actions": {
                "snap": {"hazard": "none", "reversibility": "reversible"},
                "classify": {"hazard": "none", "reversibility": "reversible"},
            }},
        )
        await device.connect()
        try:
            yield device
        finally:
            await device.disconnect()

    async def test_describe_lists_both_actions(self, device):
        features = device.features()
        commands = {c.name for c in features["Thing"].commands}
        assert commands == {"snap", "classify"}

    async def test_read_all_properties(self, device):
        properties = await device.read_all()
        assert properties["Thing.camera_resolution"] == "640x480"
        assert properties["Thing.tpu_present"] is True

    async def test_invoke_snap(self, device):
        from labbench.core.device import ExecutionContext

        result = await device.invoke("Thing", "snap", {}, ExecutionContext())
        assert result["width"] == 640
        assert result["mean_brightness"] > 0

    async def test_invoke_snap_through_the_real_gateway(self, server, data_dir):
        """The full path: Gateway.invoke -> ledger -> safety kernel -> the
        real driver -> the real (fake-camera-backed) HTTP handler."""
        from labbench.core.registry import DeviceConfig, LabConfig
        from labbench.gateway import Gateway

        config = LabConfig(
            name="pi-test-bench",
            devices=[DeviceConfig(
                id="pi1", driver="wot", settings={
                    "td_url": f"{server}/.well-known/wot-thing-description",
                    "profile_overrides": {"actions": {
                        "snap": {"hazard": "none", "reversibility": "reversible"},
                    }},
                },
            )],
        )
        gateway = Gateway(config, data_dir=data_dir)
        await gateway.start()
        try:
            result = await gateway.invoke("pi1", "Thing", "snap", reason="test")
            assert result["width"] == 640
            kinds = [r.kind for r in gateway.ledger.query(device_id="pi1")]
            assert "command_request" in kinds and "command_result" in kinds
        finally:
            await gateway.close()
