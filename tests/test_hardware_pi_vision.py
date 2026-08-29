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


class _FakeInterpreter:
    """A minimal stand-in for `tflite_runtime.interpreter.Interpreter`,
    shaped exactly like the real thing's public surface -- get/set_tensor,
    get_*_details, invoke -- since that surface, not pycoral, is what
    `_input_size`/`_set_input`/`_top_k_classes` actually call."""

    def __init__(self, output_raw: np.ndarray, quantization: tuple[float, float]) -> None:
        self._output_raw = output_raw
        self._quantization = quantization
        self.last_input: np.ndarray | None = None

    def get_input_details(self) -> list[dict]:
        return [{"shape": [1, 224, 224, 3], "index": 0}]

    def get_output_details(self) -> list[dict]:
        return [{"index": 1, "quantization": self._quantization}]

    def set_tensor(self, index: int, value: np.ndarray) -> None:
        self.last_input = value

    def invoke(self) -> None:
        pass

    def get_tensor(self, index: int) -> np.ndarray:
        return self._output_raw[None, :]  # add the batch dimension back


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
    # Raw quantized scores at indices [0, 1, 2]; dequantized via scale=1/255,
    # zero_point=0 they become ~[0.051, 0.910, 0.012] -- index 1 wins.
    station.interpreter = _FakeInterpreter(
        output_raw=np.array([13, 232, 3], dtype=np.uint8), quantization=(1 / 255, 0),
    )
    station.labels = {0: "background", 1: "coffee_mug", 2: "cup"}
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

    def test_classify_reports_labelled_predictions(self, equipped_station):
        result = equipped_station.classify(top_k=2)
        assert result["predictions"] == [
            {"label": "coffee_mug", "score": 0.9098}, {"label": "background", "score": 0.051},
        ]
        assert result["inference_ms"] >= 0.0
        # The interpreter actually received a resized, batched tensor -- not
        # just a mock that was never fed anything.
        assert equipped_station.interpreter.last_input.shape == (1, 224, 224, 3)


class TestStaticImageSource:
    """`--test-image`: the real, permanent answer to "no camera at all" --
    not a fake object standing in for one, the actual `_StaticImageSource`
    class a real VisionStation uses when given the flag."""

    @pytest.fixture
    def test_image(self, tmp_path) -> Path:
        from PIL import Image

        path = tmp_path / "fixed_frame.jpg"
        # A distinct, uniform colour so mean_brightness is exactly checkable
        # rather than merely "close to something plausible".
        Image.new("RGB", (300, 200), color=(60, 60, 60)).save(path)
        return path

    def test_station_uses_the_file_instead_of_a_camera(self, pi_module, tmp_path, test_image):
        station = pi_module.VisionStation(
            model_path=None, labels_path=None, image_dir=tmp_path, test_image=str(test_image),
        )
        assert isinstance(station.camera, pi_module._StaticImageSource)
        assert station.resolution == (300, 200)

    def test_snap_reports_the_fixed_frame_for_real(self, pi_module, tmp_path, test_image):
        station = pi_module.VisionStation(
            model_path=None, labels_path=None, image_dir=tmp_path, test_image=str(test_image),
        )
        result = station.snap()
        assert result["width"] == 300
        assert result["height"] == 200
        assert result["mean_brightness"] == pytest.approx(60.0, abs=0.5)

    def test_snap_is_reproducible_across_calls(self, pi_module, tmp_path, test_image):
        """Every call serves the same fixed frame -- the point of a static
        source is a known-good, repeatable answer, not a new random one."""
        station = pi_module.VisionStation(
            model_path=None, labels_path=None, image_dir=tmp_path, test_image=str(test_image),
        )
        first, second = station.snap(), station.snap()
        assert first["mean_brightness"] == second["mean_brightness"]


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
