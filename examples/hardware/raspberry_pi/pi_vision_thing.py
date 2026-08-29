#!/usr/bin/env python3
"""Runs ON the Raspberry Pi. Turns its camera and Coral Edge TPU into a W3C
Web of Things Thing that LabBench's existing `wot` driver talks to with zero
new LabBench code -- the same driver that already talks to any other
Thing-Description-publishing device (see src/labbench/drivers/http_wot.py).

Why WoT rather than a bespoke LabBench driver: the whole point of a
self-describing protocol is that adding a new instrument means writing one of
these, not touching the gateway. This script IS the instrument's side of that
contract, and nothing else has to know it exists until you point a lab
config at its URL.

Prerequisites (Raspberry Pi OS Bookworm, Pi 4 or 5, or any Linux x86_64/aarch64
box with a Coral USB Accelerator -- this script does not require picamera2 or
a Pi specifically for the Edge TPU half of it):

    sudo apt install -y python3-picamera2         # camera; Pi only
    pip install tflite-runtime                    # or: sudo apt install python3-tflite-runtime

    # The native Edge TPU runtime -- not a Python package, a shared library
    # the delegate loader dlopen()s. This is the one piece that needs root,
    # because it registers a udev rule granting non-root USB access:
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | sudo gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg
    echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    sudo apt-get update && sudo apt-get install -y libedgetpu1-std

    # A Coral USB Accelerator works on both Pi 4 and Pi 5 over USB3; plug it
    # in before starting this script. No PCIe/M.2 Coral is assumed.

    mkdir -p ~/labbench-pi/model && cd ~/labbench-pi/model
    curl -LO https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_quant_edgetpu.tflite
    curl -LO https://github.com/google-coral/test_data/raw/master/imagenet_labels.txt

Deliberately not `pycoral`: the package PyPI serves under that name is not
Google's real library -- it is three lines with none of the actual
`pycoral.utils`/`pycoral.adapters` code, discovered the hard way while
bringing this script up against a real device. The real pycoral is
apt-only (`python3-pycoral`, from the same repo as libedgetpu above), which
works fine on a Pi but means anyone testing off-Pi -- exactly the situation
of developing this script at all -- hits a silent dead end. `tflite-runtime`
plus `Interpreter.experimental_delegates=[load_delegate(...)]` is the whole
of what pycoral wrapped, and it is a real, current PyPI package.

Run it:

    python3 pi_vision_thing.py --model ~/labbench-pi/model/mobilenet_v2_1.0_224_quant_edgetpu.tflite \
        --labels ~/labbench-pi/model/imagenet_labels.txt

Then point a lab config at it -- see ../../../configs/raspberry-pi-vision-lab.yaml.
No camera or TPU is required just to try the shape of this: with neither
installed the script still serves cpu_temp_c/cpu_load_pct/uptime_s and reports
`tpu_present: false`; `snap`/`classify` fail with a clear message instead of
faking a frame, the same "a driver that cannot predict must say so" rule the
rest of this project holds simulated drivers to.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

START = time.time()

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None  # reported per-request, not fatal at import time

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from tensorflow.lite.python import interpreter as tflite  # a full TF install also works
    except ImportError:
        tflite = None

#: dlopen()'d by the delegate loader; Linux-only, matching this project's
#: "the wire protocol is written for real servers, not every OS" scope.
_EDGETPU_SHARED_LIB = "libedgetpu.so.1"


def _make_interpreter(model_path: str) -> Any:
    """The whole of what `pycoral.utils.edgetpu.make_interpreter` did: load
    the Edge TPU delegate, hand it to a plain tflite Interpreter. Raises
    ValueError/OSError exactly when there is no library or no device to
    claim -- the caller turns that into "no TPU", not a crash."""
    delegate = tflite.load_delegate(_EDGETPU_SHARED_LIB)
    interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[delegate])
    interpreter.allocate_tensors()
    return interpreter


def _input_size(interpreter: Any) -> tuple[int, int]:
    _, height, width, _ = interpreter.get_input_details()[0]["shape"]
    return int(width), int(height)


def _set_input(interpreter: Any, image: np.ndarray) -> None:
    """`image` must already be resized to `_input_size` and in RGB order."""
    index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(index, np.expand_dims(image, axis=0))


def _top_k_classes(interpreter: Any, top_k: int) -> list[tuple[int, float]]:
    """(class_id, score) pairs, dequantized per the output tensor's own
    scale/zero-point -- the quantization the model's own converter chose,
    not a guess this script makes up."""
    detail = interpreter.get_output_details()[0]
    raw = interpreter.get_tensor(detail["index"])[0]
    scale, zero_point = detail["quantization"]
    scores = raw.astype(float) if scale == 0 else scale * (raw.astype(np.int32) - zero_point)
    order = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in order]


def read_label_file(path: str) -> dict[int, str]:
    """Coral's label files are "<index> <name>" per line."""
    labels: dict[int, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            index_str, _, name = line.partition(" ")
            labels[int(index_str)] = name.strip()
    return labels


class VisionStation:
    """Owns the camera and the interpreter; the HTTP handler just calls this."""

    def __init__(self, *, model_path: str | None, labels_path: str | None, image_dir: Path) -> None:
        self.image_dir = image_dir
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self._frame_count = 0
        self._lock = threading.Lock()  # one camera; one caller at a time

        self.camera = None
        self.resolution = (1920, 1080)
        if Picamera2 is not None:
            self.camera = Picamera2()
            config = self.camera.create_still_configuration(
                main={"size": self.resolution, "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(1.0)  # let auto-exposure/white-balance settle before the first real frame

        self.interpreter = None
        self.labels: dict[int, str] = {}
        if model_path and tflite is not None:
            try:
                self.interpreter = _make_interpreter(model_path)
                if labels_path:
                    self.labels = read_label_file(labels_path)
            except (ValueError, OSError) as exc:
                # No libedgetpu.so.1 to dlopen, or no device for it to claim --
                # a real and expected failure mode, not a bug. Reported once,
                # here, rather than on every classify() call.
                print(f"warning: could not initialise the Edge TPU ({exc}); tpu_present will be false")

    @property
    def tpu_present(self) -> bool:
        return self.interpreter is not None

    # -- telemetry ----------------------------------------------------------

    def cpu_temp_c(self) -> float:
        try:
            raw = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
            return round(int(raw) / 1000.0, 1)
        except (FileNotFoundError, ValueError):
            return -1.0  # not a Pi, or the thermal zone moved; -1 reads as "unknown", not a lie

    def cpu_load_pct(self) -> float:
        one_min, _, _ = os.getloadavg()
        return round(one_min / (os.cpu_count() or 1) * 100.0, 1)

    def uptime_s(self) -> int:
        return int(time.time() - START)

    # -- actions --------------------------------------------------------------

    def capture(self) -> tuple[Any, Path]:
        """One frame, as a numpy array for inference and a JPEG on disk for
        the artifact URI. Held under a lock: one physical camera, one frame
        in flight at a time, the same discipline `PlateStore.hold` applies to
        a physical plate two instruments could otherwise race on."""
        if self.camera is None:
            raise RuntimeError(
                "no camera: picamera2 is not installed, or none was detected at startup"
            )
        with self._lock:
            array = self.camera.capture_array()
            self._frame_count += 1
            path = self.image_dir / f"frame_{self._frame_count:04d}.jpg"
            self.camera.capture_file(str(path))
        return array, path

    def snap(self) -> dict[str, Any]:
        array, path = self.capture()
        height, width = array.shape[:2]
        return {
            "artifact_uri": f"/artifacts/{path.name}",  # resolved to an absolute URL by the handler
            "width": width, "height": height,
            "mean_brightness": round(float(array.mean()), 2),
        }

    def classify(self, top_k: int = 3) -> dict[str, Any]:
        if self.interpreter is None:
            raise RuntimeError(
                "no Edge TPU interpreter: tflite-runtime is not installed, libedgetpu.so.1 "
                "is not installed, no Coral device was detected at startup, or --model was "
                "not given"
            )
        array, path = self.capture()
        size = _input_size(self.interpreter)
        try:
            from PIL import Image

            resized = np.asarray(Image.fromarray(array).resize(size))
        except ImportError:
            # Nearest-neighbour fallback with no Pillow dependency -- coarser,
            # but this script should not hard-require a package the Edge TPU
            # delegate itself does not.
            y_idx = (np.arange(size[1]) * array.shape[0] / size[1]).astype(int)
            x_idx = (np.arange(size[0]) * array.shape[1] / size[0]).astype(int)
            resized = array[y_idx][:, x_idx]
        _set_input(self.interpreter, resized)

        started = time.perf_counter()
        self.interpreter.invoke()
        inference_ms = (time.perf_counter() - started) * 1000.0

        predictions = [
            {"label": self.labels.get(class_id, str(class_id)), "score": round(score, 4)}
            for class_id, score in _top_k_classes(self.interpreter, top_k)
        ]
        return {
            "predictions": predictions,
            "artifact_uri": f"/artifacts/{path.name}",
            "inference_ms": round(inference_ms, 2),
        }


def thing_description(station: VisionStation, base_url: str) -> dict[str, Any]:
    return {
        "title": "pi-vision-station",
        "manufacturer": "Raspberry Pi Foundation / Google Coral",
        "description": "A Raspberry Pi with a camera and a Coral Edge TPU: capture a frame, "
                        "classify it on-device, report system telemetry.",
        "base": base_url,
        "properties": {
            "cpu_temp_c": {
                "type": "number", "description": "SoC temperature.", "unit": "degC",
                "readOnly": True,
                "forms": [{"href": "/properties/cpu_temp_c", "op": "readproperty"}],
            },
            "cpu_load_pct": {
                "type": "number",
                "description": "1-minute load average, as a percentage of one core.",
                "unit": "%", "readOnly": True,
                "forms": [{"href": "/properties/cpu_load_pct", "op": "readproperty"}],
            },
            "uptime_s": {
                "type": "integer", "description": "Seconds since this service started.",
                "unit": "s", "readOnly": True,
                "forms": [{"href": "/properties/uptime_s", "op": "readproperty"}],
            },
            "camera_resolution": {
                "type": "string", "description": "Capture resolution, WxH.", "readOnly": True,
                "forms": [{"href": "/properties/camera_resolution", "op": "readproperty"}],
            },
            "tpu_present": {
                "type": "boolean",
                "description": "Whether a Coral Edge TPU was detected at startup.",
                "readOnly": True,
                "forms": [{"href": "/properties/tpu_present", "op": "readproperty"}],
            },
        },
        "actions": {
            "snap": {
                "description": "Capture one still frame from the camera.",
                "output": {
                    "type": "object",
                    "properties": {
                        "artifact_uri": {"type": "string"}, "width": {"type": "integer"},
                        "height": {"type": "integer"}, "mean_brightness": {"type": "number"},
                    },
                },
                "forms": [{"href": "/actions/snap", "op": "invokeaction", "htv:methodName": "POST"}],
            },
            "classify": {
                "description": "Capture a frame and classify it on the Edge TPU.",
                "input": {
                    "type": "object",
                    "properties": {"top_k": {"type": "integer", "minimum": 1, "maximum": 10}},
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "predictions": {"type": "array"}, "artifact_uri": {"type": "string"},
                        "inference_ms": {"type": "number"},
                    },
                },
                "forms": [{"href": "/actions/classify", "op": "invokeaction",
                           "htv:methodName": "POST"}],
            },
        },
    }


def make_handler(station: VisionStation, base_url: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            pass  # the provenance ledger is the record that matters; stderr noise is not

        def _send_json(self, code: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, code: int, message: str) -> None:
            self._send_json(code, {"error": message})

        def _send_jpeg(self, path: Path) -> None:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            try:
                if self.path == "/.well-known/wot-thing-description":
                    self._send_json(200, thing_description(station, base_url))
                elif self.path == "/properties/cpu_temp_c":
                    self._send_json(200, station.cpu_temp_c())
                elif self.path == "/properties/cpu_load_pct":
                    self._send_json(200, station.cpu_load_pct())
                elif self.path == "/properties/uptime_s":
                    self._send_json(200, station.uptime_s())
                elif self.path == "/properties/camera_resolution":
                    w, h = station.resolution
                    self._send_json(200, f"{w}x{h}")
                elif self.path == "/properties/tpu_present":
                    self._send_json(200, station.tpu_present)
                elif self.path.startswith("/artifacts/"):
                    name = self.path.removeprefix("/artifacts/")
                    path = station.image_dir / name
                    if ".." in name or not path.is_file():
                        self._send_error(404, "no such artifact")
                    else:
                        self._send_jpeg(path)
                else:
                    self._send_error(404, "no such route")
            except Exception as exc:  # noqa: BLE001 - report to the caller, keep the server up
                self._send_error(500, str(exc))

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            args = json.loads(self.rfile.read(length)) if length else {}
            try:
                if self.path == "/actions/snap":
                    self._send_json(200, station.snap())
                elif self.path == "/actions/classify":
                    self._send_json(200, station.classify(top_k=int(args.get("top_k", 3))))
                else:
                    self._send_error(404, "no such route")
            except RuntimeError as exc:
                # A missing camera/TPU is a real, expected failure mode, not a
                # bug -- reported as such rather than a 500 with a traceback.
                self._send_error(503, str(exc))
            except Exception as exc:  # noqa: BLE001
                self._send_error(500, str(exc))

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", default=None, help="Edge-TPU-compiled .tflite model")
    parser.add_argument("--labels", default=None, help="Coral-format label file")
    parser.add_argument("--image-dir", default="~/labbench-pi/images")
    args = parser.parse_args()

    if Picamera2 is None:
        print("warning: picamera2 not importable; snap/classify will report 'no camera'")
    if args.model and tflite is None:
        print("warning: tflite-runtime not importable; classify will report 'no Edge TPU interpreter'")

    station = VisionStation(
        model_path=args.model, labels_path=args.labels,
        image_dir=Path(args.image_dir).expanduser(),
    )
    base_url = f"http://{args.host if args.host != '0.0.0.0' else _local_ip()}:{args.port}"
    server = ThreadingHTTPServer((args.host, args.port), make_handler(station, base_url))
    print(f"pi-vision-station serving on {base_url}")
    print(f"  Thing Description: {base_url}/.well-known/wot-thing-description")
    print(f"  camera: {'present' if station.camera is not None else 'NOT DETECTED'}")
    print(f"  Edge TPU: {'present' if station.tpu_present else 'NOT DETECTED'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def _local_ip() -> str:
    """Best-effort LAN IP, for the printed Thing Description URL only -- the
    server itself binds 0.0.0.0 regardless."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


if __name__ == "__main__":
    main()
