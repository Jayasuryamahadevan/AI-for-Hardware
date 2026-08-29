# Raspberry Pi + Coral Edge TPU, as a real LabBench instrument

Everything else in `examples/` shows an AI model driving the *simulated* lab.
This is the first one that isn't simulated: a Raspberry Pi 4 or 5, a
Pi-compatible camera, and a Coral Edge TPU accelerator, reached the same way
any W3C Web of Things device is — `pi_vision_thing.py` is the instrument's
side of that contract, and no code in `src/labbench/` changed to support it.
The `wot` driver that talks to it already existed.

## What it exposes

One Thing, two actions and five read-only properties:

| Name | Kind | What it does |
|---|---|---|
| `snap` | action | Capture one still frame; returns its size, mean brightness, and a URI to fetch the JPEG |
| `classify` | action | Capture a frame and classify it on the Edge TPU; returns top-k labels and confidence scores |
| `cpu_temp_c`, `cpu_load_pct`, `uptime_s` | property | System telemetry |
| `camera_resolution`, `tpu_present` | property | What's actually attached |

Both actions are declared `hazard: none` in the shipped config — a still
capture and an on-device classification are genuinely non-actuating; nothing
about the physical world changes. That's an honest fact about *this*
instrument, not a simplifying assumption the way it is for the simulated
microscope, whose `Camera.snap` is deliberately `Hazard.SAMPLE` because
illumination really does bleach a specimen there.

## Setup

On the Raspberry Pi (Raspberry Pi OS Bookworm, Pi 4 or 5), or any Linux
x86_64/aarch64 machine with a Coral USB Accelerator attached — the Edge TPU
half of this script does not require a Pi:

```bash
sudo apt install -y python3-picamera2      # camera; Pi only
pip install tflite-runtime                 # NOT pycoral -- see below

# The native Edge TPU runtime is a shared library, not a Python package, and
# is the one piece that genuinely needs root (it installs a udev rule
# granting non-root USB access):
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg
echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update && sudo apt-get install -y libedgetpu1-std
# A Coral USB Accelerator works over USB3 on both Pi 4 and Pi 5; plug it in
# before starting the script. No PCIe/M.2 Coral module is assumed.

mkdir -p ~/labbench-pi/model && cd ~/labbench-pi/model
curl -LO https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_quant_edgetpu.tflite
curl -LO https://github.com/google-coral/test_data/raw/master/imagenet_labels.txt

python3 pi_vision_thing.py \
  --model ~/labbench-pi/model/mobilenet_v2_1.0_224_quant_edgetpu.tflite \
  --labels ~/labbench-pi/model/imagenet_labels.txt
```

**Not `pip install pycoral`.** The package PyPI serves under that name is not
Google's real library — three lines with none of the actual
`pycoral.utils`/`pycoral.adapters` code, found the hard way bringing this
script up against a real Coral device. The real pycoral is apt-only
(`python3-pycoral`, from the repo above), which is fine on a Pi but a silent
dead end anywhere else. This script needs neither: `tflite-runtime` plus
`Interpreter.experimental_delegates=[load_delegate(...)]` is the whole of
what pycoral wrapped.

It prints the URL to put in a lab config, e.g.:

```
pi-vision-station serving on http://192.168.1.50:8080
  Thing Description: http://192.168.1.50:8080/.well-known/wot-thing-description
  camera: present
  Edge TPU: present
```

**No camera at all — none attached, or a Pi camera that's physically
blocked?** Pass `--test-image` and every `snap`/`classify` serves a fixed
file for real instead of a live frame; the classic Coral demo image gives a
known-good answer to check against:

```bash
curl -LO https://github.com/google-coral/test_data/raw/master/parrot.jpg
python3 pi_vision_thing.py --test-image parrot.jpg \
  --model ~/labbench-pi/model/mobilenet_v2_1.0_224_quant_edgetpu.tflite \
  --labels ~/labbench-pi/model/imagenet_labels.txt
```

This is the permanent, real answer to "no camera" — `_StaticImageSource`
implements the exact same interface `Picamera2` does, so `VisionStation`
cannot tell the difference and every other code path (resize, inference,
artifact serving) runs for real. It is not a substitute for testing the
Edge TPU itself: `classify` still needs a real Coral device and
`libedgetpu1-std` regardless of where the pixels came from.

**Neither the camera nor the TPU is required just to try the shape of
this.** With picamera2/tflite-runtime missing, or no Coral device plugged
in, the script still serves telemetry and reports `tpu_present: false`;
`snap`/`classify` fail with a clear `503` explaining exactly what's missing
rather than fabricating a frame — the same "a driver that cannot predict
must say so" rule the rest of this project holds simulated drivers to.

## Point LabBench at it

Edit `configs/raspberry-pi-vision-lab.yaml`'s `td_url` to the address the
script printed, then, from a machine that can reach the Pi (the Pi itself,
or anywhere on the same network):

```bash
labbench devices -c configs/raspberry-pi-vision-lab.yaml
labbench call -c configs/raspberry-pi-vision-lab.yaml device.read device=pi1 feature=Thing
labbench call -c configs/raspberry-pi-vision-lab.yaml device.invoke \
  device=pi1 feature=Thing command=snap reason="check the camera is alive"
labbench call -c configs/raspberry-pi-vision-lab.yaml device.invoke \
  device=pi1 feature=Thing command=classify args='{"top_k": 3}' reason="what is in front of the camera"
```

Or serve the gateway and hand the tool schemas to a real agent loop exactly
as with the simulated lab:

```bash
labbench serve -c configs/raspberry-pi-vision-lab.yaml --transport ws &
python ../../agent_anthropic.py "Take a picture and tell me what's in front of the camera."
```

## Proof this actually talks to the real driver, not just in theory

`pi_vision_thing.py`'s HTTP contract (routes, Thing Description shape,
response bodies) is exercised end to end against the real
`labbench.drivers.http_wot.WoTThing` driver and the real gateway — ledger,
safety kernel, and all — in the same way `tests/test_driver_wot.py` proves
the driver against any WoT Thing: a real socket standing in for the Pi,
serving the identical contract this script does. That is what makes handing
you this script safe to do sight-unseen: the wire shape it must produce is
already pinned down and checked, not something either side is guessing at.
