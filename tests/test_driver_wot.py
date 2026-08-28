"""WoT driver, against a real (if minimal) Thing serving a Thing Description.

`http.server.ThreadingHTTPServer` rather than a mock transport: httpx's
`MockTransport` would validate what this driver *sends*, but the point of a
Thing Description-driven client is the round trip -- fetch the TD, build a
capability model from it, then follow the `forms` it declared -- and a real
socket is the honest way to prove that loop actually closes.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

httpx = pytest.importorskip("httpx")

from labbench.core.device import DeviceDescriptor
from labbench.drivers.http_wot import WoTThing

STATE = {"brightness": 50}

TD = {
    "title": "TestLamp",
    "properties": {
        "brightness": {
            "type": "integer", "minimum": 0, "maximum": 100, "unit": "%",
            "forms": [{"href": "/brightness", "op": ["readproperty", "writeproperty"],
                       "htv:methodName": "PUT"}],
        },
    },
    "actions": {
        "toggle": {
            "description": "Toggle the lamp.",
            "output": {"type": "object", "properties": {"on": {"type": "boolean"}}},
            "forms": [{"href": "/toggle", "op": ["invokeaction"]}],
        },
    },
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/td":
            self._send(200, TD)
        elif self.path == "/brightness":
            self._send(200, STATE["brightness"])
        else:
            self._send(404, {"error": "not found"})

    def do_PUT(self):
        length = int(self.headers.get("Content-Length", 0))
        value = json.loads(self.rfile.read(length))
        if self.path == "/brightness":
            STATE["brightness"] = value
            self._send(200, STATE["brightness"])
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/toggle":
            self._send(200, {"on": True})
        else:
            self._send(404, {"error": "not found"})


@pytest.fixture(scope="module")
def thing_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()


@pytest.fixture
async def lamp(thing_url):
    dev = WoTThing(DeviceDescriptor(id="lamp1"), td_url=f"{thing_url}/td")
    await dev.connect()
    try:
        yield dev
    finally:
        await dev.disconnect()


class TestCapabilityModel:
    async def test_td_properties_become_labbench_properties(self, lamp):
        feature = lamp.features()["Thing"]
        prop = feature.property("brightness")
        assert prop.schema_.type == "integer"
        assert prop.schema_.unit == "%"

    async def test_td_actions_become_labbench_commands(self, lamp):
        feature = lamp.features()["Thing"]
        assert feature.command("toggle") is not None


class TestDataPlane:
    async def test_read_and_write_property(self, lamp):
        before = await lamp.read("Thing", "brightness")
        assert before.value == 50
        await lamp.write("Thing", "brightness", 77)
        after = await lamp.read("Thing", "brightness")
        assert after.value == 77

    async def test_invoke_action(self, lamp):
        result = await lamp.invoke("Thing", "toggle", {})
        assert result["on"] is True


class TestSimulate:
    async def test_no_digital_twin(self, lamp):
        sim = await lamp.simulate("Thing", "toggle", {})
        assert sim.fidelity == "none"


class TestConfiguration:
    def test_missing_td_source_is_rejected(self):
        from labbench.core.errors import ValidationError

        with pytest.raises(ValidationError):
            WoTThing(DeviceDescriptor(id="d"))

    async def test_inline_thing_description_needs_no_fetch(self):
        dev = WoTThing(DeviceDescriptor(id="d2"), thing_description=TD)
        await dev.connect()
        try:
            assert "toggle" in [c.name for c in dev.features()["Thing"].commands]
        finally:
            await dev.disconnect()
