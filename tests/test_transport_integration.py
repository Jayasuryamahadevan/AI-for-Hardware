"""HTTP and WebSocket, end to end: real sockets, the real client, the real server.

Every other test drives the `Router` directly; these drive it the way an
actual remote agent would -- through `protocol.client.connect()` against a
`protocol.http.HttpServer` with the WebSocket upgrade wired in, on a real
loopback socket.
"""

from __future__ import annotations

import asyncio

import pytest

from labbench.protocol.client import HttpClient, WebSocketClient
from labbench.protocol.http import HttpServer
from labbench.protocol.router import Router, RpcContext
from labbench.protocol.websocket import WebSocketEndpoint


def make_router() -> Router:
    router = Router()

    @router.method("echo")
    async def echo(value: int) -> dict:
        return {"value": value}

    @router.method("progressive")
    async def progressive(ctx: RpcContext) -> dict:
        await ctx.progress(0.5, "halfway")
        return {"done": True}

    return router


@pytest.fixture
async def server():
    router = make_router()
    http_server = HttpServer(router, host="127.0.0.1", port=0)
    # port=0 would need OS-assigned-port introspection HttpServer doesn't
    # expose, so a fixed high port is used instead, same trade-off as the
    # OPC UA LADS test fixture.
    http_server.port = 18765
    endpoint = WebSocketEndpoint(router)
    http_server.set_upgrade_handler(endpoint.handle_upgrade)
    http_server.ws_endpoint = endpoint  # test convenience only
    await http_server.start()
    try:
        yield http_server
    finally:
        await http_server.close()


class TestHttp:
    async def test_rpc_call(self, server):
        client = HttpClient(f"http://127.0.0.1:{server.port}")
        result = await client.call("echo", value=5)
        assert result == {"value": 5}

    async def test_notifications_ride_back_with_the_reply(self, server):
        events = []
        client = HttpClient(
            f"http://127.0.0.1:{server.port}",
            on_notification=lambda m, p: events.append((m, p)),
        )
        result = await client.call("progressive")
        assert result == {"done": True}
        assert events == [("progress", {"fraction": 0.5, "message": "halfway"})]

    async def test_healthz_needs_no_auth(self, server):
        import urllib.request

        def fetch() -> int:
            # A synchronous urlopen() call would block this test's own event
            # loop -- the same loop the server's accept coroutine needs to run
            # on to answer it -- so it is pushed to a worker thread instead.
            with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/healthz") as resp:
                return resp.status

        status = await asyncio.to_thread(fetch)
        assert status == 200

    async def test_unauthorised_without_a_token(self):
        router = make_router()
        http_server = HttpServer(router, host="127.0.0.1", port=18766, token="secret")
        await http_server.start()
        try:
            client = HttpClient(f"http://127.0.0.1:{http_server.port}")
            from labbench.protocol.client import ClientError

            with pytest.raises(ClientError, match="unauthorised"):
                await client.call("echo", value=1)
        finally:
            await http_server.close()

    async def test_authorised_with_the_right_token(self):
        router = make_router()
        http_server = HttpServer(router, host="127.0.0.1", port=18767, token="secret")
        await http_server.start()
        try:
            client = HttpClient(f"http://127.0.0.1:{http_server.port}", token="secret")
            result = await client.call("echo", value=9)
            assert result == {"value": 9}
        finally:
            await http_server.close()


class TestWebSocket:
    async def test_rpc_call_over_ws(self, server):
        client = await WebSocketClient(f"ws://127.0.0.1:{server.port}/ws").connect()
        try:
            result = await client.call("echo", value=7)
            assert result == {"value": 7}
        finally:
            await client.close()

    async def test_notifications_arrive_live_while_the_call_is_open(self, server):
        events = []
        client = await WebSocketClient(
            f"ws://127.0.0.1:{server.port}/ws",
            on_notification=lambda m, p: events.append((m, p)),
        ).connect()
        try:
            result = await client.call("progressive")
            assert result == {"done": True}
            progress_events = [e for e in events if e[0] == "progress"]
            assert progress_events, events
        finally:
            await client.close()

    async def test_broadcast_reaches_a_connected_client(self, server):
        events = asyncio.Queue()
        client = await WebSocketClient(
            f"ws://127.0.0.1:{server.port}/ws",
            on_notification=lambda m, p: events.put_nowait((m, p)),
        ).connect()
        try:
            await asyncio.sleep(0.05)  # let the upgrade complete server-side
            await server.ws_endpoint.broadcast("device.event", {"hello": "world"})
            # The connection's own "ready" notification arrives first.
            method, payload = await asyncio.wait_for(events.get(), timeout=2)
            while method != "device.event":
                method, payload = await asyncio.wait_for(events.get(), timeout=2)
            assert payload == {"hello": "world"}
        finally:
            await client.close()
