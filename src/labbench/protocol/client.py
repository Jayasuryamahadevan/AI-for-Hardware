"""Client side of the connector.

Written for two reasons. It is what the test suite drives the server with, and
it is what an agent framework embeds when it wants LabBench in-process rather
than behind a subprocess. Keeping the client in the same package as the server
also keeps them honest: a protocol change that breaks one breaks the other in
the same commit.

Transport is chosen by URL scheme, and the call surface is identical across
all three:

    ws://host:port/ws      duplex; notifications arrive while calls are open
    http://host:port       request/response; notifications ride with the reply
    stdio://path/to/cmd    spawn the gateway as a subprocess
"""

from __future__ import annotations

import asyncio
import base64
import os
import shlex
import struct
import urllib.parse
from collections.abc import Awaitable, Callable
from typing import Any, Self

from .framing import Framing, encode, read_frame
from .jsonrpc import JsonRpcError, Request, serialise
from .websocket import (
    OP_BINARY,
    OP_CLOSE,
    OP_PING,
    OP_PONG,
    OP_TEXT,
    accept_key,
)

NotificationHandler = Callable[[str, dict[str, Any]], Awaitable[None] | None]


class ClientError(Exception):
    """Transport-level failure. A JSON-RPC error surfaces as JsonRpcError."""


class _Pending:
    """Correlates a reply with the call that is waiting for it."""

    __slots__ = ("future", "method")

    def __init__(self, method: str) -> None:
        self.future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self.method = method


class BaseClient:
    """Shared request-id allocation and reply correlation."""

    def __init__(self, *, on_notification: NotificationHandler | None = None) -> None:
        self._next_id = 0
        self._pending: dict[int, _Pending] = {}
        self.on_notification = on_notification
        self.notifications: list[dict[str, Any]] = []

    def _allocate(self, method: str) -> tuple[int, _Pending]:
        self._next_id += 1
        pending = _Pending(method)
        self._pending[self._next_id] = pending
        return self._next_id, pending

    async def _deliver(self, message: Any) -> None:
        """Route one decoded wire message to whoever is waiting for it."""
        if isinstance(message, list):
            for item in message:
                await self._deliver(item)
            return
        if not isinstance(message, dict):
            return
        if "method" in message and "id" not in message:
            method = message["method"]
            params = message.get("params") or {}
            self.notifications.append({"method": method, "params": params})
            if self.on_notification is not None:
                result = self.on_notification(method, params)
                if asyncio.iscoroutine(result):
                    await result
            return
        pending = self._pending.pop(message.get("id"), None)  # type: ignore[arg-type]
        if pending is None or pending.future.done():
            return
        if "error" in message:
            err = message["error"]
            pending.future.set_exception(
                JsonRpcError(err.get("code", 0), err.get("message", ""), err.get("data"))
            )
        else:
            pending.future.set_result(message.get("result"))

    def _fail_all(self, exc: BaseException) -> None:
        for pending in self._pending.values():
            if not pending.future.done():
                pending.future.set_exception(exc)
        self._pending.clear()


class WebSocketClient(BaseClient):
    """Duplex client. Notifications arrive while calls are still open."""

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        actor: str = "agent",
        on_notification: NotificationHandler | None = None,
    ) -> None:
        super().__init__(on_notification=on_notification)
        self.url = url
        self.token = token
        self.actor = actor
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._pump: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()

    async def connect(self) -> WebSocketClient:
        parts = urllib.parse.urlsplit(self.url)
        host = parts.hostname or "127.0.0.1"
        port = parts.port or 80
        path = parts.path or "/ws"
        reader, writer = await asyncio.open_connection(host, port)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        lines = [
            f"GET {path} HTTP/1.1",
            f"Host: {host}:{port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
            f"X-LabBench-Actor: {self.actor}",
        ]
        if self.token:
            lines.append(f"Authorization: Bearer {self.token}")
        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("ascii"))
        await writer.drain()

        status = (await reader.readline()).decode("latin-1").strip()
        headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if not line or line in (b"\r\n", b"\n"):
                break
            name, _, value = line.decode("latin-1").partition(":")
            headers[name.strip().lower()] = value.strip()
        if "101" not in status:
            raise ClientError(f"upgrade refused: {status}")
        expected = accept_key(key)
        if headers.get("sec-websocket-accept") != expected:
            # A proxy or cache answered instead of the server.
            raise ClientError("Sec-WebSocket-Accept did not match; the peer is not our server")

        self._reader, self._writer = reader, writer
        self._pump = asyncio.create_task(self._read_loop())
        return self

    async def __aenter__(self) -> Self:
        return await self.connect()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @staticmethod
    def _mask_frame(opcode: int, payload: bytes) -> bytes:
        """Client frames MUST be masked (RFC 6455 section 5.3)."""
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
        first = 0x80 | opcode
        length = len(payload)
        if length < 126:
            header = struct.pack("!BB", first, 0x80 | length)
        elif length < 65536:
            header = struct.pack("!BBH", first, 0x80 | 126, length)
        else:
            header = struct.pack("!BBQ", first, 0x80 | 127, length)
        return header + mask + masked

    async def _send(self, opcode: int, payload: bytes) -> None:
        if self._writer is None:
            raise ClientError("not connected")
        async with self._write_lock:
            self._writer.write(self._mask_frame(opcode, payload))
            await self._writer.drain()

    async def _read_loop(self) -> None:
        import json

        assert self._reader is not None
        buffer = bytearray()
        try:
            while True:
                header = await self._reader.readexactly(2)
                first, second = header[0], header[1]
                fin = bool(first & 0x80)
                opcode = first & 0x0F
                length = second & 0x7F
                if length == 126:
                    (length,) = struct.unpack("!H", await self._reader.readexactly(2))
                elif length == 127:
                    (length,) = struct.unpack("!Q", await self._reader.readexactly(8))
                mask = await self._reader.readexactly(4) if second & 0x80 else None
                payload = await self._reader.readexactly(length)
                if mask:
                    payload = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))

                if opcode == OP_CLOSE:
                    break
                if opcode == OP_PING:
                    await self._send(OP_PONG, payload)
                    continue
                if opcode == OP_PONG:
                    continue
                if opcode == OP_BINARY:
                    continue
                buffer += payload
                if not fin:
                    continue
                text, buffer = buffer.decode("utf-8"), bytearray()
                await self._deliver(json.loads(text))
        except (asyncio.IncompleteReadError, ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self._fail_all(ClientError("connection closed"))

    async def call(self, method: str, timeout: float | None = 60.0, **params: Any) -> Any:
        rid, pending = self._allocate(method)
        await self._send(
            OP_TEXT, serialise(Request(method, params, rid).to_dict()).encode("utf-8")
        )
        try:
            return await asyncio.wait_for(pending.future, timeout)
        except TimeoutError:
            self._pending.pop(rid, None)
            raise ClientError(f"{method} timed out after {timeout}s") from None

    async def notify(self, method: str, **params: Any) -> None:
        payload = Request(method, params, is_notification=True).to_dict()
        await self._send(OP_TEXT, serialise(payload).encode("utf-8"))

    async def close(self) -> None:
        if self._writer is not None:
            try:
                await self._send(OP_CLOSE, struct.pack("!H", 1000))
            except (ClientError, ConnectionResetError, BrokenPipeError):
                pass
        if self._pump is not None:
            self._pump.cancel()
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError, RuntimeError):
                pass
        self._reader = self._writer = None


class HttpClient(BaseClient):
    """Request/response client over the from-scratch HTTP transport.

    One connection per call, deliberately. Keep-alive would be faster, but this
    client's job is to be obviously correct and dependency-free; an agent that
    needs throughput uses the WebSocket transport, which is what it is for.
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:8765",
        *,
        token: str | None = None,
        actor: str = "agent",
        on_notification: NotificationHandler | None = None,
    ) -> None:
        super().__init__(on_notification=on_notification)
        parts = urllib.parse.urlsplit(url)
        self.host = parts.hostname or "127.0.0.1"
        self.port = parts.port or 80
        self.path = parts.path.rstrip("/") + "/rpc" if parts.path.rstrip("/") else "/rpc"
        self.token = token
        self.actor = actor

    async def call(self, method: str, timeout: float | None = 60.0, **params: Any) -> Any:
        import json

        rid, pending = self._allocate(method)
        body = serialise(Request(method, params, rid).to_dict()).encode("utf-8")
        headers = [
            f"POST {self.path} HTTP/1.1",
            f"Host: {self.host}:{self.port}",
            "Content-Type: application/json",
            f"Content-Length: {len(body)}",
            f"X-LabBench-Actor: {self.actor}",
            "Connection: close",
        ]
        if self.token:
            headers.append(f"Authorization: Bearer {self.token}")

        async def once() -> Any:
            reader, writer = await asyncio.open_connection(self.host, self.port)
            try:
                writer.write(("\r\n".join(headers) + "\r\n\r\n").encode("ascii") + body)
                await writer.drain()
                status_line = (await reader.readline()).decode("latin-1").strip()
                head: dict[str, str] = {}
                while True:
                    line = await reader.readline()
                    if not line or line in (b"\r\n", b"\n"):
                        break
                    name, _, value = line.decode("latin-1").partition(":")
                    head[name.strip().lower()] = value.strip()
                length = int(head.get("content-length", "0"))
                payload = await reader.readexactly(length) if length else b""
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except (ConnectionResetError, BrokenPipeError, RuntimeError):
                    pass

            code = int(status_line.split()[1]) if len(status_line.split()) > 1 else 0
            if code == 204:
                return None
            if code == 401:
                raise ClientError("unauthorised: the gateway requires a bearer token")
            if not payload:
                raise ClientError(f"empty reply ({status_line})")
            decoded = json.loads(payload)
            # HTTP attaches in-call notifications alongside the reply.
            if isinstance(decoded, dict) and "response" in decoded and "notifications" in decoded:
                for note in decoded["notifications"]:
                    self.notifications.append(note)
                    if self.on_notification is not None:
                        result = self.on_notification(note["method"], note.get("params") or {})
                        if asyncio.iscoroutine(result):
                            await result
                decoded = decoded["response"]
            await self._deliver(decoded)
            return await pending.future

        try:
            return await asyncio.wait_for(once(), timeout)
        except TimeoutError:
            self._pending.pop(rid, None)
            raise ClientError(f"{method} timed out after {timeout}s") from None

    async def close(self) -> None:  # symmetry with the other clients
        return None


class StdioClient(BaseClient):
    """Spawns the gateway as a subprocess and speaks to it over pipes."""

    def __init__(
        self,
        command: str | list[str],
        *,
        actor: str = "agent",
        on_notification: NotificationHandler | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        super().__init__(on_notification=on_notification)
        self.command = shlex.split(command) if isinstance(command, str) else list(command)
        self.actor = actor
        self.env = env
        self.process: asyncio.subprocess.Process | None = None
        self._pump: asyncio.Task[None] | None = None

    async def connect(self) -> StdioClient:
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,  # let the child's logs reach the operator's terminal
            env={**os.environ, **(self.env or {})},
        )
        self._pump = asyncio.create_task(self._read_loop())
        return self

    async def __aenter__(self) -> Self:
        return await self.connect()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _read_loop(self) -> None:
        import json

        assert self.process is not None and self.process.stdout is not None
        try:
            while True:
                line = await read_frame(self.process.stdout, Framing.LINE)
                if line is None:
                    break
                await self._deliver(json.loads(line))
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            self._fail_all(ClientError("gateway process exited"))

    async def call(self, method: str, timeout: float | None = 60.0, **params: Any) -> Any:
        if self.process is None or self.process.stdin is None:
            raise ClientError("not connected")
        rid, pending = self._allocate(method)
        self.process.stdin.write(
            encode(serialise(Request(method, params, rid).to_dict()), Framing.LINE)
        )
        await self.process.stdin.drain()
        try:
            return await asyncio.wait_for(pending.future, timeout)
        except TimeoutError:
            self._pending.pop(rid, None)
            raise ClientError(f"{method} timed out after {timeout}s") from None

    async def close(self) -> None:
        if self.process is None:
            return
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self._pump is not None:
            self._pump.cancel()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=5)
        except TimeoutError:
            self.process.kill()
        self.process = None


def connect(url: str, **kwargs: Any) -> WebSocketClient | HttpClient | StdioClient:
    """Build the right client for a URL. The call surface is identical."""
    if url.startswith(("ws://", "wss://")):
        return WebSocketClient(url, **kwargs)
    if url.startswith(("http://", "https://")):
        return HttpClient(url, **kwargs)
    if url.startswith("stdio://"):
        return StdioClient(url[len("stdio://"):], **kwargs)
    raise ValueError(f"unsupported URL scheme: {url!r}")
