"""WebSocket (RFC 6455), implemented over the from-scratch HTTP server.

The third transport, and the one for an agent that wants both directions at
once over a single socket: it issues calls while device events, job progress
and telemetry stream back, with no polling and no second connection.

SSE covers the read-only case and stdio covers the local case, so the reason
this exists is duplex over one connection through one firewall hole. That is
the shape a remote operator console wants, and the shape a long-running
autonomous agent wants when it must both watch a tile scan and steer it.

Scope: the server half of RFC 6455 for text frames -- handshake, framing,
fragmentation, control frames, and the mandatory unmasking of client payloads.
Not implemented: `permessage-deflate` (instrument JSON is small and already
repetitive enough that TCP handles it), and extensions generally.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import struct
import time
from typing import Any

from .auth import Identity
from .jsonrpc import INTERNAL_ERROR, JsonRpcError, Response, parse_message, serialise
from .router import Router, RpcContext

log = logging.getLogger("labbench.ws")

#: The RFC 6455 handshake constant. Fixed by the specification.
_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# Opcodes
OP_CONTINUATION = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA

#: Matches the JSON-RPC frame ceiling. A device reply that needs more than this
#: should be returning an artifact reference instead.
MAX_MESSAGE_BYTES = 16 * 1024 * 1024

# Close codes used here (RFC 6455 section 7.4.1)
CLOSE_NORMAL = 1000
CLOSE_GOING_AWAY = 1001
CLOSE_PROTOCOL_ERROR = 1002
CLOSE_UNSUPPORTED_DATA = 1003
CLOSE_INVALID_PAYLOAD = 1007
CLOSE_POLICY_VIOLATION = 1008
CLOSE_TOO_LARGE = 1009
CLOSE_INTERNAL_ERROR = 1011


def accept_key(client_key: str) -> str:
    """Derive `Sec-WebSocket-Accept` from the client's `Sec-WebSocket-Key`.

    The concatenate-with-GUID-then-SHA1 dance is not security; it exists so a
    cache or a naive proxy cannot accidentally complete a handshake it did not
    understand.
    """
    digest = hashlib.sha1((client_key + _GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


class ProtocolError(Exception):
    """The peer violated RFC 6455. The connection must be closed."""

    def __init__(self, message: str, code: int = CLOSE_PROTOCOL_ERROR) -> None:
        super().__init__(message)
        self.code = code


class WebSocketConnection:
    """One connected peer.

    Writes are serialised behind a lock. Without that, a broadcast fired from a
    background job could interleave its bytes with a reply and corrupt both.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        peer: str = "",
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.peer = peer
        self.id = os.urandom(6).hex()
        self.opened = time.time()
        self.closed = False
        self._write_lock = asyncio.Lock()
        self._pong = asyncio.Event()

    # -- framing ----------------------------------------------------------

    @staticmethod
    def _build_frame(opcode: int, payload: bytes, *, fin: bool = True) -> bytes:
        """Encode one frame. Server-to-client frames are never masked."""
        first = (0x80 if fin else 0x00) | opcode
        length = len(payload)
        if length < 126:
            header = struct.pack("!BB", first, length)
        elif length < 65536:
            header = struct.pack("!BBH", first, 126, length)
        else:
            header = struct.pack("!BBQ", first, 127, length)
        return header + payload

    async def _send_frame(self, opcode: int, payload: bytes) -> None:
        if self.closed:
            return
        async with self._write_lock:
            try:
                self.writer.write(self._build_frame(opcode, payload))
                await self.writer.drain()
            except (ConnectionResetError, BrokenPipeError, RuntimeError):
                self.closed = True

    async def send_text(self, text: str) -> None:
        await self._send_frame(OP_TEXT, text.encode("utf-8"))

    async def send_json(self, payload: Any) -> None:
        await self.send_text(serialise(payload))

    async def ping(self, data: bytes = b"") -> None:
        await self._send_frame(OP_PING, data)

    async def close(self, code: int = CLOSE_NORMAL, reason: str = "") -> None:
        if self.closed:
            return
        payload = struct.pack("!H", code) + reason.encode("utf-8")[:123]
        await self._send_frame(OP_CLOSE, payload)
        self.closed = True
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except (ConnectionResetError, BrokenPipeError, RuntimeError):
            pass

    async def _read_frame(self) -> tuple[int, bytes, bool]:
        """Read one frame: (opcode, payload, fin)."""
        header = await self.reader.readexactly(2)
        first, second = header[0], header[1]
        fin = bool(first & 0x80)
        if first & 0x70:
            # Reserved bits set with no negotiated extension.
            raise ProtocolError("reserved bits set but no extension was negotiated")
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F

        if length == 126:
            (length,) = struct.unpack("!H", await self.reader.readexactly(2))
        elif length == 127:
            (length,) = struct.unpack("!Q", await self.reader.readexactly(8))
            if length >> 63:
                raise ProtocolError("frame length has the high bit set")

        if opcode >= 0x8:
            # Control frames: never fragmented, payload capped at 125 bytes.
            if not fin:
                raise ProtocolError("control frames must not be fragmented")
            if length > 125:
                raise ProtocolError("control frame payload exceeds 125 bytes")
        if length > MAX_MESSAGE_BYTES:
            raise ProtocolError(f"frame of {length} bytes is too large", CLOSE_TOO_LARGE)
        if not masked:
            # The specification requires client frames to be masked, and a
            # server MUST fail the connection otherwise. This is not pedantry:
            # unmasked client traffic is what makes proxy cache-poisoning work.
            raise ProtocolError("client frames must be masked")

        mask = await self.reader.readexactly(4)
        payload = bytearray(await self.reader.readexactly(length))
        for i in range(length):
            payload[i] ^= mask[i & 3]
        return opcode, bytes(payload), fin

    async def receive(self) -> str | None:
        """Read one complete application message, reassembling fragments.

        Returns None when the peer closes. Control frames are handled inline
        and never surface to the caller.
        """
        buffer = bytearray()
        message_opcode: int | None = None
        while True:
            try:
                opcode, payload, fin = await self._read_frame()
            except (asyncio.IncompleteReadError, ConnectionResetError):
                self.closed = True
                return None

            if opcode == OP_CLOSE:
                code = struct.unpack("!H", payload[:2])[0] if len(payload) >= 2 else CLOSE_NORMAL
                log.debug("peer %s closed: %d", self.peer, code)
                await self.close(CLOSE_NORMAL)
                return None
            if opcode == OP_PING:
                await self._send_frame(OP_PONG, payload)
                continue
            if opcode == OP_PONG:
                self._pong.set()
                continue

            if opcode == OP_CONTINUATION:
                if message_opcode is None:
                    raise ProtocolError("continuation frame with nothing to continue")
            elif opcode in (OP_TEXT, OP_BINARY):
                if message_opcode is not None:
                    raise ProtocolError("new data frame began before the previous one finished")
                message_opcode = opcode
            else:
                raise ProtocolError(f"unknown opcode {opcode:#x}")

            buffer += payload
            if len(buffer) > MAX_MESSAGE_BYTES:
                raise ProtocolError("reassembled message is too large", CLOSE_TOO_LARGE)
            if not fin:
                continue

            if message_opcode == OP_BINARY:
                raise ProtocolError(
                    "binary frames are not accepted; this protocol is JSON text",
                    CLOSE_UNSUPPORTED_DATA,
                )
            try:
                return buffer.decode("utf-8")
            except UnicodeDecodeError:
                raise ProtocolError("text frame is not valid UTF-8", CLOSE_INVALID_PAYLOAD) from None


class WebSocketEndpoint:
    """Serves JSON-RPC over WebSocket, and fans events out to every peer."""

    def __init__(self, router: Router, *, heartbeat_s: float = 30.0) -> None:
        self.router = router
        self.heartbeat_s = heartbeat_s
        self.connections: set[WebSocketConnection] = set()

    async def broadcast(self, method: str, params: dict[str, Any]) -> int:
        """Push a notification to every connected peer."""
        payload = Response.notification(method, params).to_dict()
        sent = 0
        for conn in list(self.connections):
            if conn.closed:
                self.connections.discard(conn)
                continue
            await conn.send_json(payload)
            sent += 1
        return sent

    async def handle_upgrade(
        self,
        request: Any,  # protocol.http.HttpRequest; untyped to avoid a cycle
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        identity: Identity | None = None,
    ) -> None:
        """Complete the handshake, then serve the connection until it closes.

        `identity` is already verified by `HttpServer` before this is ever
        called -- see `protocol/auth.py`. It is not this method's job to
        authenticate the connection, only to trust the actor it was handed.
        """
        key = request.header("sec-websocket-key")
        version = request.header("sec-websocket-version")
        if not key:
            await _reject(writer, 400, "missing Sec-WebSocket-Key")
            return
        if version != "13":
            # Version 13 is the only one in the published RFC.
            await _reject(
                writer, 426, f"unsupported WebSocket version {version!r}; this server speaks 13",
                extra={"Sec-WebSocket-Version": "13"},
            )
            return

        handshake = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_key(key)}\r\n\r\n"
        )
        writer.write(handshake.encode("ascii"))
        await writer.drain()

        conn = WebSocketConnection(reader, writer, peer=request.peer)
        self.connections.add(conn)
        actor = identity.actor if identity is not None else request.header(
            "x-labbench-actor", "agent:websocket"
        )
        ctx = RpcContext(
            actor=actor,
            session_id=request.header("x-labbench-session", conn.id),
            transport="websocket",
            peer=request.peer,
            notify=lambda m, p: conn.send_json(Response.notification(m, p).to_dict()),
        )
        heartbeat = asyncio.create_task(self._heartbeat(conn))
        inflight: set[asyncio.Task[None]] = set()
        try:
            await conn.send_json(
                Response.notification(
                    "ready",
                    {"connection": conn.id, "transport": "websocket",
                     "methods": sorted(self.router.methods)},
                ).to_dict()
            )
            while not conn.closed:
                try:
                    message = await conn.receive()
                except ProtocolError as exc:
                    log.warning("protocol error from %s: %s", conn.peer, exc)
                    await conn.close(exc.code, str(exc)[:120])
                    break
                if message is None:
                    break
                # Each call is its own task, so a long acquisition does not
                # block the reads that would cancel it.
                task = asyncio.create_task(self._serve_message(conn, ctx, message))
                inflight.add(task)
                task.add_done_callback(inflight.discard)
        finally:
            heartbeat.cancel()
            for task in inflight:
                task.cancel()
            self.connections.discard(conn)
            await conn.close(CLOSE_GOING_AWAY)

    async def _serve_message(
        self, conn: WebSocketConnection, ctx: RpcContext, message: str
    ) -> None:
        try:
            requests, is_batch = parse_message(message)
        except JsonRpcError as exc:
            await conn.send_json(Response.fail(None, exc).to_dict())
            return
        try:
            if is_batch:
                replies = await self.router.dispatch_many(requests, ctx)
                if replies:
                    await conn.send_json([r.to_dict() for r in replies])
                return
            reply = await self.router.dispatch(requests[0], ctx)
            if reply is not None:
                await conn.send_json(reply.to_dict())
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover
            log.exception("websocket dispatch failed")
            await conn.send_json(
                Response.fail(
                    requests[0].id if not is_batch else None,
                    JsonRpcError(INTERNAL_ERROR, str(exc)),
                ).to_dict()
            )

    async def _heartbeat(self, conn: WebSocketConnection) -> None:
        """Ping periodically.

        A half-open TCP connection to an agent looks identical to an idle one,
        and the difference matters when the agent is meant to be supervising a
        running experiment.
        """
        try:
            while not conn.closed:
                await asyncio.sleep(self.heartbeat_s)
                await conn.ping(b"lb")
        except asyncio.CancelledError:
            pass


async def _reject(
    writer: asyncio.StreamWriter,
    status: int,
    message: str,
    extra: dict[str, str] | None = None,
) -> None:
    body = message.encode("utf-8")
    headers = {
        "Content-Type": "text/plain; charset=utf-8",
        "Content-Length": str(len(body)),
        "Connection": "close",
        **(extra or {}),
    }
    head = f"HTTP/1.1 {status} Upgrade Failed\r\n" + "".join(
        f"{k}: {v}\r\n" for k, v in headers.items()
    ) + "\r\n"
    try:
        writer.write(head.encode("ascii") + body)
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    except (ConnectionResetError, BrokenPipeError, RuntimeError):
        pass
