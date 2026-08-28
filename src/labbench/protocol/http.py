"""An HTTP/1.1 server, written against asyncio and the standard library.

This is the transport for a *remote* agent, and the one that makes "any AI
model" true in practice: a model with nothing but an HTTP client and a JSON
parser can drive the bench. There is no SDK to install on the agent's side.

Implementing HTTP rather than importing a framework is a deliberate, bounded
cost. What a gateway needs is a small subset -- request line, headers, a body
delimited by Content-Length or chunked encoding, keep-alive, and one streaming
response type -- and that subset is stable in a way that web frameworks are
not. The alternative was making a laboratory's agent-facing contract depend on
a framework major version landing mid-experiment.

What is deliberately *not* implemented: TLS (terminate it at a reverse proxy,
which is where certificate rotation belongs), HTTP/2, and file uploads.
Artifacts are fetched, never posted.

Endpoints
---------
``POST /rpc``      JSON-RPC 2.0, single or batch.
``GET  /events``   Server-Sent Events: device events, job progress, telemetry.
``GET  /tools``    Tool schemas in a chosen AI dialect (``?dialect=openai``).
``GET  /healthz``  Liveness, unauthenticated.
``GET  /``         Human-readable index of what this gateway is.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
import urllib.parse
from collections.abc import Awaitable, Callable
from typing import Any

from .jsonrpc import INTERNAL_ERROR, JsonRpcError, Response, parse_message, serialise
from .router import Router, RpcContext

log = logging.getLogger("labbench.http")

MAX_HEADER_BYTES = 64 * 1024
MAX_BODY_BYTES = 8 * 1024 * 1024
#: Keep-alive idle timeout. Long enough that an agent thinking between calls
#: keeps its connection; short enough that a dead peer is reaped.
IDLE_TIMEOUT_S = 120.0

_STATUS_TEXT = {
    200: "OK", 202: "Accepted", 204: "No Content", 400: "Bad Request",
    401: "Unauthorized", 403: "Forbidden", 404: "Not Found",
    405: "Method Not Allowed", 408: "Request Timeout", 411: "Length Required",
    413: "Payload Too Large", 415: "Unsupported Media Type",
    426: "Upgrade Required", 431: "Request Header Fields Too Large",
    500: "Internal Server Error", 503: "Service Unavailable",
}


class HttpRequest:
    """One parsed request."""

    __slots__ = ("method", "target", "path", "query", "headers", "body", "version", "peer")

    def __init__(
        self,
        method: str,
        target: str,
        version: str,
        headers: dict[str, str],
        body: bytes,
        peer: str,
    ) -> None:
        self.method = method
        self.target = target
        self.version = version
        self.headers = headers
        self.body = body
        self.peer = peer
        parsed = urllib.parse.urlsplit(target)
        self.path = urllib.parse.unquote(parsed.path) or "/"
        self.query = {k: v[-1] for k, v in urllib.parse.parse_qs(parsed.query).items()}

    def header(self, name: str, default: str = "") -> str:
        return self.headers.get(name.lower(), default)

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    @property
    def wants_upgrade(self) -> bool:
        return (
            "upgrade" in self.header("connection").lower()
            and self.header("upgrade").lower() == "websocket"
        )


class HttpResponse:
    """One response to write back."""

    __slots__ = ("status", "body", "headers", "content_type")

    def __init__(
        self,
        status: int = 200,
        body: bytes | str = b"",
        content_type: str = "application/json",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self.body = body.encode("utf-8") if isinstance(body, str) else body
        self.content_type = content_type
        self.headers = headers or {}

    @classmethod
    def json(cls, payload: Any, status: int = 200) -> "HttpResponse":
        return cls(status, serialise(payload), "application/json")

    @classmethod
    def text(cls, body: str, status: int = 200) -> "HttpResponse":
        return cls(status, body, "text/plain; charset=utf-8")

    @classmethod
    def error(cls, status: int, message: str) -> "HttpResponse":
        return cls.json({"error": _STATUS_TEXT.get(status, "Error"), "message": message}, status)

    def render(self, *, keep_alive: bool) -> bytes:
        reason = _STATUS_TEXT.get(self.status, "Unknown")
        lines = [f"HTTP/1.1 {self.status} {reason}"]
        headers = {
            "Content-Type": self.content_type,
            "Content-Length": str(len(self.body)),
            "Connection": "keep-alive" if keep_alive else "close",
            # A gateway is not a cache, and a stale device state is worse than
            # no device state.
            "Cache-Control": "no-store",
            **self.headers,
        }
        lines += [f"{k}: {v}" for k, v in headers.items()]
        return ("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + self.body


Handler = Callable[[HttpRequest], Awaitable[HttpResponse]]


class EventStream:
    """One connected Server-Sent Events subscriber.

    SSE rather than WebSocket for the event feed because the feed is one-way
    and SSE survives proxies, reconnects on its own, and can be read with
    `curl`. A dashboard, a `tail -f`, and an agent's watchdog all work the same
    way. Bidirectional callers use the WebSocket transport instead.
    """

    def __init__(self, writer: asyncio.StreamWriter, *, topics: set[str] | None = None) -> None:
        self.writer = writer
        self.topics = topics
        self.id = secrets.token_hex(6)
        self.opened = time.time()
        self._alive = True

    def wants(self, topic: str) -> bool:
        return self.topics is None or topic in self.topics

    async def send(self, event: str, data: Any) -> bool:
        """Push one event. Returns False once the subscriber has gone away."""
        if not self._alive:
            return False
        try:
            payload = serialise(data)
            self.writer.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
            await self.writer.drain()
            return True
        except (ConnectionResetError, BrokenPipeError, RuntimeError):
            self._alive = False
            return False

    async def comment(self, text: str = "keepalive") -> bool:
        """An SSE comment. Proxies drop idle connections; this stops that."""
        if not self._alive:
            return False
        try:
            self.writer.write(f": {text}\n\n".encode("utf-8"))
            await self.writer.drain()
            return True
        except (ConnectionResetError, BrokenPipeError, RuntimeError):
            self._alive = False
            return False

    def close(self) -> None:
        self._alive = False


class HttpServer:
    """The gateway's HTTP surface.

    Authentication is a bearer token compared in constant time, or nothing at
    all when no token is configured. That second mode is not an oversight: a
    bench gateway bound to loopback with no token is the normal development
    case, and pretending otherwise leads people to hardcode a token instead.
    Binding to a non-loopback address without a token is refused outright.
    """

    def __init__(
        self,
        router: Router,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        token: str | None = None,
        allow_origin: str = "*",
        server_name: str = "labbench",
    ) -> None:
        self.router = router
        self.host = host
        self.port = port
        self.token = token
        self.allow_origin = allow_origin
        self.server_name = server_name
        self.routes: dict[tuple[str, str], Handler] = {}
        self.subscribers: set[EventStream] = set()
        self._server: asyncio.Server | None = None
        self._upgrade_handler: Callable[..., Awaitable[None]] | None = None
        self._register_builtin_routes()

    # -- registration -----------------------------------------------------

    def route(self, method: str, path: str) -> Callable[[Handler], Handler]:
        def decorator(fn: Handler) -> Handler:
            self.routes[(method.upper(), path)] = fn
            return fn

        return decorator

    def set_upgrade_handler(self, fn: Callable[..., Awaitable[None]]) -> None:
        """Install the WebSocket upgrade handler (see `protocol.websocket`)."""
        self._upgrade_handler = fn

    # -- event fan-out ----------------------------------------------------

    async def broadcast(self, event: str, data: Any) -> int:
        """Push an event to every interested subscriber; reap the dead ones."""
        dead = []
        sent = 0
        for sub in list(self.subscribers):
            if not sub.wants(event):
                continue
            if await sub.send(event, data):
                sent += 1
            else:
                dead.append(sub)
        for sub in dead:
            self.subscribers.discard(sub)
        return sent

    # -- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        loopback = self.host in ("127.0.0.1", "::1", "localhost")
        if not loopback and not self.token:
            raise RuntimeError(
                f"refusing to bind {self.host}:{self.port} without an auth token. "
                "A gateway reachable off-host controls physical hardware; set a token "
                "(--token / LABBENCH_TOKEN) or bind to 127.0.0.1."
            )
        self._server = await asyncio.start_server(self._on_connection, self.host, self.port)
        log.info(
            "listening on http://%s:%d (auth: %s)",
            self.host, self.port, "token" if self.token else "none - loopback only",
        )

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        for sub in list(self.subscribers):
            sub.close()
        self.subscribers.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}"

    # -- connection handling ----------------------------------------------

    async def _on_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer_info = writer.get_extra_info("peername")
        peer = f"{peer_info[0]}:{peer_info[1]}" if peer_info else "?"
        try:
            while True:
                try:
                    request = await asyncio.wait_for(
                        _read_request(reader, peer), timeout=IDLE_TIMEOUT_S
                    )
                except asyncio.TimeoutError:
                    return  # idle keep-alive expiry; not an error
                except _BadRequest as exc:
                    writer.write(HttpResponse.error(exc.status, exc.message).render(keep_alive=False))
                    await writer.drain()
                    return
                if request is None:
                    return

                if request.wants_upgrade and self._upgrade_handler is not None:
                    # Hand the socket over; it is no longer an HTTP connection.
                    await self._upgrade_handler(request, reader, writer)
                    return

                keep_alive = self._should_keep_alive(request)
                response = await self._handle(request, writer)
                if response is None:
                    return  # a streaming handler has taken over the socket
                for key, value in self._cors_headers(request).items():
                    response.headers.setdefault(key, value)
                response.headers.setdefault("Server", self.server_name)
                writer.write(response.render(keep_alive=keep_alive))
                await writer.drain()
                if not keep_alive:
                    return
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception:  # pragma: no cover - connection-level guard
            log.exception("connection from %s failed", peer)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError, RuntimeError):
                pass

    @staticmethod
    def _should_keep_alive(request: HttpRequest) -> bool:
        connection = request.header("connection").lower()
        if request.version == "HTTP/1.0":
            return "keep-alive" in connection
        return "close" not in connection

    def _cors_headers(self, request: HttpRequest) -> dict[str, str]:
        if not self.allow_origin:
            return {}
        return {
            "Access-Control-Allow-Origin": self.allow_origin,
            "Access-Control-Allow-Headers": "Authorization, Content-Type",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Vary": "Origin",
        }

    def _authorised(self, request: HttpRequest) -> bool:
        if not self.token:
            return True
        header = request.header("authorization")
        prefix = "bearer "
        if not header.lower().startswith(prefix):
            return False
        # Constant-time: a token check that leaks length or prefix by timing is
        # a token check an attacker can walk.
        return secrets.compare_digest(header[len(prefix):].strip(), self.token)

    async def _handle(
        self, request: HttpRequest, writer: asyncio.StreamWriter
    ) -> HttpResponse | None:
        if request.method == "OPTIONS":
            return HttpResponse(204, b"", "text/plain")
        handler = self.routes.get((request.method, request.path))
        if handler is None:
            if any(p == request.path for m, p in self.routes):
                allowed = sorted(m for m, p in self.routes if p == request.path)
                return HttpResponse.error(
                    405, f"{request.path} accepts {', '.join(allowed)}"
                )
            return HttpResponse.error(
                404, f"no route {request.method} {request.path}; "
                     f"try {', '.join(sorted({p for _, p in self.routes}))}"
            )
        if request.path != "/healthz" and not self._authorised(request):
            return HttpResponse(
                401,
                serialise({"error": "Unauthorized", "message": "bearer token required"}),
                headers={"WWW-Authenticate": 'Bearer realm="labbench"'},
            )
        # The SSE handler needs the raw socket, so it is dispatched specially.
        if request.path == "/events" and request.method == "GET":
            await self._serve_events(request, writer)
            return None
        return await handler(request)

    # -- built-in routes --------------------------------------------------

    def _register_builtin_routes(self) -> None:
        @self.route("POST", "/rpc")
        async def rpc(request: HttpRequest) -> HttpResponse:
            """JSON-RPC 2.0 over HTTP."""
            # Content-Type is deliberately not enforced. The point of this
            # transport is that a model with an HTTP client and nothing else
            # can drive the bench, and half of those clients send
            # form-urlencoded by default. The body either parses as JSON-RPC or
            # it does not, and parse_message says so precisely.
            try:
                requests, is_batch = parse_message(request.body)
            except JsonRpcError as exc:
                return HttpResponse.json(Response.fail(None, exc).to_dict(), 200)

            ctx = RpcContext(
                actor=request.header("x-labbench-actor", "agent:http"),
                session_id=request.header("x-labbench-session", ""),
                transport="http",
                peer=request.peer,
            )
            try:
                if is_batch:
                    replies = await self.router.dispatch_many(requests, ctx)
                    payload: Any = [r.to_dict() for r in replies]
                    if not replies:
                        return HttpResponse(204, b"", "application/json")
                else:
                    reply = await self.router.dispatch(requests[0], ctx)
                    if reply is None:
                        return HttpResponse(204, b"", "application/json")
                    payload = reply.to_dict()
            except Exception as exc:  # pragma: no cover
                log.exception("rpc dispatch failed")
                return HttpResponse.json(
                    Response.fail(None, JsonRpcError(INTERNAL_ERROR, str(exc))).to_dict()
                )
            # HTTP has no out-of-band channel, so progress emitted during the
            # call rides back attached to the reply. A caller that wants it
            # live connects to /events instead.
            if ctx.buffered:
                payload = {"response": payload, "notifications": ctx.buffered}
            return HttpResponse.json(payload)

        @self.route("GET", "/healthz")
        async def healthz(request: HttpRequest) -> HttpResponse:
            """Liveness. Unauthenticated on purpose, so a probe needs no secret."""
            return HttpResponse.json(
                {"status": "ok", "service": self.server_name, "subscribers": len(self.subscribers)}
            )

        @self.route("GET", "/methods")
        async def methods(request: HttpRequest) -> HttpResponse:
            """Every JSON-RPC method this gateway exposes, with its summary."""
            return HttpResponse.json({"methods": self.router.methods})

        # Declared so /events resolves and authorises; the body is served by
        # _serve_events, which needs the socket rather than a response object.
        self.routes[("GET", "/events")] = _sse_placeholder

    async def _serve_events(
        self, request: HttpRequest, writer: asyncio.StreamWriter
    ) -> None:
        topics = None
        if "topics" in request.query:
            topics = {t.strip() for t in request.query["topics"].split(",") if t.strip()}
        headers = {
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-store",
            "Connection": "keep-alive",
            # Nginx buffers by default, which turns a live feed into a
            # surprise batch delivered at disconnect.
            "X-Accel-Buffering": "no",
            **self._cors_headers(request),
        }
        head = "HTTP/1.1 200 OK\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n"
        writer.write(head.encode("ascii"))
        await writer.drain()

        stream = EventStream(writer, topics=topics)
        self.subscribers.add(stream)
        await stream.send(
            "ready",
            {"subscriber": stream.id, "topics": sorted(topics) if topics else "all"},
        )
        try:
            while True:
                await asyncio.sleep(15)
                if not await stream.comment():
                    break
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            pass
        finally:
            stream.close()
            self.subscribers.discard(stream)


async def _sse_placeholder(request: HttpRequest) -> HttpResponse:  # pragma: no cover
    return HttpResponse.error(500, "event stream not wired")


# -- request parsing ------------------------------------------------------


class _BadRequest(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


async def _read_request(reader: asyncio.StreamReader, peer: str) -> HttpRequest | None:
    """Read one request. Returns None at clean end of connection."""
    try:
        start = await reader.readline()
    except (ConnectionResetError, asyncio.IncompleteReadError):
        return None
    if not start:
        return None
    # Tolerate leading CRLF, which some clients emit between pipelined requests.
    while start in (b"\r\n", b"\n"):
        start = await reader.readline()
        if not start:
            return None
    try:
        line = start.decode("ascii").rstrip("\r\n")
    except UnicodeDecodeError:
        raise _BadRequest(400, "request line is not ASCII") from None
    parts = line.split()
    if len(parts) != 3:
        raise _BadRequest(400, f"malformed request line: {line[:80]!r}")
    method, target, version = parts
    if not version.startswith("HTTP/"):
        raise _BadRequest(400, f"unsupported protocol {version!r}")

    headers: dict[str, str] = {}
    consumed = len(start)
    while True:
        raw = await reader.readline()
        if not raw:
            raise _BadRequest(400, "connection closed inside headers")
        consumed += len(raw)
        if consumed > MAX_HEADER_BYTES:
            raise _BadRequest(431, "header block too large")
        if raw in (b"\r\n", b"\n"):
            break
        try:
            text = raw.decode("latin-1").rstrip("\r\n")
        except UnicodeDecodeError:  # pragma: no cover
            raise _BadRequest(400, "undecodable header") from None
        name, sep, value = text.partition(":")
        if not sep:
            raise _BadRequest(400, f"malformed header: {text[:60]!r}")
        # Repeated headers join with a comma, per RFC 9110.
        key = name.strip().lower()
        headers[key] = f"{headers[key]}, {value.strip()}" if key in headers else value.strip()

    body = await _read_body(reader, headers)
    return HttpRequest(method.upper(), target, version, headers, body, peer)


async def _read_body(reader: asyncio.StreamReader, headers: dict[str, str]) -> bytes:
    if headers.get("transfer-encoding", "").lower().endswith("chunked"):
        return await _read_chunked(reader)
    raw_length = headers.get("content-length")
    if raw_length is None:
        return b""
    try:
        length = int(raw_length)
    except ValueError:
        raise _BadRequest(400, f"malformed Content-Length: {raw_length!r}") from None
    if length < 0:
        raise _BadRequest(400, "negative Content-Length")
    if length > MAX_BODY_BYTES:
        raise _BadRequest(413, f"body of {length} bytes exceeds the {MAX_BODY_BYTES} byte limit")
    if length == 0:
        return b""
    try:
        return await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        raise _BadRequest(400, "connection closed inside body") from None


async def _read_chunked(reader: asyncio.StreamReader) -> bytes:
    """Chunked transfer decoding.

    Supported because plenty of HTTP clients stream a POST body by default, and
    a gateway that rejected them would look broken for no good reason.
    """
    chunks = bytearray()
    while True:
        size_line = await reader.readline()
        if not size_line:
            raise _BadRequest(400, "connection closed inside chunked body")
        # A chunk size may carry extensions after a semicolon.
        size_text = size_line.split(b";")[0].strip()
        try:
            size = int(size_text, 16)
        except ValueError:
            raise _BadRequest(400, f"malformed chunk size: {size_text[:32]!r}") from None
        if size == 0:
            # Consume trailers up to the terminating blank line.
            while True:
                trailer = await reader.readline()
                if not trailer or trailer in (b"\r\n", b"\n"):
                    break
            return bytes(chunks)
        if len(chunks) + size > MAX_BODY_BYTES:
            raise _BadRequest(413, "chunked body exceeds the size limit")
        try:
            chunks += await reader.readexactly(size)
            await reader.readexactly(2)  # trailing CRLF
        except asyncio.IncompleteReadError:
            raise _BadRequest(400, "connection closed inside chunk") from None
