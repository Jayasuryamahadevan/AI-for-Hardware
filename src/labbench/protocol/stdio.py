"""Stdio transport: the gateway as a subprocess.

This is the transport for a *local* agent -- the process that spawned LabBench
reads and writes its stdin/stdout. There is no port to secure, no token to
manage, and the operating system already answers the only authorisation
question that matters: whoever can spawn the process can drive the bench.

Two rules make it work:

Nothing but protocol may be written to stdout. A stray `print` in a driver
would corrupt the stream, so this module rebinds `sys.stdout` to stderr for the
lifetime of the session. Logs go to stderr, where they belong.

The transport is duplex. Notifications -- progress, device events, telemetry --
are written as they happen rather than being attached to a reply, so an agent
watching a forty-minute tile scan sees it move.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from typing import Any

from .framing import FrameError, Framing, encode, read_frame
from .jsonrpc import INTERNAL_ERROR, JsonRpcError, Response, parse_message, serialise
from .router import Router, RpcContext

log = logging.getLogger("labbench.stdio")


class StdinUnavailable(RuntimeError):
    """Stdin cannot carry a protocol stream on this process."""


def stdin_is_usable() -> bool:
    """Whether stdin can be polled for readability.

    A pipe or a terminal can; a regular file or /dev/null cannot be registered
    with epoll, which is what a daemonised process usually has. Checking up
    front turns a confusing asyncio traceback into a decision the caller can
    make.
    """
    try:
        import selectors
        import stat

        mode = os.fstat(sys.stdin.fileno()).st_mode
        if not (stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode) or stat.S_ISCHR(mode)):
            return False
        # A character device may still be /dev/null, which epoll refuses.
        selector = selectors.DefaultSelector()
        try:
            selector.register(sys.stdin.fileno(), selectors.EVENT_READ)
            selector.unregister(sys.stdin.fileno())
            return True
        finally:
            selector.close()
    except (OSError, ValueError, AttributeError):
        return False


async def _connect_stdin() -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    loop = asyncio.get_running_loop()
    try:
        await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    except (PermissionError, OSError, ValueError) as exc:
        raise StdinUnavailable(
            f"stdin cannot carry a protocol stream ({exc}). The stdio transport needs "
            "a pipe from a parent process; a daemon with stdin on /dev/null should use "
            "the http or ws transport instead."
        ) from None
    return reader


class StdioServer:
    """Serve one session over stdin/stdout."""

    def __init__(
        self,
        router: Router,
        *,
        framing: Framing = Framing.LINE,
        actor: str = "agent:stdio",
    ) -> None:
        self.router = router
        self.framing = framing
        self.actor = actor
        self._out_lock = asyncio.Lock()
        self._stdout = sys.stdout.buffer

    async def _write(self, payload: dict[str, Any]) -> None:
        data = encode(serialise(payload), self.framing)
        # One writer at a time: a notification emitted from a background job
        # must not interleave its bytes with a reply.
        async with self._out_lock:
            self._stdout.write(data)
            self._stdout.flush()

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._write(Response.notification(method, params).to_dict())

    async def serve(self) -> None:
        # Protect the stream from anything in a driver that prints.
        real_stdout, sys.stdout = sys.stdout, sys.stderr
        reader = await _connect_stdin()
        ctx = RpcContext(
            actor=self.actor, transport="stdio", peer="parent-process", notify=self._notify
        )
        tasks: set[asyncio.Task[None]] = set()
        try:
            while True:
                try:
                    raw = await read_frame(reader, self.framing)
                except FrameError as exc:
                    log.error("framing error, closing session: %s", exc)
                    break
                if raw is None:
                    break  # clean EOF: the parent closed the pipe
                task = asyncio.create_task(self._handle(raw, ctx))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        finally:
            # Let in-flight calls finish so hardware is not abandoned mid-move.
            if tasks:
                await asyncio.wait(tasks, timeout=30)
            for task in tasks:
                task.cancel()
            sys.stdout = real_stdout

    async def _handle(self, raw: str, ctx: RpcContext) -> None:
        try:
            requests, is_batch = parse_message(raw)
        except JsonRpcError as exc:
            await self._write(Response.fail(None, exc).to_dict())
            return
        try:
            if is_batch:
                replies = await self.router.dispatch_many(requests, ctx)
                if replies:  # an all-notification batch gets no reply at all
                    await self._write([r.to_dict() for r in replies])  # type: ignore[arg-type]
                return
            reply = await self.router.dispatch(requests[0], ctx)
            if reply is not None:
                await self._write(reply.to_dict())
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - last-resort guard
            log.exception("dispatch failed")
            rid = requests[0].id if requests and not is_batch else None
            await self._write(
                Response.fail(rid, JsonRpcError(INTERNAL_ERROR, str(exc))).to_dict()
            )


async def serve_stdio(router: Router, **kwargs: Any) -> None:
    await StdioServer(router, **kwargs).serve()


def run_stdio(router: Router, **kwargs: Any) -> None:
    """Blocking entry point for the CLI."""
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve_stdio(router, **kwargs))
