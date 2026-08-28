"""Method registry and dispatch.

The router is the only place that knows how a Python exception becomes a wire
error, and it is deliberately strict about that mapping. A `LabBenchError`
carries a recovery class and a `state_uncertain` flag; both survive into the
JSON-RPC `error.data` so that an agent on the far side can decide between
retrying, re-homing and stopping. An *unexpected* exception is a different
thing entirely -- it means a driver has a bug and the instrument's state is
unknown -- so it is reported as such rather than flattened into the same shape.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ..core.errors import (
    ApprovalRequired,
    CapabilityNotFound,
    DeviceBusy,
    DeviceNotFound,
    JobNotFound,
    LabBenchError,
    SafetyViolation,
    ValidationError,
)
from .jsonrpc import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    SERVER_ERROR,
    JsonRpcError,
    Request,
    Response,
)

log = logging.getLogger("labbench.rpc")

#: Error classes that deserve a distinct wire code, so a client can branch on
#: the integer without parsing `data`. Everything else lands on SERVER_ERROR.
_ERROR_CODES: dict[type[LabBenchError], int] = {
    ValidationError: INVALID_PARAMS,
    DeviceNotFound: -32001,
    CapabilityNotFound: -32002,
    DeviceBusy: -32003,
    SafetyViolation: -32004,
    ApprovalRequired: -32005,
    JobNotFound: -32006,
}


def error_code_for(exc: LabBenchError) -> int:
    """Most specific registered code for this error's class."""
    for cls in type(exc).__mro__:
        if cls in _ERROR_CODES:
            return _ERROR_CODES[cls]  # type: ignore[index]
    return SERVER_ERROR


NotifySink = Callable[[str, dict[str, Any]], Awaitable[None]]


class RpcContext:
    """Per-call state handed to every handler.

    `notify` is the important one: it is how a handler pushes progress, device
    events and telemetry back to the caller *during* a call. On a duplex
    transport (WebSocket, stdio) those go out immediately; on a request/response
    transport they are buffered and attached to the reply, so no handler has to
    know which transport it is running under.
    """

    __slots__ = ("actor", "session_id", "transport", "peer", "_notify", "buffered", "started")

    def __init__(
        self,
        *,
        actor: str = "anonymous",
        session_id: str = "",
        transport: str = "unknown",
        peer: str = "",
        notify: NotifySink | None = None,
    ) -> None:
        self.actor = actor
        self.session_id = session_id
        self.transport = transport
        self.peer = peer
        self._notify = notify
        #: Notifications collected when the transport cannot push them live.
        self.buffered: list[dict[str, Any]] = []
        self.started = time.time()

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        if self._notify is not None:
            await self._notify(method, params)
        else:
            self.buffered.append({"method": method, "params": params})

    async def progress(self, fraction: float, message: str = "", **extra: Any) -> None:
        await self.notify(
            "progress",
            {"fraction": round(max(0.0, min(1.0, fraction)), 4), "message": message, **extra},
        )


Handler = Callable[..., Any]


class Router:
    """Maps method names to handlers.

    Handlers may be sync or async and are always awaited from the event loop;
    a synchronous handler is run in a worker thread rather than blocking it,
    because a driver that talks to a serial port over a blocking library is the
    normal case, not the exception.
    """

    def __init__(self) -> None:
        self._methods: dict[str, Handler] = {}
        self._descriptions: dict[str, str] = {}
        self._wants_context: dict[str, bool] = {}

    # -- registration -----------------------------------------------------

    def add(self, name: str, fn: Handler, description: str = "") -> None:
        if name in self._methods:
            raise ValueError(f"method {name!r} is already registered")
        sig = inspect.signature(fn)
        self._methods[name] = fn
        self._descriptions[name] = description or (inspect.getdoc(fn) or "").split("\n")[0]
        self._wants_context[name] = "ctx" in sig.parameters

    def method(self, name: str, description: str = "") -> Callable[[Handler], Handler]:
        def decorator(fn: Handler) -> Handler:
            self.add(name, fn, description)
            return fn

        return decorator

    def include(self, other: "Router", *, prefix: str = "") -> None:
        for name, fn in other._methods.items():
            self.add(f"{prefix}{name}", fn, other._descriptions.get(name, ""))

    @property
    def methods(self) -> dict[str, str]:
        return dict(self._descriptions)

    def has(self, name: str) -> bool:
        return name in self._methods

    # -- dispatch ---------------------------------------------------------

    async def dispatch(self, request: Request, ctx: RpcContext) -> Response | None:
        """Run one request. Returns None for a notification, always."""
        try:
            result = await self._invoke(request, ctx)
        except JsonRpcError as exc:
            if request.is_notification:
                log.info("notification %s failed: %s", request.method, exc.message)
                return None
            return Response.fail(request.id, exc)
        if request.is_notification:
            return None
        return Response.ok(request.id, result)

    async def _invoke(self, request: Request, ctx: RpcContext) -> Any:
        fn = self._methods.get(request.method)
        if fn is None:
            raise JsonRpcError(
                METHOD_NOT_FOUND,
                f"unknown method {request.method!r}",
                {"available": sorted(self._methods)},
            )
        if isinstance(request.params, list):
            raise JsonRpcError(
                INVALID_PARAMS,
                f"{request.method!r} takes named parameters, not positional; "
                "argument names carry the units and must not be dropped",
            )
        kwargs = dict(request.kwargs)
        if self._wants_context[request.method]:
            kwargs["ctx"] = ctx

        try:
            if inspect.iscoroutinefunction(fn):
                return await fn(**kwargs)
            # A blocking driver call must not stall every other device.
            return await asyncio.to_thread(lambda: fn(**kwargs))
        except TypeError as exc:
            # Distinguish "you called it wrong" from "it raised TypeError".
            if _is_signature_error(exc, fn):
                raise JsonRpcError(
                    INVALID_PARAMS, f"{request.method}: {exc}", {"method": request.method}
                ) from None
            raise JsonRpcError(
                INTERNAL_ERROR, f"{request.method}: {exc}", {"type": "TypeError"}
            ) from exc
        except LabBenchError as exc:
            raise JsonRpcError(error_code_for(exc), exc.message, exc.to_dict()) from None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A driver bug. The instrument's physical state is now unknown, and
            # saying so is more useful than a tidy generic message.
            log.exception("unhandled error in %s", request.method)
            raise JsonRpcError(
                SERVER_ERROR,
                f"{request.method} failed unexpectedly: {exc}",
                {
                    "error": "internal_error",
                    "message": str(exc),
                    "recovery": "human_required",
                    "state_uncertain": True,
                    "detail": {"exception": type(exc).__name__},
                },
            ) from exc

    async def dispatch_many(
        self, requests: list[Request], ctx: RpcContext
    ) -> list[Response]:
        """Run a batch concurrently, preserving order and dropping notifications.

        Concurrency here is what makes a batch worth sending: reading six
        instruments in one round trip should cost one instrument's latency, not
        six. Ordering of the *replies* is still positional.
        """
        results = await asyncio.gather(
            *(self.dispatch(r, ctx) for r in requests), return_exceptions=False
        )
        return [r for r in results if r is not None]


def _is_signature_error(exc: TypeError, fn: Handler) -> bool:
    """True when a TypeError came from calling `fn` wrongly, not from inside it."""
    text = str(exc)
    markers = (
        "required positional argument",
        "required keyword-only argument",
        "unexpected keyword argument",
        "positional arguments but",
        "missing 1 required",
    )
    if not any(m in text for m in markers):
        return False
    name = getattr(fn, "__name__", "")
    # The message names the callee when the call itself was malformed; a
    # TypeError raised *inside* the function names something else.
    return not name or name in text or "<lambda>" in text
