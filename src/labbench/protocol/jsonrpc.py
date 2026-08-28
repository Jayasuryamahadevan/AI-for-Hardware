"""JSON-RPC 2.0, implemented to the letter of the specification.

Why this and not something bespoke: an agent gateway needs request/response
correlation, out-of-band notifications, and a batch form -- which is exactly
JSON-RPC's remit and nothing more. Anything we invented would converge on it
badly. Anything larger (gRPC, GraphQL) would drag in a toolchain.

The one place we extend the specification is the error `data` field, which
carries the structured `LabBenchError` payload: an agent needs to know whether
a failure is retryable and whether the instrument's physical state is still
trustworthy, and a bare error string cannot say that.
"""

from __future__ import annotations

import json
from typing import Any, Literal

# -- Specification error codes -------------------------------------------
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
#: -32000..-32099 is the implementation-defined range. Everything LabBench
#: raises that is not a protocol fault lands on SERVER_ERROR and carries its
#: real taxonomy in `data`.
SERVER_ERROR = -32000

VERSION = "2.0"


class JsonRpcError(Exception):
    """An error destined for the wire.

    `data` is where the useful part lives. For a driver failure it holds the
    `LabBenchError.to_dict()` payload -- error code, recovery hint, and the
    `state_uncertain` flag that tells an agent whether it may keep going.
    """

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            out["data"] = self.data
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"JsonRpcError(code={self.code}, message={self.message!r})"


class Request:
    """One inbound call.

    A request with no `id` is a *notification*: the peer wants no reply and
    must not receive one, even on failure. Telemetry and progress updates use
    that form, which is why the distinction is kept explicit rather than
    collapsed into `id is None`.
    """

    __slots__ = ("id", "is_notification", "method", "params")

    def __init__(
        self,
        method: str,
        params: dict[str, Any] | list[Any] | None = None,
        id: str | int | None = None,
        *,
        is_notification: bool = False,
    ) -> None:
        self.method = method
        self.params = params if params is not None else {}
        self.id = id
        self.is_notification = is_notification

    @property
    def kwargs(self) -> dict[str, Any]:
        """Params as keyword arguments.

        Positional params are legal JSON-RPC but a poor fit for an instrument
        API, where the argument *name* is what carries the unit. We surface
        them as empty so the router can reject them with a real message rather
        than a TypeError from deep inside a driver.
        """
        return self.params if isinstance(self.params, dict) else {}

    @classmethod
    def from_dict(cls, obj: Any) -> Request:
        if not isinstance(obj, dict):
            raise JsonRpcError(INVALID_REQUEST, "request must be a JSON object")
        if obj.get("jsonrpc") != VERSION:
            raise JsonRpcError(
                INVALID_REQUEST, f"jsonrpc must be {VERSION!r}", {"got": obj.get("jsonrpc")}
            )
        method = obj.get("method")
        if not isinstance(method, str) or not method:
            raise JsonRpcError(INVALID_REQUEST, "method must be a non-empty string")
        params = obj.get("params")
        if params is not None and not isinstance(params, (dict, list)):
            raise JsonRpcError(INVALID_PARAMS, "params must be an object or array")
        # Absent `id` means notification. A *present* null id is malformed, but
        # treating it as a notification is friendlier than refusing the call.
        has_id = "id" in obj and obj["id"] is not None
        rid = obj.get("id") if has_id else None
        if has_id and not isinstance(rid, (str, int)):
            raise JsonRpcError(INVALID_REQUEST, "id must be a string or number")
        return cls(method, params, rid, is_notification=not has_id)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"jsonrpc": VERSION, "method": self.method}
        if self.params:
            out["params"] = self.params
        if not self.is_notification:
            out["id"] = self.id
        return out


class Response:
    """One outbound reply, or a server-initiated notification.

    Exactly one of `result` and `error` is present on a reply, per the
    specification. `Response.notification` builds the id-less form used for
    progress, telemetry and device events.
    """

    __slots__ = ("error", "id", "kind", "method", "result")

    def __init__(
        self,
        id: str | int | None = None,
        result: Any = None,
        error: JsonRpcError | None = None,
        method: str | None = None,
        kind: Literal["result", "error", "notification"] = "result",
    ) -> None:
        self.id = id
        self.result = result
        self.error = error
        self.method = method
        self.kind = kind

    @classmethod
    def ok(cls, id: str | int | None, result: Any) -> Response:
        return cls(id=id, result=result, kind="result")

    @classmethod
    def fail(cls, id: str | int | None, error: JsonRpcError) -> Response:
        return cls(id=id, error=error, kind="error")

    @classmethod
    def notification(cls, method: str, params: Any = None) -> Response:
        return cls(result=params, method=method, kind="notification")

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "notification":
            out: dict[str, Any] = {"jsonrpc": VERSION, "method": self.method}
            if self.result is not None:
                out["params"] = self.result
            return out
        if self.kind == "error":
            assert self.error is not None
            return {"jsonrpc": VERSION, "id": self.id, "error": self.error.to_dict()}
        # `result` is required on success, and null is a legal result.
        return {"jsonrpc": VERSION, "id": self.id, "result": self.result}


def parse_message(raw: str | bytes) -> tuple[list[Request], bool]:
    """Parse one wire message into requests.

    Returns `(requests, is_batch)`. Batch order is preserved so a caller that
    ignores ids can still match replies positionally.

    Raises `JsonRpcError` for undecodable JSON and for the empty batch, both of
    which the specification calls out explicitly.
    """
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise JsonRpcError(PARSE_ERROR, f"payload is not valid UTF-8: {exc}") from None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JsonRpcError(PARSE_ERROR, f"invalid JSON: {exc.msg} at position {exc.pos}") from None

    if isinstance(obj, list):
        if not obj:
            raise JsonRpcError(INVALID_REQUEST, "batch must not be empty")
        return [Request.from_dict(item) for item in obj], True
    return [Request.from_dict(obj)], False


def serialise(payload: Any) -> str:
    """Encode a reply.

    `default=str` is a safety net, not a design. numpy scalars and Path objects
    do escape from drivers, and dropping an entire instrument reply because one
    field was a `PosixPath` would be a poor trade.
    """
    return json.dumps(payload, separators=(",", ":"), default=str)
