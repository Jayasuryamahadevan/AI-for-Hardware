"""The wire protocol, implemented here rather than imported.

LabBench speaks JSON-RPC 2.0 over four interchangeable transports: stdio,
HTTP/1.1, Server-Sent Events and WebSocket. All four are written from scratch
against `asyncio` and the standard library.

That is a deliberate cost. The alternative -- taking a web framework and an
agent-vendor SDK -- would have meant the agent-facing contract of a laboratory
gateway was only as stable as two third-party release cadences, and that an
instrument running a five-year experiment could be broken by someone else's
major version. The protocol surface here is small enough to own outright:
roughly a thousand lines, fully specified by RFC 6455 and the JSON-RPC 2.0
specification, with no runtime dependency beyond the standard library.
"""

from .jsonrpc import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    SERVER_ERROR,
    JsonRpcError,
    Request,
    Response,
    parse_message,
    serialise,
)
from .router import Router, RpcContext

__all__ = [
    "JsonRpcError", "Request", "Response", "parse_message", "serialise",
    "PARSE_ERROR", "INVALID_REQUEST", "METHOD_NOT_FOUND", "INVALID_PARAMS",
    "INTERNAL_ERROR", "SERVER_ERROR",
    "Router", "RpcContext",
]
