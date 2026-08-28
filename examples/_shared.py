"""Shared plumbing for the agent-loop examples.

Every example in this directory does the same three things, whichever AI
SDK it wraps: connect to a running gateway, fetch tool declarations in that
SDK's dialect, and dispatch a tool call the model emits back to the gateway.
This module is that shared middle step, factored out so each example reads
as "the agent loop", not "the agent loop plus RPC bookkeeping".

Nothing here is a labbench import surprise: `labbench.protocol.client` is the
package's own client, with no vendor SDK dependency, used exactly the way an
external process would use it -- these examples talk to the gateway over the
wire, the same as any other agent, rather than importing the Gateway class
in-process.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from labbench.bridge.schema import sanitise_name
from labbench.protocol.client import BaseClient, connect
from labbench.protocol.jsonrpc import JsonRpcError


class LabBenchTools:
    """Fetches tool declarations and maps a sanitised wire name back to the real method.

    Every dialect emitter replaces `.` with `_` in tool names (RFC-legal
    identifiers per vendor, `device.invoke` -> `device_invoke`) because no AI
    dialect accepts a dot in a function name. An agent loop that forgets this
    and calls the gateway with the sanitised name verbatim gets
    `METHOD_NOT_FOUND` back -- this class is the one place that translation
    happens, so no example has to remember it.
    """

    def __init__(self, client: BaseClient) -> None:
        self.client = client
        self._by_wire_name: dict[str, str] = {}

    async def fetch(self, dialect: str, *, strict: bool = False) -> Any:
        methods = (await self.client.call("tools.list"))["methods"]
        self._by_wire_name = {sanitise_name(name): name for name in methods}
        return await self.client.call("tools.schema", dialect=dialect, strict=strict)

    async def dispatch(self, wire_name: str, args: dict[str, Any]) -> str:
        """Call the real method for a tool name the model emitted.

        Returns a JSON string either way: a `SafetyViolation` or
        `ApprovalRequired` from the gateway is handed back to the model as
        the tool's result, not raised -- the model needs to *see* that its
        action was refused (and often the `approval_id` to retry with) to
        recover sensibly, the same as a human operator would.
        """
        method = self._by_wire_name.get(wire_name, wire_name)
        try:
            result = await self.client.call(method, **args)
            return json.dumps(result)
        except JsonRpcError as exc:
            return json.dumps({"error": exc.message, "data": exc.data})


def common_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--gateway", default="ws://127.0.0.1:8765/ws",
        help="ws://, http:// or stdio:// URL of a running `labbench serve` (default: %(default)s)",
    )
    parser.add_argument(
        "prompt", nargs="?",
        default="What instruments are in this lab, and what can you do with the microscope?",
        help="the instruction to hand the agent",
    )
    return parser


async def connect_gateway(url: str) -> BaseClient:
    client = connect(url, actor="agent:example")
    if hasattr(client, "connect"):
        await client.connect()
    return client
