#!/usr/bin/env python3
"""Drive a LabBench gateway with no AI vendor SDK at all.

This is the one to copy for a model with no maintained Python client -- a
local model behind a bespoke API, a research checkpoint served over a raw
socket, or any agent loop you already own. It uses only LabBench's own
dependency-free client (`labbench.protocol.client`) and the neutral
`jsonschema` dialect, and the one function a real integration replaces is
`decide_next_action`: give it your model's output, get back a tool name and
arguments.

Run it as-is against the simulated lab and it drives a real, if scripted,
sequence -- home the stage, take a snapshot -- so you can see the exact
request/response shape a model needs to produce before wiring one up for
real:

    labbench serve -c configs/simulated-lab.yaml --transport ws
    python examples/agent_generic.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from _shared import LabBenchTools, common_args, connect_gateway

#: Replace this with a call to your own model. It receives the tool
#: declarations (neutral JSON Schema, with LabBench's hazard/annotations
#: metadata intact -- see `bridge/schema.py`'s `_json_schema` emitter) and the
#: conversation so far, and must return either `("device.invoke", {...})` or
#: `None` to end the turn. The scripted default below never looks at the
#: model output at all; it exists so this file is runnable without any model,
#: to demonstrate the wire shape.
_SCRIPT = iter([
    ("lab.find", {"feature": "MotionControl"}),
    ("device.invoke", {"device": "scope1", "feature": "MotionControl", "command": "home",
                        "reason": "example: prepare for imaging"}),
    ("device.invoke", {"device": "scope1", "feature": "Camera", "command": "snap",
                        "reason": "example: capture one frame"}),
])


def decide_next_action(
    tool_declarations: list[dict], conversation: list[dict],
) -> tuple[str, dict[str, Any]] | None:
    return next(_SCRIPT, None)


async def main() -> None:
    args = common_args(__doc__.splitlines()[0]).parse_args()
    gateway = await connect_gateway(args.gateway)
    tools = LabBenchTools(gateway)

    try:
        tool_declarations = await tools.fetch("jsonschema")
        conversation: list[dict] = [{"role": "user", "content": args.prompt}]

        while True:
            action = decide_next_action(tool_declarations, conversation)
            if action is None:
                break
            method, method_args = action
            print(f"  -> {method}({json.dumps(method_args)})", file=sys.stderr)
            # LabBenchTools.dispatch() expects a wire (underscore) name; the
            # dotted RPC name works too since dispatch() falls back to it
            # verbatim when it is not a known sanitised name.
            outcome = await tools.dispatch(method.replace(".", "_"), method_args)
            print(f"  <- {outcome}", file=sys.stderr)
            conversation.append({"role": "tool", "name": method, "content": outcome})
    finally:
        await gateway.close()


if __name__ == "__main__":
    asyncio.run(main())
