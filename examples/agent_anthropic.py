#!/usr/bin/env python3
"""Drive a LabBench gateway from Claude's Messages API tool-use loop.

Prerequisites, neither of which is a LabBench dependency:

    pip install anthropic
    export ANTHROPIC_API_KEY=...

Then, in one terminal:

    labbench serve -c configs/simulated-lab.yaml --transport ws

and in another:

    python examples/agent_anthropic.py "Home the microscope and take a snapshot."

The loop below is the standard Anthropic tool-use shape: send the messages
and tool declarations, and for every `tool_use` content block Claude
returns, run it against the gateway and feed the result back as a
`tool_result` block, until Claude stops asking for tools.
"""

from __future__ import annotations

import asyncio
import json
import sys

from _shared import LabBenchTools, common_args, connect_gateway

MODEL = "claude-sonnet-5"


async def main() -> None:
    args = common_args(__doc__.splitlines()[0]).parse_args()

    try:
        import anthropic
    except ImportError:
        raise SystemExit("pip install anthropic") from None

    client = anthropic.AsyncAnthropic()
    gateway = await connect_gateway(args.gateway)
    tools = LabBenchTools(gateway)

    try:
        tool_declarations = await tools.fetch("anthropic")
        messages: list[dict] = [{"role": "user", "content": args.prompt}]

        while True:
            response = await client.messages.create(
                model=MODEL, max_tokens=4096, tools=tool_declarations, messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            for block in response.content:
                if block.type == "text":
                    print(block.text)

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                print(f"  -> {block.name}({json.dumps(block.input)})", file=sys.stderr)
                outcome = await tools.dispatch(block.name, block.input)
                print(f"  <- {outcome}", file=sys.stderr)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": outcome}
                )
            messages.append({"role": "user", "content": tool_results})
    finally:
        await gateway.close()


if __name__ == "__main__":
    asyncio.run(main())
