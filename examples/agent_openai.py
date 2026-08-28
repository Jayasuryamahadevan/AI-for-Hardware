#!/usr/bin/env python3
"""Drive a LabBench gateway from OpenAI's function-calling loop.

Prerequisites, neither of which is a LabBench dependency:

    pip install openai
    export OPENAI_API_KEY=...

Then, in one terminal:

    labbench serve -c configs/simulated-lab.yaml --transport ws

and in another:

    python examples/agent_openai.py "Home the microscope and take a snapshot."

This is also the example to adapt for a *local* model: point `base_url` at
any OpenAI-compatible server (llama.cpp, vLLM, Ollama's OpenAI shim, ...) and
the rest of the loop is unchanged -- the "openai" dialect is not
OpenAI-specific, it is whatever speaks that wire shape, which is most local
inference servers today.
"""

from __future__ import annotations

import asyncio
import json
import sys

from _shared import LabBenchTools, common_args, connect_gateway

MODEL = "gpt-4o"


async def main() -> None:
    args = common_args(__doc__.splitlines()[0]).parse_args()

    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise SystemExit("pip install openai") from None

    client = AsyncOpenAI()  # base_url="http://localhost:8080/v1" for a local server
    gateway = await connect_gateway(args.gateway)
    tools = LabBenchTools(gateway)

    try:
        tool_declarations = await tools.fetch("openai")
        messages: list[dict] = [{"role": "user", "content": args.prompt}]

        while True:
            response = await client.chat.completions.create(
                model=MODEL, messages=messages, tools=tool_declarations,
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))

            if message.content:
                print(message.content)
            if not message.tool_calls:
                break

            for call in message.tool_calls:
                call_args = json.loads(call.function.arguments or "{}")
                print(f"  -> {call.function.name}({call.function.arguments})", file=sys.stderr)
                outcome = await tools.dispatch(call.function.name, call_args)
                print(f"  <- {outcome}", file=sys.stderr)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": outcome}
                )
    finally:
        await gateway.close()


if __name__ == "__main__":
    asyncio.run(main())
