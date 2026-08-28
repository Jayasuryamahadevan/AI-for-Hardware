#!/usr/bin/env python3
"""Drive a LabBench gateway from Gemini's function-calling loop.

Prerequisites, neither of which is a LabBench dependency:

    pip install google-generativeai
    export GOOGLE_API_KEY=...

Then, in one terminal:

    labbench serve -c configs/simulated-lab.yaml --transport ws

and in another:

    python examples/agent_gemini.py "Home the microscope and take a snapshot."

Gemini's Python SDK has churned across major versions (this targets the
`google-generativeai` package's `GenerativeModel` chat API, current at the
time of writing); check https://ai.google.dev/gemini-api/docs/function-calling
if `genai.GenerativeModel` has moved by the time you read this. The dialect
LabBench emits -- one `function_declarations` list per tool -- is the stable
part regardless of which client class ends up consuming it.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from _shared import LabBenchTools, common_args, connect_gateway

MODEL = "gemini-2.0-flash"


async def main() -> None:
    args = common_args(__doc__.splitlines()[0]).parse_args()

    try:
        import google.generativeai as genai
    except ImportError:
        raise SystemExit("pip install google-generativeai") from None

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gateway = await connect_gateway(args.gateway)
    tools = LabBenchTools(gateway)

    try:
        tool_declarations = await tools.fetch("gemini")
        model = genai.GenerativeModel(MODEL, tools=tool_declarations)
        chat = model.start_chat()

        # google-generativeai's chat API is synchronous; run it in a worker
        # thread so it does not block the event loop this example's own RPC
        # calls to the gateway are running on.
        response = await asyncio.to_thread(chat.send_message, args.prompt)

        while True:
            calls = [
                part.function_call for part in response.candidates[0].content.parts
                if part.function_call
            ]
            for part in response.candidates[0].content.parts:
                if part.text:
                    print(part.text)
            if not calls:
                break

            reply_parts = []
            for call in calls:
                call_args = dict(call.args)
                print(f"  -> {call.name}({json.dumps(call_args)})", file=sys.stderr)
                outcome = await tools.dispatch(call.name, call_args)
                print(f"  <- {outcome}", file=sys.stderr)
                reply_parts.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=call.name, response={"result": json.loads(outcome)},
                        )
                    )
                )
            response = await asyncio.to_thread(chat.send_message, reply_parts)
    finally:
        await gateway.close()


if __name__ == "__main__":
    asyncio.run(main())
