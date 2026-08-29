"""Policies: one per AI dialect, each hiding a vendor SDK's own conversation
shape behind the same three-method interface the harness drives.

This is deliberately the same loop `examples/agent_*.py` hand-rolls per
vendor -- send messages and tools, get text and tool calls back, feed tool
results back in -- refactored so the harness can drive Claude, GPT and
Gemini identically and so a fourth, dependency-free `ScriptedPolicy` can
stand in for all three in a test that must not need an API key or a network.

No policy is a LabBench runtime dependency. Importing a vendor SDK happens
inside the policy's constructor, exactly where `examples/agent_*.py` does it,
so `pip install labbench[evals]`... does not exist, and does not need to --
you install whichever one SDK you are actually evaluating.
"""

from __future__ import annotations

import abc
import json
from typing import Any

from .types import AgentTurn, ToolCall


class Policy(abc.ABC):
    """One conversation with one model, across one episode.

    `dialect` selects which of `tools.schema`'s projections the harness
    fetches for this policy -- the whole reason `bridge/schema.py` exists is
    so this class does not have to know what a tool schema looks like, only
    what its own vendor's *messages* look like.
    """

    dialect: str

    @abc.abstractmethod
    async def start(
        self, system_prompt: str, user_prompt: str, tool_declarations: Any
    ) -> AgentTurn: ...

    @abc.abstractmethod
    async def continue_with(self, results: list[tuple[str, str]]) -> AgentTurn:
        """`results` is (tool_call_id, JSON result string) pairs, in the order
        the previous turn's tool calls were made."""


class ScriptedPolicy(Policy):
    """Plays back a fixed sequence of turns. No model, no network, no key.

    This is what `tests/test_evals.py` uses to prove the harness's mechanics
    (turn-looping, tool dispatch, the max-turns cap) and every task's grader
    logic without spending a cent or depending on a vendor being reachable --
    the same role `agent_generic.py`'s scripted loop plays in the examples.
    """

    dialect = "jsonschema"

    def __init__(self, turns: list[AgentTurn]) -> None:
        self._turns = list(turns)
        self._i = 0

    async def start(self, system_prompt: str, user_prompt: str, tool_declarations: Any) -> AgentTurn:
        return self._next()

    async def continue_with(self, results: list[tuple[str, str]]) -> AgentTurn:
        return self._next()

    def _next(self) -> AgentTurn:
        if self._i >= len(self._turns):
            return AgentTurn(text="(scripted policy exhausted)")
        turn = self._turns[self._i]
        self._i += 1
        return turn


class AnthropicPolicy(Policy):
    dialect = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        try:
            import anthropic
        except ImportError:
            raise SystemExit("pip install anthropic") from None
        self._client = anthropic.AsyncAnthropic()
        self._model = model
        self._messages: list[dict[str, Any]] = []
        self._tools: Any = None
        self._system = ""

    async def start(self, system_prompt: str, user_prompt: str, tool_declarations: Any) -> AgentTurn:
        self._system = system_prompt
        self._tools = tool_declarations
        self._messages = [{"role": "user", "content": user_prompt}]
        return await self._turn()

    async def continue_with(self, results: list[tuple[str, str]]) -> AgentTurn:
        self._messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_id, "content": content}
                for tool_id, content in results
            ],
        })
        return await self._turn()

    async def _turn(self) -> AgentTurn:
        kwargs: dict[str, Any] = {
            "model": self._model, "max_tokens": 4096,
            "tools": self._tools, "messages": self._messages,
        }
        if self._system:
            kwargs["system"] = self._system
        response = await self._client.messages.create(**kwargs)
        self._messages.append({"role": "assistant", "content": response.content})
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, args=b.input)
            for b in response.content if b.type == "tool_use"
        ]
        return AgentTurn(text=text, tool_calls=calls)


class OpenAIPolicy(Policy):
    """Also the policy for a local OpenAI-compatible server: pass `base_url`."""

    dialect = "openai"

    def __init__(self, model: str = "gpt-4o", *, base_url: str | None = None) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise SystemExit("pip install openai") from None
        self._client = AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI()
        self._model = model
        self._messages: list[dict[str, Any]] = []
        self._tools: Any = None

    async def start(self, system_prompt: str, user_prompt: str, tool_declarations: Any) -> AgentTurn:
        self._tools = tool_declarations
        self._messages = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})
        self._messages.append({"role": "user", "content": user_prompt})
        return await self._turn()

    async def continue_with(self, results: list[tuple[str, str]]) -> AgentTurn:
        for tool_id, content in results:
            self._messages.append({"role": "tool", "tool_call_id": tool_id, "content": content})
        return await self._turn()

    async def _turn(self) -> AgentTurn:
        response = await self._client.chat.completions.create(
            model=self._model, messages=self._messages, tools=self._tools,
        )
        message = response.choices[0].message
        self._messages.append(message.model_dump(exclude_none=True))
        calls = [
            ToolCall(id=c.id, name=c.function.name, args=json.loads(c.function.arguments or "{}"))
            for c in (message.tool_calls or [])
        ]
        return AgentTurn(text=message.content or "", tool_calls=calls)


class GeminiPolicy(Policy):
    dialect = "gemini"

    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        try:
            import google.generativeai as genai
        except ImportError:
            raise SystemExit("pip install google-generativeai") from None
        import os

        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self._genai = genai
        self._model_name = model
        self._chat: Any = None

    async def start(self, system_prompt: str, user_prompt: str, tool_declarations: Any) -> AgentTurn:
        import asyncio

        model = self._genai.GenerativeModel(
            self._model_name, tools=tool_declarations,
            system_instruction=system_prompt or None,
        )
        self._chat = model.start_chat()
        response = await asyncio.to_thread(self._chat.send_message, user_prompt)
        return self._turn_from(response)

    async def continue_with(self, results: list[tuple[str, str]]) -> AgentTurn:
        import asyncio

        parts = [
            self._genai.protos.Part(
                function_response=self._genai.protos.FunctionResponse(
                    name=tool_id, response={"result": json.loads(content)},
                )
            )
            for tool_id, content in results
        ]
        response = await asyncio.to_thread(self._chat.send_message, parts)
        return self._turn_from(response)

    def _turn_from(self, response: Any) -> AgentTurn:
        parts = response.candidates[0].content.parts
        text = "".join(p.text for p in parts if p.text)
        # Gemini has no per-call id; the function's own name stands in for
        # one, matched back up in `continue_with` the same way `_shared.py`'s
        # example does it.
        calls = [
            ToolCall(id=p.function_call.name, name=p.function_call.name, args=dict(p.function_call.args))
            for p in parts if p.function_call
        ]
        return AgentTurn(text=text, tool_calls=calls)
