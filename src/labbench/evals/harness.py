"""The runner: drives one task through one policy against a fresh gateway.

Deliberately in-process rather than a real `labbench serve` plus a real
socket. Every call still crosses the real `Router.dispatch` -- the same
JSON-RPC method resolution, context construction and error-code mapping a
wire transport uses -- so this is not a mock of the gateway, only of the
transport underneath it. That trade is the right one for an eval: a wire
round-trip changes nothing a grader could observe, and an eval suite that
runs a few hundred episodes across three models cannot afford one HTTP
process per episode.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ..bridge.schema import sanitise_name
from ..core.registry import LabConfig
from ..gateway import Gateway
from ..protocol.jsonrpc import JsonRpcError, Request
from ..protocol.router import RpcContext
from .policy import Policy
from .types import EvalResult, EvalTranscript, ToolCallRecord


class _InProcessClient:
    """The same `.call(method, **params)` surface `protocol.client.BaseClient`
    exposes, minus the socket -- so the dispatcher below cannot tell the
    difference and does not need to."""

    def __init__(self, gateway: Gateway, *, actor: str = "agent:eval") -> None:
        self._gateway = gateway
        self._ctx = RpcContext(actor=actor, transport="eval")
        self._next_id = 0

    async def call(self, method: str, **params: Any) -> Any:
        self._next_id += 1
        response = await self._gateway.router.dispatch(
            Request(method, params, id=self._next_id), self._ctx
        )
        assert response is not None
        if response.error is not None:
            raise response.error
        return response.result


class _ToolDispatcher:
    """`examples/_shared.py`'s `LabBenchTools`, against an in-process client.

    Kept separate from the examples rather than imported from them: those
    scripts are reference material meant to be copied by an external caller
    wiring up their own agent, not a dependency of the package itself.
    """

    def __init__(self, client: _InProcessClient) -> None:
        self._client = client
        self._by_wire_name: dict[str, str] = {}

    async def fetch(self, dialect: str) -> Any:
        methods = (await self._client.call("tools.list"))["methods"]
        self._by_wire_name = {sanitise_name(name): name for name in methods}
        return await self._client.call("tools.schema", dialect=dialect)

    async def dispatch(self, wire_name: str, args: dict[str, Any]) -> str:
        method = self._by_wire_name.get(wire_name, wire_name)
        try:
            result = await self._client.call(method, **args)
            return json.dumps(result)
        except JsonRpcError as exc:
            return json.dumps({"error": exc.message, "data": exc.data})


SetupFn = Callable[[Gateway], Awaitable[None]]
GradeFn = Callable[[Gateway, EvalTranscript], Awaitable[Any]]  # returns Verdict


class EvalRunner:
    """Runs one `EvalTask` through one `Policy`, start to graded finish."""

    def __init__(self, *, data_dir: str | Path, max_turns: int = 8) -> None:
        self.data_dir = Path(data_dir)
        self.max_turns = max_turns

    async def run(self, task: Any, policy: Policy) -> EvalResult:
        config = LabConfig.load(task.config_path) if task.config_path else LabConfig()
        # A fresh directory per *episode*, not per task: a grader that reads
        # the ledger (several of them do -- an escalation leaves no trace
        # anywhere else) must never see a record left behind by an earlier
        # run of the same task, whether that is a repeated CLI invocation or
        # two policies compared back to back in one process.
        episode_dir = self.data_dir / task.id / uuid.uuid4().hex[:12]
        gateway = Gateway(config, data_dir=episode_dir)
        await gateway.start()
        try:
            if task.setup is not None:
                await task.setup(gateway)

            client = _InProcessClient(gateway)
            dispatcher = _ToolDispatcher(client)
            tool_declarations = await dispatcher.fetch(policy.dialect)

            transcript = EvalTranscript(task_id=task.id)
            turn = await policy.start(task.system_prompt, task.prompt, tool_declarations)
            transcript.turns = 1

            while True:
                if turn.text:
                    transcript.text.append(turn.text)
                if not turn.tool_calls:
                    break
                if transcript.turns >= self.max_turns:
                    transcript.truncated = True
                    break
                results: list[tuple[str, str]] = []
                for call in turn.tool_calls:
                    outcome = await dispatcher.dispatch(call.name, call.args)
                    transcript.calls.append(
                        ToolCallRecord(name=call.name, args=call.args, result=outcome)
                    )
                    results.append((call.id, outcome))
                turn = await policy.continue_with(results)
                transcript.turns += 1

            # A command like `home` is observable: `device.invoke` handed the
            # model a job handle immediately rather than blocking, and the
            # model is not required to poll it before it stops talking.
            # Grading must judge what actually happened physically, not the
            # chat transcript at the instant the model went quiet -- so any
            # job the episode left running gets a bounded chance to finish
            # before the grader looks at device state.
            await self._settle(gateway)

            verdict = await task.grade(gateway, transcript)
            return EvalResult(
                task_id=task.id, category=task.category, passed=verdict.passed,
                score=verdict.score, reasons=verdict.reasons, metrics=verdict.metrics,
                transcript=transcript,
            )
        finally:
            await gateway.close()

    async def _settle(self, gateway: Gateway, timeout: float = 15.0) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        for job in gateway.jobs.list():
            if job.status.terminal:
                continue
            remaining = max(0.0, deadline - asyncio.get_event_loop().time())
            await gateway.jobs.wait(job.id, remaining)
