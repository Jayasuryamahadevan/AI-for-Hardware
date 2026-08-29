"""Scored agent evals: fixed tasks against the simulated lab, graded
mechanically (ledger and device state, never a second model's opinion of the
first one's prose), runnable across any of the AI dialects `bridge/schema.py`
supports so a claim about agent behaviour is a number, not a demo.

    labbench eval list
    labbench eval run --dialect anthropic --task envelope_refusal

See `tasks.py` for what is actually being tested and why each grader is
built the way it is; see `harness.py` for how one episode is driven and
graded end to end.
"""

from __future__ import annotations

from .harness import EvalRunner
from .policy import AnthropicPolicy, GeminiPolicy, OpenAIPolicy, Policy, ScriptedPolicy
from .report import render_table, summarize
from .tasks import TASKS, EvalTask, all_tasks, get
from .types import AgentTurn, EvalResult, EvalTranscript, ToolCall, ToolCallRecord, Verdict

__all__ = [
    "TASKS",
    "AgentTurn",
    "AnthropicPolicy",
    "EvalResult",
    "EvalRunner",
    "EvalTask",
    "EvalTranscript",
    "GeminiPolicy",
    "OpenAIPolicy",
    "Policy",
    "ScriptedPolicy",
    "ToolCall",
    "ToolCallRecord",
    "Verdict",
    "all_tasks",
    "get",
    "render_table",
    "summarize",
]
