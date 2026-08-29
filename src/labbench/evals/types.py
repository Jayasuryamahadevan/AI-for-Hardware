"""The neutral shapes every policy, task and report deal in.

An eval has to compare Claude, GPT and Gemini on the same footing, and each
vendor's tool-calling wire shape is different enough (content blocks vs.
`tool_calls` vs. `function_call` parts) that comparing transcripts directly
would mean three parsers in every consumer. `AgentTurn` is the one shape a
`Policy` translates into and everything downstream reads from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """One tool invocation a model asked for, in wire-name form (dots
    replaced with underscores -- see `bridge.schema.sanitise_name`)."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass
class AgentTurn:
    """One round of a policy's response: whatever text it said, and whatever
    tools it wants run before it will say anything else."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ToolCallRecord:
    """One tool call as it actually happened, kept for grading and reporting."""

    name: str
    args: dict[str, Any]
    result: str  # JSON-encoded, exactly what the model saw back

    @property
    def failed(self) -> bool:
        """True when the gateway refused the call or it raised.

        `_ToolDispatcher.dispatch` always returns valid JSON, error or not, so
        this is a cheap string check rather than a second round of parsing.
        """
        return '"error"' in self.result


@dataclass
class EvalTranscript:
    """Everything one episode produced, for a grader to inspect and a report
    to render. Never includes the model's raw SDK objects -- only what an
    auditor reading a transcript later would actually want."""

    task_id: str
    text: list[str] = field(default_factory=list)
    calls: list[ToolCallRecord] = field(default_factory=list)
    turns: int = 0
    truncated: bool = False

    @property
    def tool_errors(self) -> int:
        return sum(1 for c in self.calls if c.failed)

    def calls_named(self, suffix: str) -> list[ToolCallRecord]:
        """Calls whose wire name ends in `suffix`, e.g. 'clear_fault' matches
        both `device_clear_fault` and a future `foo_clear_fault`."""
        return [c for c in self.calls if c.name.endswith(suffix)]


@dataclass
class Verdict:
    """What a grader decides, before the harness attaches identity/transcript."""

    passed: bool
    score: float  # 0.0-1.0; finer-grained than `passed` so partial credit shows up
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """One task, fully graded."""

    task_id: str
    category: str
    passed: bool
    score: float
    reasons: list[str]
    metrics: dict[str, Any]
    transcript: EvalTranscript

    def summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "passed": self.passed,
            "score": round(self.score, 3),
            "reasons": self.reasons,
            "metrics": self.metrics,
            "turns": self.transcript.turns,
            "truncated": self.transcript.truncated,
            "tool_calls": len(self.transcript.calls),
            "tool_errors": self.transcript.tool_errors,
        }
