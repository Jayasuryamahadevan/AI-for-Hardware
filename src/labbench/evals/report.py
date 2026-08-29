"""Turning a list of `EvalResult`s into something a person or a CI dashboard
reads in one glance -- a per-category rollup plus a fixed-width table, no
plotting dependency required for the case that matters most: a terminal."""

from __future__ import annotations

from statistics import mean
from typing import Any

from .types import EvalResult


def summarize(results: list[EvalResult]) -> dict[str, Any]:
    if not results:
        return {"tasks": 0, "passed": 0, "mean_score": 0.0, "by_category": {}, "results": []}
    by_category: dict[str, dict[str, Any]] = {}
    for result in results:
        bucket = by_category.setdefault(result.category, {"tasks": 0, "passed": 0, "scores": []})
        bucket["tasks"] += 1
        bucket["passed"] += int(result.passed)
        bucket["scores"].append(result.score)
    for bucket in by_category.values():
        bucket["mean_score"] = round(mean(bucket.pop("scores")), 3)
    return {
        "tasks": len(results),
        "passed": sum(r.passed for r in results),
        "mean_score": round(mean(r.score for r in results), 3),
        "by_category": by_category,
        "results": [r.summary() for r in results],
    }


def render_table(results: list[EvalResult]) -> str:
    header = f"{'TASK':<24}{'CATEGORY':<12}{'RESULT':<7}{'SCORE':<7}{'TURNS':<7}REASONS"
    lines = [header, "-" * len(header)]
    for result in results:
        verdict = "PASS" if result.passed else "FAIL"
        reasons = "; ".join(result.reasons)
        lines.append(
            f"{result.task_id:<24}{result.category:<12}{verdict:<7}"
            f"{result.score:<7.2f}{result.transcript.turns:<7}{reasons}"
        )
    summary = summarize(results)
    lines.append("-" * len(header))
    lines.append(f"{summary['passed']}/{summary['tasks']} passed, mean score {summary['mean_score']}")
    return "\n".join(lines)
