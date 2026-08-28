"""Protocols: a declared sequence of `device.invoke` calls, checked before it runs.

An agent could already drive a multi-step procedure by calling `device.invoke`
in a loop. A `Protocol` exists because that loop is worth naming, saving, and
validating as a whole *before* the first motor turns -- the same reason a
`SimulationResult` exists for one command, generalised to a run: a plan is
cheap to check and expensive to discover is wrong three steps in.

A step's arguments may reference an earlier step's result or a run variable
with ``${...}``, e.g. ``z_um: "${autofocus.result.z_um}"``. This is
deliberately not a scripting language -- no conditionals, no loops beyond
`repeat` -- because a protocol is a record meant to be read by a human and
re-run verbatim, and a Turing-complete format stops being either.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import ValidationError

_REF = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\}")


class ProtocolStep(BaseModel):
    """One `device.invoke` call, plus how to react when it does not go cleanly."""

    model_config = ConfigDict(extra="forbid")

    #: Referenced by later steps as `${label.result....}`. Auto-generated if omitted.
    label: str = ""
    device: str
    feature: str
    command: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    #: Re-attempt this many times on a driver/transport failure. Does not
    #: retry a safety denial or a validation error -- those are wrong plans,
    #: not flaky hardware, and retrying them would just waste the interlock's time.
    repeat: int = 1
    #: Move on to the next step instead of failing the run.
    continue_on_error: bool = False
    #: Block this step (not the whole run) until a human answers, up to this
    #: many seconds. Omit it and the run parks instead -- see `ExperimentManager`.
    wait_for_approval_s: float | None = None

    def resolved_label(self, index: int) -> str:
        return self.label or f"step{index + 1}_{self.device}.{self.feature}.{self.command}"


class Protocol(BaseModel):
    """A named, versioned procedure: metadata plus an ordered list of steps."""

    model_config = ConfigDict(extra="forbid")

    name: str = "protocol"
    description: str = ""
    version: str = "1.0"
    #: Defaults an operator may override at `experiment.start` time.
    variables: dict[str, Any] = Field(default_factory=dict)
    steps: list[ProtocolStep] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> Protocol:
        text = Path(path).expanduser().read_text(encoding="utf-8")
        return cls.model_validate(yaml.safe_load(text) or {})

    def resolve_args(
        self, step: ProtocolStep, results: dict[str, dict[str, Any]], variables: dict[str, Any]
    ) -> dict[str, Any]:
        """Substitute `${...}` references against prior results and variables.

        `steps` is a reserved top-level name in the reference path
        (`${steps.autofocus.result.z_um}`); everything else resolves directly
        against a run variable, so `${exposure_ms}` needs no prefix.
        """
        scope: dict[str, Any] = {**variables, "steps": results}
        return _resolve(step.args, scope)

    def validate_against(self, gateway: Any) -> list[str]:
        """As much static checking as is possible before a run has any results.

        Checks that every device/feature/command exists and that every
        *literal* argument satisfies the command's schema. An argument that is
        a `${...}` reference cannot be checked yet -- its value depends on a
        step that has not run -- so its presence is confirmed and its type is
        not, honestly, rather than pretending. Returns a list of problems;
        empty means "nothing detectable from here", not "guaranteed to run".
        """
        problems: list[str] = []
        known_labels: set[str] = set()
        for index, step in enumerate(self.steps):
            label = step.resolved_label(index)
            if label in known_labels:
                problems.append(f"step {index + 1} ({label!r}): duplicate step label")
            known_labels.add(label)
            try:
                device = gateway.device(step.device)
            except Exception as exc:  # noqa: BLE001 - reported as a validation problem, not raised
                problems.append(f"step {index + 1} ({label!r}): {exc}")
                continue
            try:
                _, cmd = device.resolve(step.feature, step.command)
            except Exception as exc:  # noqa: BLE001 - reported as a validation problem, not raised
                problems.append(f"step {index + 1} ({label!r}): {exc}")
                continue
            by_name = {p.name: p for p in cmd.parameters}
            unknown = set(step.args) - set(by_name)
            if unknown:
                problems.append(
                    f"step {index + 1} ({label!r}): unknown parameter(s) {sorted(unknown)}; "
                    f"expected {sorted(by_name)}"
                )
            for pname, pdef in by_name.items():
                if pname not in step.args:
                    if pdef.required and pdef.default is None:
                        problems.append(
                            f"step {index + 1} ({label!r}): missing required parameter {pname!r}"
                        )
                    continue
                value = step.args[pname]
                if isinstance(value, str) and _REF.search(value):
                    continue  # resolved at run time; cannot be checked now
                try:
                    pdef.validate_value(value)
                except ValidationError as exc:
                    problems.append(f"step {index + 1} ({label!r}): {exc.message}")
            for match in _REF.finditer(str(step.args)):
                ref = match.group(1)
                if ref.startswith("steps."):
                    ref_label = ref.split(".", 2)[1]
                    if ref_label not in known_labels:
                        problems.append(
                            f"step {index + 1} ({label!r}): references step "
                            f"{ref_label!r}, which has not run by this point"
                        )
        return problems


def _lookup(path: str, scope: dict[str, Any]) -> Any:
    node: Any = scope
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ValidationError(
                f"unresolved reference '${{{path}}}': no {part!r} here",
                reference=path,
            )
        node = node[part]
    return node


def _resolve(value: Any, scope: dict[str, Any]) -> Any:
    if isinstance(value, str):
        whole = _REF.fullmatch(value)
        if whole:
            return _lookup(whole.group(1), scope)  # preserves the referenced type
        return _REF.sub(lambda m: str(_lookup(m.group(1), scope)), value)
    if isinstance(value, dict):
        return {k: _resolve(v, scope) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, scope) for v in value]
    return value
