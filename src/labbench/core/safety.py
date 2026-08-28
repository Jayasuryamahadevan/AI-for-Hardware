"""The safety kernel: the one component an agent cannot route around.

Design follows the Safe-SDL framing of the "syntax-to-safety gap" — an LLM
emits commands that are perfectly well-formed and physically catastrophic. The
answer is not a better prompt; it is an interposed layer that (a) knows the
declared operating envelope, (b) simulates before it actuates, and (c) escalates
to a human whenever the action's hazard exceeds the granted autonomy.

Three gates, in order, cheapest first:

  1. Policy     — is this actor allowed to run this command at all?
  2. Envelope   — do the arguments sit inside the declared ODD?
  3. Simulation — does a model of the device say the trajectory stays safe?

Only after all three does anything physical happen. That ordering matters:
the expensive check runs last, and the irreversible one runs never if an
earlier gate says no.
"""

from __future__ import annotations

import fnmatch
import time
from collections import deque
from enum import Enum, IntEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .capability import Command, Constraint, Hazard, Reversibility
from .device import Device, SimulationResult
from .errors import ApprovalRequired, ConstraintViolation, SafetyViolation


class AutonomyLevel(IntEnum):
    """Graduated autonomy, after the Safe-SDL six-level ladder.

    A session is granted a level; the level caps which hazard classes may run
    without a human signature. Levels are meant to be *earned* — raise only
    after an incident-free record on the tier below.
    """

    MANUAL = 0        # agent advises; every state change needs a signature
    ASSISTED = 1      # per-step approval for anything that moves
    HUMAN_ON_LOOP = 2 # routine motion autonomous, human watching
    BOUNDED = 3       # autonomous inside a strictly declared ODD
    SUPERVISED = 4    # extended autonomy, periodic oversight
    FULL = 5          # reserved; not grantable by configuration


#: Highest hazard class that may execute unattended at each autonomy level.
DEFAULT_HAZARD_CEILING: dict[AutonomyLevel, Hazard] = {
    AutonomyLevel.MANUAL: Hazard.NONE,
    AutonomyLevel.ASSISTED: Hazard.BENIGN,
    AutonomyLevel.HUMAN_ON_LOOP: Hazard.MOTION,
    AutonomyLevel.BOUNDED: Hazard.SAMPLE,
    AutonomyLevel.SUPERVISED: Hazard.CHEMICAL,
    AutonomyLevel.FULL: Hazard.RADIOLOGICAL,
}

#: Hazards that always require an explicit human signature, at any level.
#: Containment and ionising-radiation decisions are not delegated to a model.
ALWAYS_APPROVE: set[Hazard] = {Hazard.BIOLOGICAL, Hazard.RADIOLOGICAL}


class Effect(str, Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class PolicyRule(BaseModel):
    """One matcher→effect rule. Glob matching, first-match-wins by `priority`.

    Rules can also *tighten* parameter constraints, which is how a site says
    "this stage may travel 0–200 µm, but on this project never above 80".
    """

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    device: str = "*"
    feature: str = "*"
    command: str = "*"
    #: Match only commands carrying all of these tags.
    tags: set[str] = Field(default_factory=set)
    #: Match only at or above this hazard class.
    min_hazard: Hazard | None = None
    actor: str = "*"
    effect: Effect = Effect.ALLOW
    reason: str = ""
    priority: int = 100
    #: Per-parameter constraint overrides applied when this rule matches.
    limits: dict[str, Constraint] = Field(default_factory=dict)
    #: Max invocations per window; None disables rate limiting.
    max_per_window: int | None = None
    window_s: float = 3600.0

    def matches(
        self, *, device: str, feature: str, command: str, actor: str, cmd: Command
    ) -> bool:
        if not fnmatch.fnmatch(device, self.device):
            return False
        if not fnmatch.fnmatch(feature, self.feature):
            return False
        if not fnmatch.fnmatch(command, self.command):
            return False
        if not fnmatch.fnmatch(actor, self.actor):
            return False
        if self.tags and not self.tags.issubset(cmd.tags):
            return False
        return self.min_hazard is None or cmd.hazard.rank >= self.min_hazard.rank


class SafetyPolicy(BaseModel):
    """The site's declared operating design domain.

    Loaded from YAML so it is reviewable, diffable and version-controlled —
    a safety envelope that lives only in code is one nobody audits.
    """

    model_config = ConfigDict(extra="forbid")

    autonomy: AutonomyLevel = AutonomyLevel.HUMAN_ON_LOOP
    #: Override the hazard ceiling implied by the autonomy level.
    hazard_ceiling: Hazard | None = None
    #: Irreversible commands always need a signature, regardless of hazard.
    approve_irreversible: bool = True
    #: Refuse commands whose driver offers no digital twin above this hazard.
    require_simulation_at_or_above: Hazard | None = Hazard.SAMPLE
    #: Refuse when the simulation reports violations (vs. merely warning).
    block_on_simulation_violation: bool = True
    #: Wall-clock windows in which autonomous operation is permitted, "HH:MM-HH:MM".
    allowed_hours: list[str] = Field(default_factory=list)
    rules: list[PolicyRule] = Field(default_factory=list)

    def ceiling(self) -> Hazard:
        return self.hazard_ceiling or DEFAULT_HAZARD_CEILING[self.autonomy]


class Decision(BaseModel):
    """The kernel's verdict. Always logged, whatever the outcome."""

    model_config = ConfigDict(extra="forbid")

    effect: Effect
    device: str
    feature: str
    command: str
    hazard: Hazard
    autonomy: AutonomyLevel
    reasons: list[str] = Field(default_factory=list)
    matched_rules: list[str] = Field(default_factory=list)
    simulation: SimulationResult | None = None
    #: Human-readable summary the agent should surface verbatim when asking.
    approval_prompt: str = ""

    @property
    def allowed(self) -> bool:
        return self.effect is Effect.ALLOW


class _RateWindow:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str, limit: int, window: float) -> tuple[bool, int]:
        now = time.time()
        q = self._hits.setdefault(key, deque())
        while q and now - q[0] > window:
            q.popleft()
        return len(q) < limit, len(q)

    def record(self, key: str) -> None:
        self._hits.setdefault(key, deque()).append(time.time())


class SafetyKernel:
    """Evaluates every command request against the policy.

    Stateless with respect to devices; the only state it keeps is rate-limit
    history and the approval record, both of which are cheap.
    """

    def __init__(self, policy: SafetyPolicy | None = None) -> None:
        self.policy = policy or SafetyPolicy()
        self._rates = _RateWindow()

    def set_policy(self, policy: SafetyPolicy) -> None:
        self.policy = policy

    # -- gate 1: policy ---------------------------------------------------

    def _apply_rules(
        self, *, device: str, feature: str, command: str, actor: str, cmd: Command
    ) -> tuple[Effect | None, list[str], list[str], dict[str, Constraint]]:
        reasons: list[str] = []
        matched: list[str] = []
        limits: dict[str, Constraint] = {}
        effect: Effect | None = None
        for rule in sorted(self.policy.rules, key=lambda r: r.priority):
            if not rule.matches(
                device=device, feature=feature, command=command, actor=actor, cmd=cmd
            ):
                continue
            label = rule.name or f"{rule.device}/{rule.feature}/{rule.command}"
            matched.append(label)
            limits.update(rule.limits)
            if rule.max_per_window is not None:
                key = f"{label}:{device}.{feature}.{command}"
                ok, seen = self._rates.check(key, rule.max_per_window, rule.window_s)
                if not ok:
                    return (
                        Effect.DENY,
                        [(f"rate limit: rule {label!r} allows "
                          f"{rule.max_per_window} per {rule.window_s:.0f}s, {seen} used")],
                        matched, limits,
                    )
            if effect is None or rule.effect is Effect.DENY:
                effect = rule.effect
                if rule.reason:
                    reasons.append(f"rule {label!r}: {rule.reason}")
            if rule.effect is Effect.DENY:
                break
        return effect, reasons, matched, limits

    def _within_hours(self) -> tuple[bool, str]:
        if not self.policy.allowed_hours:
            return True, ""
        now = time.localtime()
        minutes = now.tm_hour * 60 + now.tm_min
        for window in self.policy.allowed_hours:
            start_s, _, end_s = window.partition("-")
            sh, _, sm = start_s.strip().partition(":")
            eh, _, em = end_s.strip().partition(":")
            start = int(sh) * 60 + int(sm or 0)
            end = int(eh) * 60 + int(em or 0)
            inside = start <= minutes <= end if start <= end else (
                minutes >= start or minutes <= end
            )
            if inside:
                return True, ""
        return False, (
            f"outside permitted autonomous hours {self.policy.allowed_hours}; "
            "human approval required"
        )

    # -- gate 2: envelope -------------------------------------------------

    @staticmethod
    def check_envelope(
        cmd: Command, args: dict[str, Any], limits: dict[str, Constraint]
    ) -> list[str]:
        """Apply policy-tightened limits on top of the driver's own constraints."""
        violations: list[str] = []
        for pname, constraint in limits.items():
            if pname not in args:
                continue
            try:
                constraint.check(args[pname], path=f"{cmd.name}.{pname}")
            except ConstraintViolation as exc:
                violations.append(exc.message)
        return violations

    # -- full evaluation --------------------------------------------------

    async def authorize(
        self,
        device: Device,
        feature: str,
        command: str,
        args: dict[str, Any],
        *,
        actor: str = "agent",
        approved_by: str | None = None,
        reason: str = "",
    ) -> Decision:
        _, cmd = device.resolve(feature, command)
        pol = self.policy
        reasons: list[str] = []

        effect, rule_reasons, matched, limits = self._apply_rules(
            device=device.id, feature=feature, command=command, actor=actor, cmd=cmd
        )
        reasons.extend(rule_reasons)

        decision = Decision(
            effect=Effect.ALLOW, device=device.id, feature=feature, command=command,
            hazard=cmd.hazard, autonomy=pol.autonomy, matched_rules=matched,
        )

        def finalize(eff: Effect) -> Decision:
            decision.effect = eff
            decision.reasons = reasons
            if eff is Effect.REQUIRE_APPROVAL:
                decision.approval_prompt = _approval_prompt(
                    device, feature, command, args, cmd, reasons, reason
                )
            return decision

        if effect is Effect.DENY:
            return finalize(Effect.DENY)

        # Reads never gate. Everything else does.
        if cmd.hazard is Hazard.NONE and not limits:
            return finalize(Effect.ALLOW)

        violations = self.check_envelope(cmd, args, limits)
        if violations:
            reasons.extend(violations)
            return finalize(Effect.DENY)

        needs_approval = effect is Effect.REQUIRE_APPROVAL
        ceiling = pol.ceiling()

        if cmd.hazard in ALWAYS_APPROVE:
            needs_approval = True
            reasons.append(
                f"hazard class {cmd.hazard.value!r} always requires a human signature"
            )
        elif cmd.hazard.rank > ceiling.rank:
            needs_approval = True
            reasons.append(
                f"hazard {cmd.hazard.value!r} exceeds the ceiling {ceiling.value!r} "
                f"for autonomy level {pol.autonomy.value} ({pol.autonomy.name})"
            )

        if pol.approve_irreversible and cmd.reversibility is Reversibility.IRREVERSIBLE:
            needs_approval = True
            reasons.append(f"{command!r} is irreversible")

        ok_hours, why = self._within_hours()
        if not ok_hours:
            needs_approval = True
            reasons.append(why)

        # -- gate 3: simulate before actuating -----------------------------
        thresh = pol.require_simulation_at_or_above
        if thresh is not None and cmd.hazard.rank >= thresh.rank:
            sim = await device.simulate(feature, command, args)
            decision.simulation = sim
            if sim.fidelity == "none":
                reasons.append(
                    f"no digital twin for {feature}.{command}; outcome unverifiable "
                    f"at hazard {cmd.hazard.value!r}"
                )
                needs_approval = True
            if not sim.feasible or sim.violations:
                reasons.extend(sim.violations or ["simulation reports the action is infeasible"])
                if pol.block_on_simulation_violation:
                    return finalize(Effect.DENY)
                needs_approval = True
            reasons.extend(f"simulation warning: {w}" for w in sim.warnings)

        if needs_approval:
            if approved_by:
                reasons.append(f"approved by {approved_by}")
                self._record_rate(matched, device.id, feature, command)
                return finalize(Effect.ALLOW)
            return finalize(Effect.REQUIRE_APPROVAL)

        self._record_rate(matched, device.id, feature, command)
        return finalize(Effect.ALLOW)

    def _record_rate(
        self, matched: list[str], device: str, feature: str, command: str
    ) -> None:
        for label in matched:
            self._rates.record(f"{label}:{device}.{feature}.{command}")

    @staticmethod
    def enforce(decision: Decision) -> None:
        """Turn a non-ALLOW decision into the right exception."""
        if decision.effect is Effect.DENY:
            raise SafetyViolation(
                f"blocked {decision.device}.{decision.feature}.{decision.command}: "
                + "; ".join(decision.reasons),
                decision=decision.model_dump(mode="json"),
            )
        if decision.effect is Effect.REQUIRE_APPROVAL:
            raise ApprovalRequired(
                decision.approval_prompt
                or f"{decision.command} requires human approval",
                decision=decision.model_dump(mode="json"),
            )


def _approval_prompt(
    device: Device, feature: str, command: str, args: dict[str, Any],
    cmd: Command, reasons: list[str], intent: str,
) -> str:
    lines = [
        f"APPROVAL REQUIRED — {device.descriptor.display_name or device.id}"
        + (" [SIMULATED DEVICE]" if device.descriptor.simulated else ""),
        f"  action     : {feature}.{command}",
        f"  arguments  : {args}",
        f"  hazard     : {cmd.hazard.value}   reversibility: {cmd.reversibility.value}",
    ]
    if cmd.duration_estimate_s >= 1:
        lines.append(f"  est. time  : {cmd.duration_estimate_s:.0f}s")
    if intent:
        lines.append(f"  stated why : {intent}")
    if reasons:
        lines.append("  because    :")
        lines.extend(f"    - {r}" for r in reasons)
    return "\n".join(lines)
