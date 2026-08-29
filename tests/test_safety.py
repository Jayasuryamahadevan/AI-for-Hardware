"""The safety kernel: policy, envelope and simulation gates."""

from __future__ import annotations

import pytest

from labbench.core.capability import Command, Constraint, Feature, Hazard, Parameter, Reversibility
from labbench.core.device import Device, DeviceDescriptor, SimulationResult
from labbench.core.errors import ApprovalRequired, SafetyViolation
from labbench.core.safety import AutonomyLevel, Effect, PolicyRule, SafetyKernel, SafetyPolicy


class _FakeDevice(Device):
    """A minimal Device whose simulate() outcome is controlled by the test."""

    def __init__(self, descriptor: DeviceDescriptor, sim: SimulationResult | None = None) -> None:
        super().__init__(descriptor)
        self._sim = sim

    def _features(self):
        return [
            Feature(
                identifier="F",
                commands=[
                    Command(name="noop", hazard=Hazard.NONE),
                    Command(name="benign", hazard=Hazard.BENIGN),
                    Command(name="move", hazard=Hazard.MOTION,
                            parameters=[Parameter(name="x", constraint=Constraint(minimum=0, maximum=100))]),
                    Command(name="consume", hazard=Hazard.SAMPLE,
                            reversibility=Reversibility.IRREVERSIBLE),
                    Command(name="culture", hazard=Hazard.BIOLOGICAL),
                    Command(name="not_simulatable", hazard=Hazard.SAMPLE, simulatable=False),
                ],
            )
        ]

    async def _read(self, feature, name):
        return None

    async def _invoke(self, feature, command, args, ctx):
        return {}

    async def _simulate(self, feature, command, args):
        if self._sim is not None:
            return self._sim
        return await super()._simulate(feature, command, args)


def make_device(
    sim: SimulationResult | None = None, *, labels: dict[str, str] | None = None,
) -> _FakeDevice:
    dev = _FakeDevice(DeviceDescriptor(id="dev1", simulated=True, labels=labels or {}), sim=sim)
    dev._state = dev._state.__class__.IDLE
    return dev


class TestHazardCeiling:
    async def test_reads_always_allowed(self):
        kernel = SafetyKernel(SafetyPolicy(autonomy=AutonomyLevel.MANUAL))
        decision = await kernel.authorize(make_device(), "F", "noop", {})
        assert decision.allowed

    async def test_hazard_above_ceiling_needs_approval(self):
        kernel = SafetyKernel(SafetyPolicy(autonomy=AutonomyLevel.ASSISTED))  # ceiling: benign
        decision = await kernel.authorize(make_device(), "F", "move", {"x": 1})
        assert decision.effect is Effect.REQUIRE_APPROVAL

    async def test_hazard_within_ceiling_allowed(self):
        kernel = SafetyKernel(SafetyPolicy(
            autonomy=AutonomyLevel.BOUNDED,  # ceiling: sample
            require_simulation_at_or_above=None,
        ))
        decision = await kernel.authorize(
            make_device(sim=SimulationResult(feasible=True, fidelity="kinematic")),
            "F", "move", {"x": 1},
        )
        assert decision.allowed

    async def test_biological_always_needs_approval_regardless_of_autonomy(self):
        kernel = SafetyKernel(SafetyPolicy(autonomy=AutonomyLevel.SUPERVISED))
        decision = await kernel.authorize(make_device(), "F", "culture", {})
        assert decision.effect is Effect.REQUIRE_APPROVAL
        assert any("biological" in r for r in decision.reasons)


class TestLabelHazardFloors:
    """A device label (biosafety, containment, ...) raising a command's
    effective hazard for gating -- see `SafetyPolicy.label_hazard_floors`."""

    def _policy(self, **overrides) -> SafetyPolicy:
        defaults = {
            "autonomy": AutonomyLevel.FULL,  # ceiling would otherwise allow everything
            "label_hazard_floors": {"biosafety:BSL3": Hazard.BIOLOGICAL},
        }
        defaults.update(overrides)
        return SafetyPolicy(**defaults)

    async def test_benign_command_on_a_labelled_device_is_escalated(self):
        kernel = SafetyKernel(self._policy())
        device = make_device(labels={"biosafety": "BSL3"})
        decision = await kernel.authorize(device, "F", "benign", {})
        assert decision.effect is Effect.REQUIRE_APPROVAL
        assert decision.hazard is Hazard.BIOLOGICAL
        assert any("hazard floor" in r for r in decision.reasons)

    async def test_same_command_on_an_unlabelled_device_is_not_escalated(self):
        kernel = SafetyKernel(self._policy())
        decision = await kernel.authorize(make_device(), "F", "benign", {})
        assert decision.allowed
        assert decision.hazard is Hazard.BENIGN

    async def test_a_pure_read_is_never_escalated(self):
        # Reads change nothing about the sample the label is describing a risk
        # to; escalating them would break the "reads are free" invariant every
        # other read-only tool call in the gateway relies on.
        kernel = SafetyKernel(self._policy())
        device = make_device(labels={"biosafety": "BSL3"})
        decision = await kernel.authorize(device, "F", "noop", {})
        assert decision.allowed
        assert decision.hazard is Hazard.NONE

    async def test_a_floor_lower_than_the_commands_own_hazard_does_nothing(self):
        policy = self._policy(label_hazard_floors={"biosafety:BSL1": Hazard.BENIGN})
        kernel = SafetyKernel(policy)
        # "consume" is intrinsically Hazard.SAMPLE, well above BSL1's benign floor.
        device = make_device(labels={"biosafety": "BSL1"})
        decision = await kernel.authorize(
            device, "F", "consume", {},
        )
        assert decision.hazard is Hazard.SAMPLE

    async def test_unrecognised_label_value_is_ignored(self):
        kernel = SafetyKernel(self._policy())
        device = make_device(labels={"biosafety": "none"})
        decision = await kernel.authorize(device, "F", "benign", {})
        assert decision.allowed


class TestEnvelope:
    async def test_policy_rule_tightens_constraint(self):
        policy = SafetyPolicy(
            autonomy=AutonomyLevel.SUPERVISED,
            rules=[PolicyRule(command="move", limits={"x": Constraint(maximum=10)})],
        )
        kernel = SafetyKernel(policy)
        decision = await kernel.authorize(
            make_device(sim=SimulationResult(feasible=True)), "F", "move", {"x": 50},
        )
        assert decision.effect is Effect.DENY
        assert any("50" in r for r in decision.reasons)

    async def test_deny_rule_wins(self):
        policy = SafetyPolicy(
            autonomy=AutonomyLevel.FULL,
            rules=[PolicyRule(name="block-move", command="move", effect=Effect.DENY)],
        )
        kernel = SafetyKernel(policy)
        decision = await kernel.authorize(make_device(), "F", "move", {"x": 1})
        assert decision.effect is Effect.DENY

    async def test_rate_limit(self):
        policy = SafetyPolicy(
            autonomy=AutonomyLevel.FULL,
            require_simulation_at_or_above=None,
            rules=[PolicyRule(name="cap", command="benign", max_per_window=1, window_s=60)],
        )
        kernel = SafetyKernel(policy)
        dev = make_device()
        first = await kernel.authorize(dev, "F", "benign", {})
        assert first.allowed
        second = await kernel.authorize(dev, "F", "benign", {})
        assert second.effect is Effect.DENY
        assert "rate limit" in second.reasons[0]


class TestSimulationGate:
    async def test_no_twin_forces_approval_above_threshold(self):
        kernel = SafetyKernel(SafetyPolicy(autonomy=AutonomyLevel.FULL))
        decision = await kernel.authorize(make_device(), "F", "consume", {})
        assert decision.effect is Effect.REQUIRE_APPROVAL
        assert decision.simulation is not None
        assert decision.simulation.fidelity == "none"

    async def test_simulation_violation_denies_by_default(self):
        kernel = SafetyKernel(SafetyPolicy(autonomy=AutonomyLevel.FULL))
        sim = SimulationResult(feasible=False, fidelity="high", violations=["would crash"])
        decision = await kernel.authorize(make_device(sim=sim), "F", "consume", {})
        assert decision.effect is Effect.DENY

    async def test_simulation_violation_can_escalate_instead_of_deny(self):
        policy = SafetyPolicy(autonomy=AutonomyLevel.FULL, block_on_simulation_violation=False)
        kernel = SafetyKernel(policy)
        sim = SimulationResult(feasible=False, fidelity="high", violations=["marginal"])
        decision = await kernel.authorize(make_device(sim=sim), "F", "consume", {})
        assert decision.effect is Effect.REQUIRE_APPROVAL

    async def test_uncheckable_command_is_not_gated_by_simulation(self):
        # simulatable=False short-circuits Device.simulate() to a "none"
        # fidelity result *without* the safety kernel needing to know that;
        # it should still escalate exactly as the "no twin" case does.
        kernel = SafetyKernel(SafetyPolicy(autonomy=AutonomyLevel.FULL))
        decision = await kernel.authorize(make_device(), "F", "not_simulatable", {})
        assert decision.effect is Effect.REQUIRE_APPROVAL


class TestIrreversibility:
    async def test_irreversible_needs_approval_even_within_ceiling(self):
        policy = SafetyPolicy(autonomy=AutonomyLevel.FULL, approve_irreversible=True)
        kernel = SafetyKernel(policy)
        sim = SimulationResult(feasible=True, fidelity="high")
        decision = await kernel.authorize(make_device(sim=sim), "F", "consume", {})
        assert decision.effect is Effect.REQUIRE_APPROVAL
        assert any("irreversible" in r for r in decision.reasons)

    async def test_approve_irreversible_false_allows_it(self):
        policy = SafetyPolicy(autonomy=AutonomyLevel.FULL, approve_irreversible=False)
        kernel = SafetyKernel(policy)
        sim = SimulationResult(feasible=True, fidelity="high")
        decision = await kernel.authorize(make_device(sim=sim), "F", "consume", {})
        assert decision.allowed


class TestEnforce:
    def test_enforce_raises_for_deny(self):
        from labbench.core.safety import Decision

        decision = Decision(effect=Effect.DENY, device="d", feature="F", command="c",
                             hazard=Hazard.NONE, autonomy=AutonomyLevel.MANUAL, reasons=["no"])
        with pytest.raises(SafetyViolation):
            SafetyKernel.enforce(decision)

    def test_enforce_raises_approval_required(self):
        from labbench.core.safety import Decision

        decision = Decision(effect=Effect.REQUIRE_APPROVAL, device="d", feature="F", command="c",
                             hazard=Hazard.NONE, autonomy=AutonomyLevel.MANUAL,
                             approval_prompt="please")
        with pytest.raises(ApprovalRequired):
            SafetyKernel.enforce(decision)
