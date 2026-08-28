"""Capability model: parameters, constraints, commands, features."""

from __future__ import annotations

import pytest

from labbench.core.capability import Command, Constraint, Feature, Parameter, Precondition
from labbench.core.errors import ConstraintViolation, ValidationError


class TestConstraint:
    def test_range_ok(self):
        Constraint(minimum=0, maximum=10).check(5, path="x")

    def test_range_violation(self):
        with pytest.raises(ConstraintViolation):
            Constraint(minimum=0, maximum=10).check(15, path="x")

    def test_exclusive_bounds(self):
        c = Constraint(exclusive_minimum=0, exclusive_maximum=10)
        c.check(5, path="x")
        with pytest.raises(ConstraintViolation):
            c.check(0, path="x")
        with pytest.raises(ConstraintViolation):
            c.check(10, path="x")

    def test_enum(self):
        c = Constraint(enum=["a", "b"])
        c.check("a", path="x")
        with pytest.raises(ConstraintViolation):
            c.check("c", path="x")

    def test_pattern(self):
        c = Constraint(pattern=r"[A-Z]\d+")
        c.check("A1", path="x")
        with pytest.raises(ConstraintViolation):
            c.check("a1", path="x")

    def test_multiple_of(self):
        c = Constraint(multiple_of=0.5)
        c.check(1.5, path="x")
        with pytest.raises(ConstraintViolation):
            c.check(1.3, path="x")

    def test_bool_is_never_a_number(self):
        # bool is a subclass of int in Python; the constraint check must not
        # treat True/False as 1/0 when bounds are numeric.
        c = Constraint(minimum=0, maximum=10)
        c.check(True, path="x")  # must not raise, and must not compare as 1

    def test_to_json_schema_only_includes_set_fields(self):
        schema = Constraint(minimum=0, maximum=10).to_json_schema()
        assert schema == {"minimum": 0, "maximum": 10}


class TestParameter:
    def test_number_accepts_int_without_change(self):
        # "number" already accepts int in its type tuple; no coercion needed.
        p = Parameter(name="x", type="number")
        assert p.validate_value(5) == 5

    def test_integer_coerces_whole_float(self):
        p = Parameter(name="x", type="integer")
        assert p.validate_value(5.0) == 5
        assert isinstance(p.validate_value(5.0), int)

    def test_integer_rejects_fractional_float(self):
        p = Parameter(name="x", type="integer")
        with pytest.raises(ValidationError):
            p.validate_value(5.5)

    def test_validate_value_rejects_bool_for_number(self):
        p = Parameter(name="x", type="number")
        with pytest.raises(ValidationError):
            p.validate_value(True)

    def test_validate_value_wrong_type(self):
        p = Parameter(name="x", type="string")
        with pytest.raises(ValidationError):
            p.validate_value(5)

    def test_nested_object(self):
        p = Parameter(
            name="pos", type="object",
            properties=[
                Parameter(name="x", unit="um"),
                Parameter(name="y", unit="um", required=False, default=0.0),
            ],
        )
        out = p.validate_value({"x": 1.0})
        assert out == {"x": 1.0, "y": 0.0}

    def test_nested_object_missing_required(self):
        p = Parameter(name="pos", type="object", properties=[Parameter(name="x")])
        with pytest.raises(ValidationError):
            p.validate_value({})

    def test_array_of_typed_items(self):
        p = Parameter(name="xs", type="array", items=Parameter(name="item", type="integer"))
        assert p.validate_value([1, 2, 3]) == [1, 2, 3]
        with pytest.raises(ValidationError):
            p.validate_value(["a"])

    def test_unit_is_rendered_into_schema_description(self):
        p = Parameter(name="z", unit="um", description="height")
        schema = p.to_json_schema()
        assert "[unit: um]" in schema["description"]


class TestCommand:
    def test_validate_args_unknown_parameter(self):
        cmd = Command(name="move", parameters=[Parameter(name="x_um", unit="um")])
        with pytest.raises(ValidationError):
            cmd.validate_args({"y_um": 1.0})

    def test_validate_args_missing_required(self):
        cmd = Command(name="move", parameters=[Parameter(name="x_um", unit="um")])
        with pytest.raises(ValidationError):
            cmd.validate_args({})

    def test_validate_args_applies_default(self):
        cmd = Command(
            name="move",
            parameters=[Parameter(name="dx_um", unit="um", required=False, default=0.0)],
        )
        assert cmd.validate_args({}) == {"dx_um": 0.0}

    def test_duplicate_parameter_names_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            Command(name="bad", parameters=[Parameter(name="x"), Parameter(name="x")])

    def test_input_schema_required_list(self):
        cmd = Command(
            name="move",
            parameters=[
                Parameter(name="x_um", unit="um"),
                Parameter(name="dx_um", required=False, default=0.0),
            ],
        )
        schema = cmd.input_schema()
        assert schema["required"] == ["x_um"]
        assert schema["additionalProperties"] is False


class TestPrecondition:
    def test_is_true(self):
        pc = Precondition(property="homed", operator="is_true")
        ok, _ = pc.evaluate({"homed": True})
        assert ok
        ok, why = pc.evaluate({"homed": False})
        assert not ok and why

    def test_unknown_property_fails_closed(self):
        pc = Precondition(property="nope", operator="is_true")
        ok, why = pc.evaluate({})
        assert not ok
        assert "unknown" in why

    @pytest.mark.parametrize(
        "operator,value,actual,expected",
        [
            ("==", 5, 5, True), ("!=", 5, 6, True), ("<", 5, 4, True),
            ("<=", 5, 5, True), (">", 5, 6, True), (">=", 5, 5, True),
            ("in", [1, 2], 1, True), ("not_in", [1, 2], 3, True),
        ],
    )
    def test_operators(self, operator, value, actual, expected):
        pc = Precondition(property="p", operator=operator, value=value)
        ok, _ = pc.evaluate({"p": actual})
        assert ok is expected


class TestFeature:
    def test_fqid(self):
        f = Feature(identifier="Motion", namespace="org.example", version="2.0")
        assert f.fqid == "org.example/Motion/v2.0"

    def test_command_and_property_lookup(self):
        f = Feature(
            identifier="X",
            commands=[Command(name="go")],
        )
        assert f.command("go") is not None
        assert f.command("missing") is None
