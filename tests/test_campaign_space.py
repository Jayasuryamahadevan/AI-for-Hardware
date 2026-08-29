"""The search space: dimensions, encoding, containment, and initial designs."""

from __future__ import annotations

import numpy as np
import pytest

from labbench.campaign.space import Dimension, ParameterSpace
from labbench.core.errors import ValidationError


class TestDimensionValidation:
    def test_categorical_needs_at_least_two_choices(self):
        with pytest.raises(ValueError, match="categorical needs"):
            Dimension(name="ch", type="categorical", choices=["only-one"])

    def test_categorical_cannot_be_log(self):
        with pytest.raises(ValueError, match="cannot be log"):
            Dimension(name="ch", type="categorical", choices=["a", "b"], log=True)

    def test_numeric_needs_both_bounds(self):
        with pytest.raises(ValueError, match="need both low and high"):
            Dimension(name="z", type="continuous", low=0.0)

    def test_high_must_exceed_low(self):
        with pytest.raises(ValueError, match="must exceed"):
            Dimension(name="z", type="continuous", low=10.0, high=5.0)

    def test_log_axis_needs_positive_low(self):
        with pytest.raises(ValueError, match="needs low > 0"):
            Dimension(name="z", type="continuous", low=0.0, high=10.0, log=True)


class TestDimensionShape:
    def test_categorical_width_is_choice_count(self):
        dim = Dimension(name="ch", type="categorical", choices=["a", "b", "c"])
        assert dim.width == 3

    def test_numeric_width_is_one(self):
        assert Dimension(name="z", low=0.0, high=1.0).width == 1

    def test_to_parameter_numeric_carries_bounds_and_unit(self):
        dim = Dimension(name="z_um", low=0.0, high=190.0, unit="um")
        param = dim.to_parameter()
        assert param.unit == "um"
        assert param.constraint.minimum == 0.0
        assert param.constraint.maximum == 190.0

    def test_to_parameter_categorical_carries_enum(self):
        dim = Dimension(name="ch", type="categorical", choices=["dapi", "gfp"])
        param = dim.to_parameter()
        assert param.type == "string"
        assert param.constraint.enum == ["dapi", "gfp"]

    def test_extremes_numeric_linear(self):
        dim = Dimension(name="z", low=0.0, high=10.0)
        assert dim.extremes() == [0.0, 5.0, 10.0]

    def test_extremes_numeric_log_uses_geometric_mean(self):
        dim = Dimension(name="e", low=1.0, high=100.0, log=True)
        low, mid, high = dim.extremes()
        assert low == 1.0 and high == 100.0
        assert mid == pytest.approx(10.0)

    def test_extremes_categorical_is_every_choice(self):
        dim = Dimension(name="ch", type="categorical", choices=["a", "b"])
        assert dim.extremes() == ["a", "b"]


class TestDimensionValues:
    def test_quantize_clamps_to_bounds(self):
        dim = Dimension(name="z", low=0.0, high=10.0)
        assert dim.quantize(-5.0) == 0.0
        assert dim.quantize(15.0) == 10.0

    def test_quantize_snaps_to_step(self):
        dim = Dimension(name="z", low=0.0, high=10.0, step=2.5)
        assert dim.quantize(3.1) == 2.5

    def test_quantize_integer_rounds(self):
        dim = Dimension(name="n", type="integer", low=0.0, high=10.0)
        assert dim.quantize(3.6) == 4
        assert isinstance(dim.quantize(3.6), int)

    def test_check_raises_outside_bounds(self):
        dim = Dimension(name="z", low=0.0, high=10.0)
        with pytest.raises(ValidationError):
            dim.check(11.0)

    def test_encode_decode_round_trip_linear(self):
        dim = Dimension(name="z", low=0.0, high=10.0)
        assert dim.decode(dim.encode(3.5)) == pytest.approx(3.5)

    def test_encode_decode_round_trip_log(self):
        dim = Dimension(name="e", low=1.0, high=1000.0, log=True)
        assert dim.decode(dim.encode(31.6)) == pytest.approx(31.6, rel=1e-2)

    def test_encode_categorical_is_one_hot(self):
        dim = Dimension(name="ch", type="categorical", choices=["a", "b", "c"])
        assert dim.encode("b") == [0.0, 1.0, 0.0]

    def test_encode_unknown_choice_raises(self):
        dim = Dimension(name="ch", type="categorical", choices=["a", "b"])
        with pytest.raises(ValidationError):
            dim.encode("nope")

    def test_decode_categorical_is_argmax(self):
        dim = Dimension(name="ch", type="categorical", choices=["a", "b", "c"])
        assert dim.decode([0.1, 0.8, 0.2]) == "b"


SPACE = ParameterSpace(dimensions=[
    Dimension(name="z_um", low=0.0, high=190.0, unit="um"),
    Dimension(name="exposure_ms", low=5.0, high=200.0, unit="ms", log=True),
    Dimension(name="channel", type="categorical", choices=["dapi", "gfp", "rfp"]),
])


class TestParameterSpace:
    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError, match="duplicate dimension"):
            ParameterSpace(dimensions=[
                Dimension(name="z", low=0.0, high=1.0),
                Dimension(name="z", low=0.0, high=1.0),
            ])

    def test_needs_at_least_one_dimension(self):
        with pytest.raises(ValueError, match="at least one dimension"):
            ParameterSpace(dimensions=[])

    def test_width_sums_dimension_widths(self):
        assert SPACE.width == 1 + 1 + 3

    def test_dimension_lookup(self):
        assert SPACE.dimension("channel").type == "categorical"
        assert SPACE.dimension("nope") is None

    def test_validate_point_rejects_unknown_dimension(self):
        with pytest.raises(ValidationError, match="unknown dimension"):
            SPACE.validate_point({"z_um": 1.0, "exposure_ms": 10.0, "channel": "dapi", "x": 1})

    def test_validate_point_rejects_missing_dimension(self):
        with pytest.raises(ValidationError, match="missing dimension"):
            SPACE.validate_point({"z_um": 1.0})

    def test_validate_point_quantizes_numeric_values(self):
        out = SPACE.validate_point({"z_um": 1.23456789, "exposure_ms": 10.0, "channel": "gfp"})
        assert out["z_um"] == round(1.23456789, 10)

    def test_contains_true_for_a_legal_point(self):
        assert SPACE.contains({"z_um": 100.0, "exposure_ms": 10.0, "channel": "gfp"})

    def test_contains_false_out_of_bounds(self):
        assert not SPACE.contains({"z_um": 999.0, "exposure_ms": 10.0, "channel": "gfp"})

    def test_encode_decode_round_trip(self):
        point = {"z_um": 42.0, "exposure_ms": 30.0, "channel": "rfp"}
        decoded = SPACE.decode(SPACE.encode(point))
        assert decoded["z_um"] == pytest.approx(42.0)
        assert decoded["exposure_ms"] == pytest.approx(30.0, rel=1e-2)
        assert decoded["channel"] == "rfp"

    def test_encode_many_shape(self):
        points = [{"z_um": 1.0, "exposure_ms": 5.0, "channel": "dapi"},
                  {"z_um": 2.0, "exposure_ms": 6.0, "channel": "gfp"}]
        assert SPACE.encode_many(points).shape == (2, SPACE.width)

    def test_encode_many_of_no_points_has_the_right_width(self):
        assert SPACE.encode_many([]).shape == (0, SPACE.width)

    def test_to_json_schema_lists_every_dimension_as_required(self):
        schema = SPACE.to_json_schema()
        assert set(schema["required"]) == {"z_um", "exposure_ms", "channel"}
        assert schema["additionalProperties"] is False


class TestDesigns:
    def test_random_returns_n_legal_points(self):
        rng = np.random.default_rng(0)
        points = SPACE.random(rng, 5)
        assert len(points) == 5
        assert all(SPACE.contains(p) for p in points)

    def test_latin_hypercube_covers_each_axis(self):
        rng = np.random.default_rng(0)
        points = SPACE.latin_hypercube(rng, 8)
        assert len(points) == 8
        assert all(SPACE.contains(p) for p in points)
        # Space-filling: the z_um values should spread across most of the range,
        # not cluster the way 8 independent uniform draws sometimes do.
        z_values = sorted(p["z_um"] for p in points)
        assert z_values[0] < 190.0 * 0.25
        assert z_values[-1] > 190.0 * 0.75

    def test_latin_hypercube_is_reproducible_with_a_seeded_rng(self):
        points_a = SPACE.latin_hypercube(np.random.default_rng(3), 4)
        points_b = SPACE.latin_hypercube(np.random.default_rng(3), 4)
        assert points_a == points_b

    def test_grid_is_a_full_factorial(self):
        small = ParameterSpace(dimensions=[
            Dimension(name="a", low=0.0, high=1.0),
            Dimension(name="b", type="categorical", choices=["x", "y"]),
        ])
        points = small.grid(per_dim=3)
        assert len(points) == 3 * 2
        assert all(small.contains(p) for p in points)
