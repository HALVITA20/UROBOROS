
import math

import pytest

from boundary import (
    effective_sample_size,
    evaluate_boundary,
    weighted_quantile,
)
from models import BoundaryState


# ---------------------------------------------------------
# EFFECTIVE SAMPLE SIZE
# ---------------------------------------------------------

def test_ess_with_equal_weights():
    assert effective_sample_size([1.0, 1.0, 1.0, 1.0]) == pytest.approx(4.0)


def test_ess_with_zero_weights():
    assert effective_sample_size([0.0, 1.0, 1.0]) == pytest.approx(2.0)


def test_ess_with_only_zero_weights():
    assert effective_sample_size([0.0, 0.0, 0.0]) == 0.0


def test_ess_handles_very_large_weights():
    result = effective_sample_size([1e308, 1e308])

    assert math.isfinite(result)
    assert result == pytest.approx(2.0)


@pytest.mark.parametrize(
    "weights",
    [
        [-1.0, 1.0],
        [float("nan"), 1.0],
        [float("inf"), 1.0],
    ],
)
def test_ess_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        effective_sample_size(weights)


# ---------------------------------------------------------
# WEIGHTED QUANTILE
# ---------------------------------------------------------

def test_quantile_zero_ignores_zero_weight_observations():
    result = weighted_quantile(
        values=[1.0, 2.0, 3.0],
        weights=[0.0, 1.0, 1.0],
        quantile=0.0,
    )

    assert result == 2.0


def test_quantile_one_ignores_zero_weight_observations():
    result = weighted_quantile(
        values=[1.0, 2.0, 3.0],
        weights=[0.0, 1.0, 1.0],
        quantile=1.0,
    )

    assert result == 3.0


def test_quantile_uses_cumulative_weight():
    result = weighted_quantile(
        values=[1.0, 2.0, 3.0],
        weights=[1.0, 3.0, 1.0],
        quantile=0.5,
    )

    assert result == 2.0


def test_quantile_rejects_zero_total_weight():
    with pytest.raises(ValueError):
        weighted_quantile(
            values=[1.0, 2.0],
            weights=[0.0, 0.0],
            quantile=0.5,
        )


@pytest.mark.parametrize(
    "quantile",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.1,
        1.1,
    ],
)
def test_quantile_rejects_invalid_quantile(quantile):
    with pytest.raises(ValueError):
        weighted_quantile(
            values=[1.0, 2.0],
            weights=[1.0, 1.0],
            quantile=quantile,
        )


def test_quantile_rejects_non_finite_values():
    with pytest.raises(ValueError):
        weighted_quantile(
            values=[1.0, float("nan")],
            weights=[1.0, 1.0],
            quantile=0.5,
        )


def test_quantile_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        weighted_quantile(
            values=[1.0, 2.0],
            weights=[1.0],
            quantile=0.5,
        )


# ---------------------------------------------------------
# BOUNDARY PARAMETER VALIDATION
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "parameter",
    [
        "min_ess",
        "min_weight_sum",
        "positive_threshold",
        "negative_threshold",
        "lower_quantile",
        "upper_quantile",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_boundary_rejects_non_finite_parameters(
    parameter,
    invalid_value,
):
    arguments = {
        "predictions": [0.9],
        "weights": [1.0],
    }

    arguments[parameter] = invalid_value

    with pytest.raises(ValueError):
        evaluate_boundary(**arguments)


def test_boundary_rejects_negative_min_ess():
    with pytest.raises(ValueError):
        evaluate_boundary(
            predictions=[0.9],
            weights=[1.0],
            min_ess=-1.0,
        )


def test_boundary_rejects_negative_min_weight_sum():
    with pytest.raises(ValueError):
        evaluate_boundary(
            predictions=[0.9],
            weights=[1.0],
            min_weight_sum=-1.0,
        )


def test_boundary_rejects_overflowing_weight_sum():
    with pytest.raises(ValueError):
        evaluate_boundary(
            predictions=[0.9, 0.8],
            weights=[1e308, 1e308],
            min_ess=0.0,
            min_weight_sum=0.0,
        )


# ---------------------------------------------------------
# BOUNDARY CLASSIFICATION
# ---------------------------------------------------------

def test_boundary_classifies_positive_support():
    result = evaluate_boundary(
        predictions=[0.9, 0.95, 0.92],
        weights=[1.0, 1.0, 1.0],
        min_ess=0.0,
        min_weight_sum=0.0,
    )

    assert result.state is BoundaryState.POSITIVE
    assert result.lower_quantile is not None
    assert result.upper_quantile is not None


def test_boundary_classifies_negative_support():
    result = evaluate_boundary(
        predictions=[0.05, 0.1, 0.15],
        weights=[1.0, 1.0, 1.0],
        min_ess=0.0,
        min_weight_sum=0.0,
    )

    assert result.state is BoundaryState.NEGATIVE


def test_boundary_returns_unknown_without_support():
    result = evaluate_boundary(
        predictions=[0.9, 0.8],
        weights=[0.0, 0.0],
        min_ess=0.0,
        min_weight_sum=0.0,
    )

    assert result.state is BoundaryState.UNKNOWN
    assert result.lower_quantile is None
    assert result.upper_quantile is None
    assert result.support_count == 0


def test_boundary_ignores_zero_weight_predictions():
    result = evaluate_boundary(
        predictions=[0.0, 0.9, 0.95],
        weights=[0.0, 1.0, 1.0],
        min_ess=0.0,
        min_weight_sum=0.0,
        lower_quantile=0.0,
        upper_quantile=1.0,
    )

    assert result.state is BoundaryState.POSITIVE
    assert result.lower_quantile == pytest.approx(0.9)
    assert result.upper_quantile == pytest.approx(0.95)
    assert result.support_count == 2
