
import pytest

from boundary import (
    effective_sample_size,
    weighted_quantile,
    evaluate_boundary,
)
from models import BoundaryState


def test_ess_equal_weights():
    assert effective_sample_size([1, 1, 1, 1]) == 4.0


def test_ess_single_weight():
    assert effective_sample_size([5]) == 1.0


def test_ess_zero_weights():
    assert effective_sample_size([0, 0, 0]) == 0.0


def test_ess_unequal_weights():
    result = effective_sample_size([1, 1, 0, 0])
    assert result == 2.0


def test_weighted_quantile_median():
    result = weighted_quantile(
        [0.1, 0.5, 0.9],
        [1, 1, 1],
        0.5,
    )
    assert result == 0.5


def test_weighted_quantile_respects_weights():
    result = weighted_quantile(
        [0.1, 0.9],
        [1, 9],
        0.5,
    )
    assert result == 0.9


def test_positive_boundary():
    result = evaluate_boundary(
        predictions=[0.9, 0.95, 0.92, 0.91, 0.94],
        weights=[1, 1, 1, 1, 1],
        min_ess=3,
    )

    assert result.state == BoundaryState.POSITIVE


def test_negative_boundary():
    result = evaluate_boundary(
        predictions=[0.05, 0.1, 0.08, 0.12, 0.09],
        weights=[1, 1, 1, 1, 1],
        min_ess=3,
    )

    assert result.state == BoundaryState.NEGATIVE


def test_insufficient_support_returns_unknown():
    result = evaluate_boundary(
        predictions=[0.95],
        weights=[0.1],
        min_ess=5,
        min_weight_sum=1,
    )

    assert result.state == BoundaryState.UNKNOWN


def test_overlapping_predictions_return_unknown():
    result = evaluate_boundary(
        predictions=[0.1, 0.5, 0.9, 0.6, 0.4],
        weights=[1, 1, 1, 1, 1],
        min_ess=3,
    )

    assert result.state == BoundaryState.UNKNOWN


@pytest.mark.parametrize(
    "weights",
    [
        [-1, 1],
        [float("nan"), 1],
        [float("inf"), 1],
    ],
)
def test_ess_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        effective_sample_size(weights)


def test_quantile_rejects_invalid_probability():
    with pytest.raises(ValueError):
        weighted_quantile([0.1, 0.9], [1, 1], 1.5)
