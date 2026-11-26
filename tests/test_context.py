
import math

import pytest

from models import Context
from context import context_distance, gaussian_weight


def test_identical_contexts_have_zero_distance():
    a = Context({"temperature": 20, "location": "forest"})
    b = Context({"temperature": 20, "location": "forest"})

    assert context_distance(a, b) == 0.0


def test_different_numeric_features():
    a = Context({"temperature": 20})
    b = Context({"temperature": 30})

    distance = context_distance(a, b)

    assert distance is not None
    assert distance > 0


def test_different_categorical_features():
    a = Context({"location": "forest"})
    b = Context({"location": "desert"})

    assert context_distance(a, b) == 1.0


def test_missing_features_are_excluded():
    a = Context({"temperature": 20, "location": None})
    b = Context({"temperature": 20, "location": "forest"})

    assert context_distance(a, b) == 0.0


def test_no_common_observed_features_returns_none():
    a = Context({"temperature": 20})
    b = Context({"location": "forest"})

    assert context_distance(a, b) is None


def test_gaussian_weight_at_zero_distance():
    assert gaussian_weight(0.0, 0.5) == 1.0


def test_gaussian_weight_decreases_with_distance():
    near = gaussian_weight(0.2, 0.5)
    far = gaussian_weight(1.0, 0.5)

    assert near > far
    assert 0 <= far <= 1


@pytest.mark.parametrize(
    "distance, bandwidth",
    [
        (-1, 0.5),
        (float("nan"), 0.5),
        (float("inf"), 0.5),
        (1.0, 0),
        (1.0, -0.5),
        (1.0, float("inf")),
    ],
)
def test_gaussian_weight_rejects_invalid_arguments(distance, bandwidth):
    with pytest.raises(ValueError):
        gaussian_weight(distance, bandwidth)
