
from __future__ import annotations

import math
from collections.abc import Sequence

from models import BoundaryResult, BoundaryState


def _finite_float(value: float, name: str) -> float:
    """Convert a value to float and reject non-finite numbers."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")

    return result


def _validated_weights(weights: Sequence[float]) -> list[float]:
    """Convert and validate a sequence of non-negative weights."""
    result = [_finite_float(w, "weights") for w in weights]

    if any(w < 0 for w in result):
        raise ValueError("weights must be non-negative")

    return result


def effective_sample_size(weights: Sequence[float]) -> float:
    """
    Calculate effective sample size (ESS).

    Scaling weights before calculation avoids numerical overflow.
    Zero-weight observations do not contribute to ESS.
    """
    values = _validated_weights(weights)

    if not values:
        return 0.0

    max_weight = max(values)

    if max_weight == 0:
        return 0.0

    scaled = [w / max_weight for w in values]

    total = math.fsum(scaled)
    squared_total = math.fsum(w * w for w in scaled)

    if squared_total == 0:
        return 0.0

    return (total * total) / squared_total


def weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> float:
    """
    Calculate an empirical weighted quantile.

    Uses the inverse cumulative distribution definition.
    Observations with zero weight are completely ignored.
    """
    q = _finite_float(quantile, "quantile")

    if not 0 <= q <= 1:
        raise ValueError("quantile must be within [0, 1]")

    if len(values) != len(weights) or len(values) == 0:
        raise ValueError(
            "values and weights must have equal non-zero length"
        )

    numeric_values = [
        _finite_float(v, "values")
        for v in values
    ]

    numeric_weights = _validated_weights(weights)

    pairs = sorted(
        (value, weight)
        for value, weight in zip(numeric_values, numeric_weights)
        if weight > 0
    )

    if not pairs:
        raise ValueError("total weight must be positive")

    max_weight = max(weight for _, weight in pairs)

    scaled_pairs = [
        (value, weight / max_weight)
        for value, weight in pairs
    ]

    total_weight = math.fsum(
        weight for _, weight in scaled_pairs
    )

    threshold = q * total_weight
    cumulative = 0.0

    for value, weight in scaled_pairs:
        cumulative += weight

        if cumulative >= threshold:
            return value

    # Protect against floating-point rounding at the upper boundary.
    return scaled_pairs[-1][0]


def evaluate_boundary(
    predictions: Sequence[float],
    weights: Sequence[float],
    min_ess: float = 5.0,
    min_weight_sum: float = 1.0,
    positive_threshold: float = 0.8,
    negative_threshold: float = 0.2,
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
) -> BoundaryResult:
    """
    Evaluate local contextual support and classify an association.

    Predictions are supplied by an external model or ensemble.
    This function does not train that model.
    """

    if len(predictions) != len(weights) or len(predictions) == 0:
        raise ValueError(
            "predictions and weights must have equal non-zero length"
        )

    min_ess = _finite_float(min_ess, "min_ess")
    min_weight_sum = _finite_float(
        min_weight_sum,
        "min_weight_sum",
    )
    positive_threshold = _finite_float(
        positive_threshold,
        "positive_threshold",
    )
    negative_threshold = _finite_float(
        negative_threshold,
        "negative_threshold",
    )
    lower_quantile = _finite_float(
        lower_quantile,
        "lower_quantile",
    )
    upper_quantile = _finite_float(
        upper_quantile,
        "upper_quantile",
    )

    if min_ess < 0 or min_weight_sum < 0:
        raise ValueError(
            "support thresholds must be non-negative"
        )

    if not 0 <= negative_threshold < positive_threshold <= 1:
        raise ValueError("invalid classification thresholds")

    if not 0 <= lower_quantile <= upper_quantile <= 1:
        raise ValueError("invalid quantile range")

    values = [
        _finite_float(p, "predictions")
        for p in predictions
    ]

    if any(not 0 <= p <= 1 for p in values):
        raise ValueError(
            "predictions must be probabilities in [0, 1]"
        )

    local_weights = _validated_weights(weights)

    try:
        weight_sum = math.fsum(local_weights)
    except OverflowError as exc:
        raise ValueError("total weight must be finite") from exc

    if not math.isfinite(weight_sum):
        raise ValueError("total weight must be finite")

    ess = effective_sample_size(local_weights)
    support_count = sum(w > 0 for w in local_weights)

    if weight_sum <= 0:
        return BoundaryResult(
            BoundaryState.UNKNOWN,
            None,
            None,
            ess,
            weight_sum,
            support_count,
            "No local contextual support.",
        )

    if ess < min_ess or weight_sum < min_weight_sum:
        return BoundaryResult(
            BoundaryState.UNKNOWN,
            None,
            None,
            ess,
            weight_sum,
            support_count,
            "Insufficient local support.",
        )

    low = weighted_quantile(
        values,
        local_weights,
        lower_quantile,
    )

    high = weighted_quantile(
        values,
        local_weights,
        upper_quantile,
    )

    if low >= positive_threshold:
        state = BoundaryState.POSITIVE
        reason = (
            "Lower empirical quantile supports positive applicability."
        )
    elif high <= negative_threshold:
        state = BoundaryState.NEGATIVE
        reason = (
            "Upper empirical quantile supports negative applicability."
        )
    else:
        state = BoundaryState.UNKNOWN
        reason = (
            "Prediction interval overlaps the uncertain region."
        )

    return BoundaryResult(
        state,
        low,
        high,
        ess,
        weight_sum,
        support_count,
        reason,
    )
