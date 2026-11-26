
from __future__ import annotations

import math
from collections.abc import Sequence

from models import BoundaryResult, BoundaryState


def effective_sample_size(weights: Sequence[float]) -> float:
    """Calculate effective sample size (ESS)."""
    values = [float(w) for w in weights]

    if any(not math.isfinite(w) or w < 0 for w in values):
        raise ValueError("weights must be finite and non-negative")

    total = sum(values)
    squared_total = sum(w * w for w in values)

    if squared_total == 0:
        return 0.0

    return total * total / squared_total


def weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> float:
    """Calculate an empirical weighted quantile."""
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be within [0, 1]")

    if len(values) != len(weights) or not values:
        raise ValueError("values and weights must have equal non-zero length")

    pairs = sorted(
        (float(v), float(w))
        for v, w in zip(values, weights)
    )

    if any(not math.isfinite(v) for v, _ in pairs):
        raise ValueError("values must be finite")

    if any(not math.isfinite(w) or w < 0 for _, w in pairs):
        raise ValueError("weights must be finite and non-negative")

    total_weight = sum(w for _, w in pairs)

    if total_weight <= 0:
        raise ValueError("total weight must be positive")

    threshold = quantile * total_weight
    cumulative = 0.0

    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return value

    return pairs[-1][0]


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

    if len(predictions) != len(weights) or not predictions:
        raise ValueError("predictions and weights must have equal non-zero length")

    if min_ess < 0 or min_weight_sum < 0:
        raise ValueError("support thresholds must be non-negative")

    if not 0 <= negative_threshold < positive_threshold <= 1:
        raise ValueError("invalid classification thresholds")

    if not 0 <= lower_quantile <= upper_quantile <= 1:
        raise ValueError("invalid quantile range")

    values = [float(p) for p in predictions]
    local_weights = [float(w) for w in weights]

    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("predictions must be finite probabilities in [0, 1]")

    if any(not math.isfinite(w) or w < 0 for w in local_weights):
        raise ValueError("weights must be finite and non-negative")

    weight_sum = sum(local_weights)
    ess = effective_sample_size(local_weights)
    support_count = sum(w > 0 for w in local_weights)

    if weight_sum <= 0:
        return BoundaryResult(
            BoundaryState.UNKNOWN, None, None,
            ess, weight_sum, support_count,
            "No local contextual support."
        )

    if ess < min_ess or weight_sum < min_weight_sum:
        return BoundaryResult(
            BoundaryState.UNKNOWN, None, None,
            ess, weight_sum, support_count,
            "Insufficient local support."
        )

    low = weighted_quantile(values, local_weights, lower_quantile)
    high = weighted_quantile(values, local_weights, upper_quantile)

    if low >= positive_threshold:
        state = BoundaryState.POSITIVE
        reason = "Lower empirical quantile supports positive applicability."
    elif high <= negative_threshold:
        state = BoundaryState.NEGATIVE
        reason = "Upper empirical quantile supports negative applicability."
    else:
        state = BoundaryState.UNKNOWN
        reason = "Prediction interval overlaps the uncertain region."

    return BoundaryResult(
        state, low, high, ess, weight_sum,
        support_count, reason
    )
