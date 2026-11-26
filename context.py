
from __future__ import annotations

import math

from models import Context


def context_distance(a: Context, b: Context) -> float | None:
    """
    RMS distance over comparable observed features.

    Numeric features are scaled by max(1, |a|, |b|).
    Categorical values use 0/1 mismatch.

    Missing values are excluded, never imputed as zero.
    Returns None if there is no comparable overlap.
    """

    a_obs = a.observed_features()
    b_obs = b.observed_features()

    common = sorted(a_obs.intersection(b_obs))

    distances: list[float] = []

    for key in common:
        av = a.features.get(key)
        bv = b.features.get(key)

        if av is None or bv is None:
            continue

        if (
            isinstance(av, (int, float))
            and not isinstance(av, bool)
            and isinstance(bv, (int, float))
            and not isinstance(bv, bool)
        ):
            av_float = float(av)
            bv_float = float(bv)

            if not (
                math.isfinite(av_float)
                and math.isfinite(bv_float)
            ):
                continue

            scale = max(
                1.0,
                abs(av_float),
                abs(bv_float),
            )

            distances.append(
                abs(av_float - bv_float) / scale
            )

        else:
            distances.append(
                0.0 if av == bv else 1.0
            )

    if not distances:
        return None

    return math.sqrt(
        sum(distance ** 2 for distance in distances)
        / len(distances)
    )


def gaussian_weight(
    distance: float,
    bandwidth: float,
) -> float:
    """
    Convert contextual distance into a Gaussian similarity weight.

    Returns a value in [0, 1].
    """

    if not math.isfinite(distance) or distance < 0:
        raise ValueError("distance must be finite and >= 0")

    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and > 0")

    return math.exp(
        -(distance ** 2) / (2.0 * bandwidth ** 2)
    )
