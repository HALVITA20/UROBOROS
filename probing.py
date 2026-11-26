
from __future__ import annotations

import math
import random
from collections.abc import Sequence

from models import ProbeCandidate, ProbeDecision


def score_candidate(candidate: ProbeCandidate) -> float:
    """Calculate the utility score of a context candidate."""

    values = (
        candidate.information_gain,
        candidate.task_value,
        candidate.cost,
        candidate.risk,
    )

    if any(not math.isfinite(float(v)) for v in values):
        raise ValueError("Candidate values must be finite")

    if candidate.cost < 0 or candidate.risk < 0:
        raise ValueError("Cost and risk must be non-negative")

    return (
        candidate.information_gain
        + candidate.task_value
        - candidate.cost
        - candidate.risk
    )


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Convert utility scores into selection probabilities."""

    if not scores:
        raise ValueError("At least one score is required")

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")

    values = [float(score) for score in scores]

    if any(not math.isfinite(score) for score in values):
        raise ValueError("Scores must be finite")

    scaled = [score / temperature for score in values]
    maximum = max(scaled)

    exponentials = [
        math.exp(value - maximum)
        for value in scaled
    ]

    total = sum(exponentials)

    return [value / total for value in exponentials]


def choose_context(
    candidates: Sequence[ProbeCandidate],
    temperature: float = 1.0,
    policy_version: str = "uroboros-probing-1.0",
    rng: random.Random | None = None,
) -> ProbeDecision:
    """
    Select a context using softmax exploration.

    The propensity field records the probability assigned
    to the selected candidate by this policy.
    """

    if not candidates:
        raise ValueError("No probe candidates provided")

    context_ids = [candidate.context_id for candidate in candidates]

    if any(not isinstance(value, str) or not value.strip()
           for value in context_ids):
        raise ValueError("Every candidate requires a context_id")

    if len(set(context_ids)) != len(context_ids):
        raise ValueError("Candidate context_id values must be unique")

    if not policy_version.strip():
        raise ValueError("policy_version cannot be empty")

    scores = [
        score_candidate(candidate)
        for candidate in candidates
    ]

    probabilities = softmax(scores, temperature)

    generator = rng if rng is not None else random.Random()

    selected_index = generator.choices(
        range(len(candidates)),
        weights=probabilities,
        k=1,
    )[0]

    selected = candidates[selected_index]

    return ProbeDecision(
        context_id=selected.context_id,
        score=scores[selected_index],
        propensity=probabilities[selected_index],
        policy_version=policy_version,
        temperature=temperature,
    )
