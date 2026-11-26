
import random

import pytest

from models import ProbeCandidate
from probing import score_candidate, softmax, choose_context


def make_candidate(
    context_id="context-001",
    information_gain=0.8,
    task_value=0.5,
    cost=0.1,
    risk=0.1,
):
    return ProbeCandidate(
        context_id=context_id,
        information_gain=information_gain,
        task_value=task_value,
        cost=cost,
        risk=risk,
    )


def test_candidate_score():
    candidate = make_candidate()

    assert score_candidate(candidate) == pytest.approx(1.1)


def test_softmax_probabilities_sum_to_one():
    probabilities = softmax([1.0, 2.0, 3.0])

    assert sum(probabilities) == pytest.approx(1.0)


def test_softmax_preserves_score_order():
    probabilities = softmax([1.0, 2.0, 3.0])

    assert probabilities[0] < probabilities[1]
    assert probabilities[1] < probabilities[2]


def test_softmax_rejects_empty_scores():
    with pytest.raises(ValueError):
        softmax([])


def test_softmax_rejects_invalid_temperature():
    with pytest.raises(ValueError):
        softmax([1.0, 2.0], temperature=0)


def test_choose_context_returns_candidate():
    candidates = [
        make_candidate("context-A"),
        make_candidate("context-B"),
        make_candidate("context-C"),
    ]

    result = choose_context(
        candidates,
        rng=random.Random(42),
    )

    assert result.context_id in {
        "context-A",
        "context-B",
        "context-C",
    }


def test_probe_decision_has_valid_propensity():
    candidates = [
        make_candidate("context-A"),
        make_candidate("context-B"),
    ]

    result = choose_context(
        candidates,
        rng=random.Random(42),
    )

    assert 0 < result.propensity <= 1
    assert result.temperature == 1.0
    assert result.policy_version == "uroboros-probing-1.0"


def test_same_seed_produces_same_decision():
    candidates = [
        make_candidate("context-A"),
        make_candidate("context-B"),
        make_candidate("context-C"),
    ]

    first = choose_context(
        candidates,
        rng=random.Random(123),
    )

    second = choose_context(
        candidates,
        rng=random.Random(123),
    )

    assert first == second


def test_duplicate_context_ids_are_rejected():
    candidates = [
        make_candidate("same"),
        make_candidate("same"),
    ]

    with pytest.raises(ValueError):
        choose_context(candidates)


def test_empty_candidates_are_rejected():
    with pytest.raises(ValueError):
        choose_context([])


def test_negative_cost_is_rejected():
    candidate = make_candidate(cost=-1)

    with pytest.raises(ValueError):
        score_candidate(candidate)
