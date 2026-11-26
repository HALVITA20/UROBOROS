
import pytest

from models import Context, EvidenceInput, EvidenceKind
from evidence import validate_evidence, EvidenceValidationError


def make_evidence(
    kind=EvidenceKind.POSITIVE,
    outcome=True,
    **overrides,
):
    values = {
        "association_id": "association-001",
        "context": Context({"location": "forest"}),
        "kind": kind,
        "outcome": outcome,
        "quality": 0.9,
        "source": "test-observation",
        "source_family": "test",
        "observed": True,
        "retrieval_succeeded": True,
        "context_complete": True,
        "valid": None,
    }

    values.update(overrides)
    return EvidenceInput(**values)


def test_valid_positive_evidence():
    item = make_evidence()

    validate_evidence(item)


def test_valid_negative_evidence():
    item = make_evidence(
        kind=EvidenceKind.NEGATIVE,
        outcome=False,
    )

    validate_evidence(item)


def test_inconclusive_evidence_is_preserved():
    item = make_evidence(
        kind=EvidenceKind.INCONCLUSIVE,
        outcome=None,
    )

    validate_evidence(item)


def test_positive_requires_true_outcome():
    item = make_evidence(outcome=False)

    with pytest.raises(EvidenceValidationError):
        validate_evidence(item)


def test_negative_requires_false_outcome():
    item = make_evidence(
        kind=EvidenceKind.NEGATIVE,
        outcome=True,
    )

    with pytest.raises(EvidenceValidationError):
        validate_evidence(item)


def test_failed_retrieval_cannot_be_negative_evidence():
    item = make_evidence(
        kind=EvidenceKind.NEGATIVE,
        outcome=False,
        retrieval_succeeded=False,
    )

    with pytest.raises(EvidenceValidationError):
        validate_evidence(item)


def test_incomplete_context_cannot_be_negative_evidence():
    item = make_evidence(
        kind=EvidenceKind.NEGATIVE,
        outcome=False,
        context_complete=False,
    )

    with pytest.raises(EvidenceValidationError):
        validate_evidence(item)


def test_low_quality_evidence_is_rejected():
    item = make_evidence(quality=0.2)

    with pytest.raises(EvidenceValidationError):
        validate_evidence(item)


def test_empty_source_is_rejected():
    item = make_evidence(source="")

    with pytest.raises(EvidenceValidationError):
        validate_evidence(item)


def test_inconclusive_cannot_claim_validity():
    item = make_evidence(
        kind=EvidenceKind.INCONCLUSIVE,
        outcome=None,
        valid=True,
    )

    with pytest.raises(EvidenceValidationError):
        validate_evidence(item)


def test_invalid_quality_threshold_is_rejected():
    item = make_evidence()

    with pytest.raises(EvidenceValidationError):
        validate_evidence(item, min_quality=1.5)
