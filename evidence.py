
from __future__ import annotations

import math

from models import EvidenceInput, EvidenceKind


class EvidenceValidationError(ValueError):
    """Raised when evidence violates UROBOROS validation rules."""


def validate_evidence(
    item: EvidenceInput,
    min_quality: float = 0.5,
) -> None:
    """
    Validate evidence before it enters the UROBOROS memory system.

    POSITIVE:
        Requires a directly observed successful outcome.

    NEGATIVE:
        Requires a directly observed unsuccessful outcome
        under valid testing conditions.

    INCONCLUSIVE:
        Preserves an uncertain result without treating it
        as positive or negative evidence.

    Caller-provided valid=True never overrides validation.
    """

    # 1. Validate quality configuration.
    if not math.isfinite(min_quality):
        raise EvidenceValidationError(
            "min_quality must be finite"
        )

    if not 0.0 <= min_quality <= 1.0:
        raise EvidenceValidationError(
            "min_quality must be within [0, 1]"
        )

    if not math.isfinite(item.quality):
        raise EvidenceValidationError(
            "quality must be finite"
        )

    if not 0.0 <= item.quality <= 1.0:
        raise EvidenceValidationError(
            "quality must be within [0, 1]"
        )

    if item.quality < min_quality:
        raise EvidenceValidationError(
            "quality below configured minimum"
        )

    # 2. Validate provenance.
    if not isinstance(item.source, str) or not item.source.strip():
        raise EvidenceValidationError(
            "source provenance is required"
        )

    if not isinstance(item.association_id, str) or not item.association_id.strip():
        raise EvidenceValidationError(
            "association_id is required"
        )

    # 3. Validate evidence classification.
    if not isinstance(item.kind, EvidenceKind):
        raise EvidenceValidationError(
            "kind must be an EvidenceKind value"
        )

    # 4. Inconclusive evidence is preserved separately.
    # It does not update positive/negative boundary estimates.
    if item.kind is EvidenceKind.INCONCLUSIVE:

        if item.outcome is not None:
            raise EvidenceValidationError(
                "INCONCLUSIVE evidence requires outcome=None"
            )

        if item.valid is True:
            raise EvidenceValidationError(
                "INCONCLUSIVE evidence cannot be marked valid"
            )

        return

    # 5. Positive and negative evidence require valid observation.
    if not item.observed:
        raise EvidenceValidationError(
            "outcome was not directly observed"
        )

    if not item.retrieval_succeeded:
        raise EvidenceValidationError(
            "retrieval/observation process failed"
        )

    if not item.context_complete:
        raise EvidenceValidationError(
            "context is incomplete"
        )

    # 6. Validate positive evidence.
    if item.kind is EvidenceKind.POSITIVE:

        if item.outcome is not True:
            raise EvidenceValidationError(
                "POSITIVE evidence requires outcome=True"
            )

        return

    # 7. Validate negative evidence.
    if item.kind is EvidenceKind.NEGATIVE:

        if item.outcome is not False:
            raise EvidenceValidationError(
                "NEGATIVE evidence requires outcome=False"
            )

        return

    raise EvidenceValidationError(
        "unsupported evidence classification"
    )
