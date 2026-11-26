
"""
UROBOROS 2.0
Contextual memory and evidence evaluation framework.

Experimental implementation.
"""

from models import (
    Context,
    TraceRecord,
    EvidenceInput,
    EvidenceRecord,
    EvidenceKind,
    BoundaryState,
    BoundaryResult,
    PotentialAssociation,
    ProbeCandidate,
    ProbeDecision,
    new_id,
)

from context import (
    context_distance,
    gaussian_weight,
)

from evidence import (
    validate_evidence,
    EvidenceValidationError,
)

from boundary import (
    effective_sample_size,
    weighted_quantile,
    evaluate_boundary,
)

from database import UroborosDatabase

from probing import (
    score_candidate,
    softmax,
    choose_context,
)

from conflicts import (
    ConflictBranch,
    HonestConflictResolution,
)

from engine import UroborosEngine


__version__ = "2.0.0"

__all__ = [
    "Context",
    "TraceRecord",
    "EvidenceInput",
    "EvidenceRecord",
    "EvidenceKind",
    "BoundaryState",
    "BoundaryResult",
    "PotentialAssociation",
    "ProbeCandidate",
    "ProbeDecision",
    "new_id",
    "context_distance",
    "gaussian_weight",
    "validate_evidence",
    "EvidenceValidationError",
    "effective_sample_size",
    "weighted_quantile",
    "evaluate_boundary",
    "UroborosDatabase",
    "score_candidate",
    "softmax",
    "choose_context",
    "ConflictBranch",
    "HonestConflictResolution",
    "UroborosEngine",
]
