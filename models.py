
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping
import time
import uuid


class EvidenceKind(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    INCONCLUSIVE = "INCONCLUSIVE"


class BoundaryState(str, Enum):
    POSITIVE = "B+"
    NEGATIVE = "B-"
    UNKNOWN = "B?"


@dataclass(frozen=True)
class Context:
    features: Mapping[str, Any]
    observed: frozenset[str] | None = None

    def observed_features(self) -> frozenset[str]:
        if self.observed is not None:
            return self.observed.intersection(self.features.keys())

        return frozenset(
            key
            for key, value in self.features.items()
            if value is not None
        )


@dataclass(frozen=True)
class TraceRecord:
    trace_id: str
    content: str
    context: Context
    source: str
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class EvidenceInput:
    association_id: str
    context: Context
    kind: EvidenceKind
    outcome: bool | None
    quality: float
    source: str
    source_family: str | None = None
    observed: bool = True
    retrieval_succeeded: bool = True
    context_complete: bool = True
    # Advisory only. It never overrides the validation gate.
    valid: bool | None = None


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    association_id: str
    context: Context
    kind: EvidenceKind
    outcome: bool | None
    quality: float
    source: str
    source_family: str | None
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class BoundaryResult:
    state: BoundaryState
    lower_quantile: float | None
    upper_quantile: float | None
    local_ess: float
    local_weight_sum: float
    support_count: int
    reason: str


@dataclass(frozen=True)
class PotentialAssociation:
    association_id: str
    source_trace_id: str
    target_trace_id: str
    score: float
    status: str = "CANDIDATE"
    note: str = ""


@dataclass(frozen=True)
class ProbeCandidate:
    context_id: str
    information_gain: float
    task_value: float
    cost: float
    risk: float


@dataclass(frozen=True)
class ProbeDecision:
    context_id: str
    score: float
    propensity: float
    policy_version: str
    temperature: float


def new_id() -> str:
    return str(uuid.uuid4())
