
from __future__ import annotations

from collections.abc import Sequence

from models import (
    Context,
    TraceRecord,
    EvidenceInput,
    ProbeCandidate,
    BoundaryResult,
    new_id,
)
from context import context_distance, gaussian_weight
from evidence import validate_evidence
from boundary import evaluate_boundary
from database import UroborosDatabase
from probing import choose_context
from conflicts import HonestConflictResolution


class UroborosEngine:
    """
    Main orchestration layer for UROBOROS.

    Coordinates memory, evidence validation, contextual
    similarity, boundary evaluation, probing and conflicts.
    """

    def __init__(
        self,
        database_path: str = "uroboros.db",
        bandwidth: float = 0.5,
        min_quality: float = 0.5,
    ):
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")

        if not 0 <= min_quality <= 1:
            raise ValueError("min_quality must be within [0, 1]")

        self.database = UroborosDatabase(database_path)
        self.bandwidth = bandwidth
        self.min_quality = min_quality
        self.conflicts = HonestConflictResolution()

    def remember(
        self,
        content: str,
        context: Context,
        source: str,
    ) -> str:
        """Store a new memory trace."""

        if not content.strip():
            raise ValueError("content cannot be empty")

        if not source.strip():
            raise ValueError("source cannot be empty")

        trace = TraceRecord(
            trace_id=new_id(),
            content=content,
            context=context,
            source=source,
        )

        return self.database.save_trace(trace)

    def recall(self, trace_id: str) -> TraceRecord | None:
        """Retrieve a memory trace by identifier."""

        return self.database.get_trace(trace_id)

    def recall_similar(
        self,
        context: Context,
        limit: int = 10,
    ) -> list[tuple[TraceRecord, float]]:
        """Retrieve traces ranked by contextual similarity."""

        if limit < 1:
            raise ValueError("limit must be positive")

        ranked = []

        for trace in self.database.list_traces(limit=10000):
            distance = context_distance(context, trace.context)

            if distance is None:
                continue

            weight = gaussian_weight(
                distance,
                self.bandwidth,
            )

            ranked.append((trace, weight))

        ranked.sort(key=lambda item: item[1], reverse=True)

        return ranked[:limit]

    def observe(
        self,
        evidence: EvidenceInput,
    ) -> str:
        """Validate and persist an evidence record."""

        return self.database.save_evidence(
            evidence,
            min_quality=self.min_quality,
        )

    def evaluate_association(
        self,
        association_id: str,
        current_context: Context,
        predictions: Sequence[float],
        min_ess: float = 5.0,
        min_weight_sum: float = 1.0,
    ) -> BoundaryResult:
        """
        Evaluate an association in the current context.

        predictions must contain one externally generated
        probability for each stored evidence record, in the
        same order returned by list_evidence().

        Inconclusive evidence is excluded from boundary support.
        """

        records = self.database.list_evidence(association_id)

        if len(predictions) != len(records):
            raise ValueError(
                "Provide exactly one prediction per stored evidence record"
            )

        usable_predictions = []
        weights = []

        for record, prediction in zip(records, predictions):
            if record.kind.value == "INCONCLUSIVE":
                continue

            distance = context_distance(
                current_context,
                record.context,
            )

            if distance is None:
                continue

            weight = gaussian_weight(
                distance,
                self.bandwidth,
            )

            usable_predictions.append(float(prediction))
            weights.append(weight)

        if not usable_predictions:
            return evaluate_boundary(
                [0.5],
                [0.0],
                min_ess=min_ess,
                min_weight_sum=min_weight_sum,
            )

        return evaluate_boundary(
            usable_predictions,
            weights,
            min_ess=min_ess,
            min_weight_sum=min_weight_sum,
        )

    def probe(
        self,
        candidates: Sequence[ProbeCandidate],
        temperature: float = 1.0,
    ):
        """Choose the next context to investigate."""

        return choose_context(
            candidates,
            temperature=temperature,
        )

    def add_conflict(
        self,
        conflict_id: str,
        interpretation: str,
        source: str,
        context_description: str = "",
        confidence: float = 0.5,
    ):
        """Preserve an alternative interpretation."""

        return self.conflicts.add_branch(
            conflict_id=conflict_id,
            interpretation=interpretation,
            source=source,
            context_description=context_description,
            confidence=confidence,
        )

    def close(self) -> None:
        """Close the database connection."""

        self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
