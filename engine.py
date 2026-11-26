
from __future__ import annotations

import math
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
from boundary import evaluate_boundary
from database import UroborosDatabase
from probing import choose_context
from conflicts import HonestConflictResolution


def _finite_number(value: float, name: str) -> float:
    """Convert a value to float and reject invalid numeric input."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a finite number"
        ) from exc

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")

    return result


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
        bandwidth = _finite_number(bandwidth, "bandwidth")
        min_quality = _finite_number(
            min_quality,
            "min_quality",
        )

        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")

        if not 0 <= min_quality <= 1:
            raise ValueError(
                "min_quality must be within [0, 1]"
            )

        if not str(database_path).strip():
            raise ValueError("database_path cannot be empty")

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

        if not isinstance(content, str) or not content.strip():
            raise ValueError("content cannot be empty")

        if not isinstance(source, str) or not source.strip():
            raise ValueError("source cannot be empty")

        if not isinstance(context, Context):
            raise ValueError("context must be a Context instance")

        trace = TraceRecord(
            trace_id=new_id(),
            content=content,
            context=context,
            source=source,
        )

        return self.database.save_trace(trace)

    def recall(self, trace_id: str) -> TraceRecord | None:
        """Retrieve a memory trace by identifier."""

        if not isinstance(trace_id, str) or not trace_id.strip():
            raise ValueError("trace_id cannot be empty")

        return self.database.get_trace(trace_id)

    def recall_similar(
        self,
        context: Context,
        limit: int = 10,
    ) -> list[tuple[TraceRecord, float]]:
        """
        Retrieve traces ranked by contextual similarity.

        Searches the most recent 10,000 traces by default.
        This is a bounded search, not a full-database search.
        """

        if not isinstance(context, Context):
            raise ValueError("context must be a Context instance")

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be a positive integer")

        if limit < 1:
            raise ValueError("limit must be positive")

        ranked = []

        for trace in self.database.list_traces(limit=10000):
            distance = context_distance(
                context,
                trace.context,
            )

            if distance is None:
                continue

            weight = gaussian_weight(
                distance,
                self.bandwidth,
            )

            ranked.append((trace, weight))

        ranked.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return ranked[:limit]

    def observe(
        self,
        evidence: EvidenceInput,
    ) -> str:
        """Validate and persist an evidence record."""

        if not isinstance(evidence, EvidenceInput):
            raise ValueError(
                "evidence must be an EvidenceInput instance"
            )

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

        Predictions must contain one externally generated
        probability for each stored evidence record, in the
        same order returned by list_evidence().

        Inconclusive evidence is excluded from boundary support.
        This method does not train a prediction model.
        """

        if (
            not isinstance(association_id, str)
            or not association_id.strip()
        ):
            raise ValueError("association_id cannot be empty")

        if not isinstance(current_context, Context):
            raise ValueError(
                "current_context must be a Context instance"
            )

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

            usable_predictions.append(
                _finite_number(prediction, "prediction")
            )
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
