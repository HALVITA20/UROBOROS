
from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from uuid import uuid4


@dataclass(frozen=True)
class ConflictBranch:
    branch_id: str
    conflict_id: str
    interpretation: str
    source: str
    context_description: str
    confidence: float
    created_at: float = field(default_factory=time)


class HonestConflictResolution:
    """
    Preserve competing interpretations without silently
    collapsing disagreement into a single answer.

    This module preserves branches. It does not automatically
    determine which interpretation is true.
    """

    def __init__(self):
        self._branches: dict[str, list[ConflictBranch]] = {}

    def add_branch(
        self,
        conflict_id: str,
        interpretation: str,
        source: str,
        context_description: str = "",
        confidence: float = 0.5,
    ) -> ConflictBranch:

        if not conflict_id.strip():
            raise ValueError("conflict_id is required")

        if not interpretation.strip():
            raise ValueError("interpretation is required")

        if not source.strip():
            raise ValueError("source is required")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")

        branch = ConflictBranch(
            branch_id=str(uuid4()),
            conflict_id=conflict_id,
            interpretation=interpretation,
            source=source,
            context_description=context_description,
            confidence=confidence,
        )

        self._branches.setdefault(conflict_id, []).append(branch)

        return branch

    def get_branches(
        self,
        conflict_id: str,
    ) -> tuple[ConflictBranch, ...]:
        """Return all preserved branches for a conflict."""

        return tuple(self._branches.get(conflict_id, []))

    def list_conflicts(self) -> tuple[str, ...]:
        """Return identifiers of all recorded conflicts."""

        return tuple(sorted(self._branches.keys()))

    def remove_branch(
        self,
        conflict_id: str,
        branch_id: str,
    ) -> bool:
        """
        Explicitly remove a branch.

        Returns False if the branch does not exist.
        """

        branches = self._branches.get(conflict_id)

        if branches is None:
            return False

        updated = [
            branch
            for branch in branches
            if branch.branch_id != branch_id
        ]

        if len(updated) == len(branches):
            return False

        if updated:
            self._branches[conflict_id] = updated
        else:
            del self._branches[conflict_id]

        return True

    def compare_branches(
        self,
        conflict_id: str,
    ) -> dict[str, object]:
        """
        Return a neutral overview of competing interpretations.
        No branch is automatically declared correct.
        """

        branches = self.get_branches(conflict_id)

        if not branches:
            return {
                "conflict_id": conflict_id,
                "branch_count": 0,
                "branches": [],
                "status": "NOT_FOUND",
            }

        return {
            "conflict_id": conflict_id,
            "branch_count": len(branches),
            "branches": [
                {
                    "branch_id": branch.branch_id,
                    "interpretation": branch.interpretation,
                    "source": branch.source,
                    "context": branch.context_description,
                    "confidence": branch.confidence,
                }
                for branch in branches
            ],
            "status": (
                "CONFLICT_PRESERVED"
                if len(branches) > 1
                else "SINGLE_BRANCH"
            ),
        }
