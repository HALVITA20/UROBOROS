
import pytest

from conflicts import HonestConflictResolution


@pytest.fixture
def resolver():
    return HonestConflictResolution()


def test_add_conflict_branch(resolver):
    branch = resolver.add_branch(
        conflict_id="conflict-001",
        interpretation="Interpretation A",
        source="source-A",
        confidence=0.8,
    )

    assert branch.conflict_id == "conflict-001"
    assert branch.interpretation == "Interpretation A"
    assert branch.confidence == 0.8


def test_preserve_multiple_interpretations(resolver):
    resolver.add_branch(
        "conflict-001",
        "Interpretation A",
        "source-A",
    )

    resolver.add_branch(
        "conflict-001",
        "Interpretation B",
        "source-B",
    )

    branches = resolver.get_branches("conflict-001")

    assert len(branches) == 2
    assert branches[0].interpretation != branches[1].interpretation


def test_conflict_comparison_reports_preserved_branches(resolver):
    resolver.add_branch(
        "conflict-002",
        "First interpretation",
        "source-A",
    )

    resolver.add_branch(
        "conflict-002",
        "Second interpretation",
        "source-B",
    )

    result = resolver.compare_branches("conflict-002")

    assert result["branch_count"] == 2
    assert result["status"] == "CONFLICT_PRESERVED"


def test_missing_conflict_returns_not_found(resolver):
    result = resolver.compare_branches("missing")

    assert result["branch_count"] == 0
    assert result["status"] == "NOT_FOUND"


def test_list_conflicts(resolver):
    resolver.add_branch("conflict-B", "B", "source-B")
    resolver.add_branch("conflict-A", "A", "source-A")

    assert resolver.list_conflicts() == (
        "conflict-A",
        "conflict-B",
    )


def test_remove_existing_branch(resolver):
    branch = resolver.add_branch(
        "conflict-003",
        "Interpretation A",
        "source-A",
    )

    removed = resolver.remove_branch(
        "conflict-003",
        branch.branch_id,
    )

    assert removed is True
    assert resolver.get_branches("conflict-003") == ()


def test_remove_missing_branch_returns_false(resolver):
    assert resolver.remove_branch(
        "conflict-004",
        "missing-branch",
    ) is False


@pytest.mark.parametrize(
    "confidence",
    [-0.1, 1.1],
)
def test_invalid_confidence_is_rejected(resolver, confidence):
    with pytest.raises(ValueError):
        resolver.add_branch(
            "conflict-005",
            "Interpretation",
            "source",
            confidence=confidence,
        )


def test_empty_interpretation_is_rejected(resolver):
    with pytest.raises(ValueError):
        resolver.add_branch(
            "conflict-006",
            "",
            "source",
        )


def test_empty_source_is_rejected(resolver):
    with pytest.raises(ValueError):
        resolver.add_branch(
            "conflict-007",
            "Interpretation",
            "",
        )
