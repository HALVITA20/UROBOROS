
import pytest

from engine import UroborosEngine
from models import (
    Context,
    EvidenceInput,
    EvidenceKind,
    BoundaryState,
    ProbeCandidate,
)


@pytest.fixture
def engine(tmp_path):
    database_path = tmp_path / "test_uroboros.db"

    instance = UroborosEngine(
        database_path=str(database_path),
        bandwidth=0.5,
        min_quality=0.5,
    )

    yield instance

    instance.close()


def make_evidence(
    association_id="association-001",
    kind=EvidenceKind.POSITIVE,
    outcome=True,
    quality=0.9,
    context=None,
    source="test-source",
):
    return EvidenceInput(
        association_id=association_id,
        context=context or Context({"environment": "test"}),
        kind=kind,
        outcome=outcome,
        quality=quality,
        source=source,
    )


# ---------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------

def test_engine_initializes(engine):
    assert engine.database is not None
    assert engine.bandwidth == 0.5
    assert engine.min_quality == 0.5


@pytest.mark.parametrize(
    "bandwidth",
    [0, -1],
)
def test_engine_rejects_non_positive_bandwidth(
    tmp_path,
    bandwidth,
):
    with pytest.raises(ValueError):
        UroborosEngine(
            database_path=str(tmp_path / "invalid.db"),
            bandwidth=bandwidth,
        )


@pytest.mark.parametrize(
    "min_quality",
    [-0.1, 1.1],
)
def test_engine_rejects_invalid_min_quality(
    tmp_path,
    min_quality,
):
    with pytest.raises(ValueError):
        UroborosEngine(
            database_path=str(tmp_path / "invalid.db"),
            min_quality=min_quality,
        )


# ---------------------------------------------------------
# TRACE MEMORY
# ---------------------------------------------------------

def test_remember_and_recall(engine):
    context = Context({
        "environment": "laboratory",
        "temperature": 22,
    })

    trace_id = engine.remember(
        content="Observed experimental event.",
        context=context,
        source="experiment-001",
    )

    assert isinstance(trace_id, str)
    assert trace_id

    restored = engine.recall(trace_id)

    assert restored is not None
    assert restored.trace_id == trace_id
    assert restored.content == "Observed experimental event."
    assert restored.source == "experiment-001"
    assert restored.context.features == context.features


def test_recall_unknown_trace_returns_none(engine):
    assert engine.recall("unknown-trace") is None


@pytest.mark.parametrize(
    "content, source",
    [
        ("", "source"),
        ("   ", "source"),
        ("content", ""),
        ("content", "   "),
    ],
)
def test_remember_rejects_empty_fields(
    engine,
    content,
    source,
):
    with pytest.raises(ValueError):
        engine.remember(
            content=content,
            context=Context({"environment": "test"}),
            source=source,
        )


def test_remember_rejects_invalid_context_serialization(engine):
    context = Context({
        "invalid_number": float("nan"),
    })

    with pytest.raises(ValueError):
        engine.remember(
            content="Invalid context",
            context=context,
            source="test",
        )


# ---------------------------------------------------------
# CONTEXTUAL RETRIEVAL
# ---------------------------------------------------------

def test_recall_similar_returns_matching_trace(engine):
    context = Context({
        "environment": "laboratory",
        "temperature": 22,
    })

    trace_id = engine.remember(
        content="Matching trace",
        context=context,
        source="test",
    )

    results = engine.recall_similar(context)

    assert len(results) == 1
    assert results[0][0].trace_id == trace_id
    assert results[0][1] == pytest.approx(1.0)


def test_recall_similar_ranks_closer_context_first(engine):
    target = Context({"temperature": 20})

    near = Context({"temperature": 21})
    far = Context({"temperature": 40})

    near_id = engine.remember(
        "Near observation",
        near,
        "source-near",
    )

    engine.remember(
        "Far observation",
        far,
        "source-far",
    )

    results = engine.recall_similar(target)

    assert len(results) == 2
    assert results[0][0].trace_id == near_id
    assert results[0][1] > results[1][1]


def test_recall_similar_excludes_unrelated_context(engine):
    engine.remember(
        "Location observation",
        Context({"location": "forest"}),
        "source",
    )

    results = engine.recall_similar(
        Context({"temperature": 20})
    )

    assert results == []


def test_recall_similar_rejects_invalid_limit(engine):
    with pytest.raises(ValueError):
        engine.recall_similar(
            Context({"temperature": 20}),
            limit=0,
        )


# ---------------------------------------------------------
# EVIDENCE STORAGE
# ---------------------------------------------------------

def test_observe_stores_valid_positive_evidence(engine):
    evidence_id = engine.observe(
        make_evidence()
    )

    records = engine.database.list_evidence(
        "association-001"
    )

    assert evidence_id
    assert len(records) == 1
    assert records[0].evidence_id == evidence_id
    assert records[0].kind is EvidenceKind.POSITIVE
    assert records[0].outcome is True


def test_observe_rejects_evidence_below_quality_threshold(engine):
    evidence = make_evidence(
        quality=0.2,
    )

    with pytest.raises(ValueError):
        engine.observe(evidence)

    assert engine.database.list_evidence(
        "association-001"
    ) == []


# ---------------------------------------------------------
# ASSOCIATION BOUNDARY EVALUATION
# ---------------------------------------------------------

def test_evaluate_association_returns_positive_boundary(engine):
    context = Context({"environment": "test"})

    engine.observe(
        make_evidence(
            context=context,
        )
    )

    result = engine.evaluate_association(
        association_id="association-001",
        current_context=context,
        predictions=[0.95],
        min_ess=0.0,
        min_weight_sum=0.0,
    )

    assert result.state is BoundaryState.POSITIVE
    assert result.lower_quantile == pytest.approx(0.95)
    assert result.upper_quantile == pytest.approx(0.95)


def test_evaluate_association_requires_one_prediction_per_record(
    engine,
):
    context = Context({"environment": "test"})

    engine.observe(
        make_evidence(context=context)
    )

    with pytest.raises(ValueError):
        engine.evaluate_association(
            association_id="association-001",
            current_context=context,
            predictions=[],
        )


def test_inconclusive_evidence_does_not_support_boundary(engine):
    context = Context({"environment": "test"})

    engine.observe(
        make_evidence(
            context=context,
            kind=EvidenceKind.INCONCLUSIVE,
            outcome=None,
        )
    )

    result = engine.evaluate_association(
        association_id="association-001",
        current_context=context,
        predictions=[0.99],
        min_ess=0.0,
        min_weight_sum=0.0,
    )

    assert result.state is BoundaryState.UNKNOWN
    assert result.support_count == 0
    assert result.local_weight_sum == 0.0


# ---------------------------------------------------------
# ACTIVE CONTEXT PROBING
# ---------------------------------------------------------

def test_probe_returns_valid_candidate(engine):
    candidates = [
        ProbeCandidate(
            context_id="context-A",
            information_gain=0.8,
            task_value=0.7,
            cost=0.1,
            risk=0.1,
        ),
        ProbeCandidate(
            context_id="context-B",
            information_gain=0.4,
            task_value=0.5,
            cost=0.2,
            risk=0.2,
        ),
    ]

    decision = engine.probe(
        candidates,
        temperature=1.0,
    )

    assert decision.context_id in {
        "context-A",
        "context-B",
    }

    assert 0.0 < decision.propensity <= 1.0


# ---------------------------------------------------------
# CONFLICT PRESERVATION
# ---------------------------------------------------------

def test_engine_preserves_multiple_conflict_branches(engine):
    first = engine.add_conflict(
        conflict_id="conflict-001",
        interpretation="Interpretation A",
        source="source-A",
        confidence=0.7,
    )

    second = engine.add_conflict(
        conflict_id="conflict-001",
        interpretation="Interpretation B",
        source="source-B",
        confidence=0.6,
    )

    branches = engine.conflicts.get_branches(
        "conflict-001"
    )

    assert len(branches) == 2
    assert first.branch_id != second.branch_id

    comparison = engine.conflicts.compare_branches(
        "conflict-001"
    )

    assert comparison["branch_count"] == 2
    assert comparison["status"] == "CONFLICT_PRESERVED"


# ---------------------------------------------------------
# PERSISTENCE
# ---------------------------------------------------------

def test_memory_survives_engine_reinitialization(tmp_path):
    database_path = tmp_path / "persistent.db"

    first_engine = UroborosEngine(
        database_path=str(database_path)
    )

    context = Context({"environment": "persistent"})

    trace_id = first_engine.remember(
        content="Persistent memory",
        context=context,
        source="persistence-test",
    )

    first_engine.close()

    second_engine = UroborosEngine(
        database_path=str(database_path)
    )

    try:
        restored = second_engine.recall(trace_id)

        assert restored is not None
        assert restored.content == "Persistent memory"
        assert restored.source == "persistence-test"
    finally:
        second_engine.close()


# ---------------------------------------------------------
# CONTEXT MANAGER
# ---------------------------------------------------------

def test_engine_supports_context_manager(tmp_path):
    database_path = tmp_path / "context-manager.db"

    with UroborosEngine(
        database_path=str(database_path)
    ) as instance:
        assert instance.database is not None
