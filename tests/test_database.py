
import pytest

from models import (
    Context,
    TraceRecord,
    EvidenceInput,
    EvidenceKind,
)
from database import UroborosDatabase


@pytest.fixture
def database(tmp_path):
    db = UroborosDatabase(tmp_path / "test_uroboros.db")
    yield db
    db.close()


def test_save_and_retrieve_trace(database):
    trace = TraceRecord(
        trace_id="trace-001",
        content="Test memory",
        context=Context({
            "location": "forest",
            "temperature": 20,
        }),
        source="unit-test",
    )

    database.save_trace(trace)

    restored = database.get_trace("trace-001")

    assert restored is not None
    assert restored.content == "Test memory"
    assert restored.context.features["location"] == "forest"
    assert restored.source == "unit-test"


def test_missing_trace_returns_none(database):
    assert database.get_trace("missing") is None


def test_list_traces(database):
    for index in range(3):
        database.save_trace(
            TraceRecord(
                trace_id=f"trace-{index}",
                content=f"Memory {index}",
                context=Context({"index": index}),
                source="test",
            )
        )

    traces = database.list_traces(limit=2)

    assert len(traces) == 2


def test_save_and_retrieve_positive_evidence(database):
    evidence = EvidenceInput(
        association_id="association-001",
        context=Context({"location": "forest"}),
        kind=EvidenceKind.POSITIVE,
        outcome=True,
        quality=0.95,
        source="experiment-001",
    )

    evidence_id = database.save_evidence(evidence)

    records = database.list_evidence("association-001")

    assert len(records) == 1
    assert records[0].evidence_id == evidence_id
    assert records[0].kind == EvidenceKind.POSITIVE
    assert records[0].outcome is True


def test_save_negative_evidence(database):
    evidence = EvidenceInput(
        association_id="association-002",
        context=Context({"location": "desert"}),
        kind=EvidenceKind.NEGATIVE,
        outcome=False,
        quality=0.9,
        source="experiment-002",
    )

    database.save_evidence(evidence)

    records = database.list_evidence("association-002")

    assert records[0].kind == EvidenceKind.NEGATIVE
    assert records[0].outcome is False


def test_invalid_evidence_is_not_saved(database):
    evidence = EvidenceInput(
        association_id="association-003",
        context=Context({"location": "forest"}),
        kind=EvidenceKind.NEGATIVE,
        outcome=False,
        quality=0.9,
        source="failed-test",
        retrieval_succeeded=False,
    )

    with pytest.raises(ValueError):
        database.save_evidence(evidence)

    assert database.list_evidence("association-003") == []


def test_metadata_round_trip(database):
    database.set_metadata(
        "engine_version",
        {"version": "2.0.0"},
    )

    result = database.get_metadata("engine_version")

    assert result == {"version": "2.0.0"}


def test_missing_metadata_returns_none(database):
    assert database.get_metadata("missing") is None


def test_invalid_limit_is_rejected(database):
    with pytest.raises(ValueError):
        database.list_traces(limit=0)

    with pytest.raises(ValueError):
        database.list_evidence("association-001", limit=0)
