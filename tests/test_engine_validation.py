
import pytest

from engine import UroborosEngine
from models import Context, EvidenceInput, EvidenceKind


@pytest.fixture
def engine(tmp_path):
    instance = UroborosEngine(
        database_path=str(tmp_path / "validation.db")
    )

    yield instance

    instance.close()


# ---------------------------------------------------------
# CONSTRUCTOR VALIDATION
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "bandwidth",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_engine_rejects_non_finite_bandwidth(
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
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_engine_rejects_non_finite_min_quality(
    tmp_path,
    min_quality,
):
    with pytest.raises(ValueError):
        UroborosEngine(
            database_path=str(tmp_path / "invalid.db"),
            min_quality=min_quality,
        )


def test_engine_rejects_empty_database_path():
    with pytest.raises(ValueError):
        UroborosEngine(database_path="   ")


def test_engine_rejects_boolean_bandwidth(tmp_path):
    with pytest.raises(ValueError):
        UroborosEngine(
            database_path=str(tmp_path / "invalid.db"),
            bandwidth=True,
        )


# ---------------------------------------------------------
# MEMORY INPUT VALIDATION
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "content",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_remember_rejects_invalid_content(engine, content):
    with pytest.raises(ValueError):
        engine.remember(
            content=content,
            context=Context({"environment": "test"}),
            source="test-source",
        )


@pytest.mark.parametrize(
    "source",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_remember_rejects_invalid_source(engine, source):
    with pytest.raises(ValueError):
        engine.remember(
            content="Test memory",
            context=Context({"environment": "test"}),
            source=source,
        )


def test_remember_rejects_invalid_context(engine):
    with pytest.raises(ValueError):
        engine.remember(
            content="Test memory",
            context={"environment": "test"},
            source="test-source",
        )


def test_recall_rejects_empty_trace_id(engine):
    with pytest.raises(ValueError):
        engine.recall("   ")


def test_recall_rejects_non_string_trace_id(engine):
    with pytest.raises(ValueError):
        engine.recall(123)


# ---------------------------------------------------------
# RETRIEVAL VALIDATION
# ---------------------------------------------------------

def test_recall_similar_rejects_invalid_context(engine):
    with pytest.raises(ValueError):
        engine.recall_similar(
            {"environment": "test"}
        )


@pytest.mark.parametrize(
    "limit",
    [
        0,
        -1,
        1.5,
        True,
        None,
    ],
)
def test_recall_similar_rejects_invalid_limit(engine, limit):
    with pytest.raises(ValueError):
        engine.recall_similar(
            Context({"environment": "test"}),
            limit=limit,
        )


# ---------------------------------------------------------
# EVIDENCE INPUT VALIDATION
# ---------------------------------------------------------

def test_observe_rejects_wrong_object_type(engine):
    with pytest.raises(ValueError):
        engine.observe("not-an-evidence-record")


def test_evaluate_association_rejects_empty_id(engine):
    with pytest.raises(ValueError):
        engine.evaluate_association(
            association_id="   ",
            current_context=Context({"environment": "test"}),
            predictions=[],
        )


def test_evaluate_association_rejects_invalid_context(engine):
    with pytest.raises(ValueError):
        engine.evaluate_association(
            association_id="association-001",
            current_context={"environment": "test"},
            predictions=[],
        )


def test_evaluate_association_rejects_non_finite_prediction(
    engine,
):
    context = Context({"environment": "test"})

    evidence = EvidenceInput(
        association_id="association-001",
        context=context,
        kind=EvidenceKind.POSITIVE,
        outcome=True,
        quality=0.9,
        source="validation-test",
    )

    engine.observe(evidence)

    with pytest.raises(ValueError):
        engine.evaluate_association(
            association_id="association-001",
            current_context=context,
            predictions=[float("nan")],
            min_ess=0.0,
            min_weight_sum=0.0,
        )


def test_evaluate_association_rejects_infinite_prediction(
    engine,
):
    context = Context({"environment": "test"})

    evidence = EvidenceInput(
        association_id="association-001",
        context=context,
        kind=EvidenceKind.POSITIVE,
        outcome=True,
        quality=0.9,
        source="validation-test",
    )

    engine.observe(evidence)

    with pytest.raises(ValueError):
        engine.evaluate_association(
            association_id="association-001",
            current_context=context,
            predictions=[float("inf")],
            min_ess=0.0,
            min_weight_sum=0.0,
        )
