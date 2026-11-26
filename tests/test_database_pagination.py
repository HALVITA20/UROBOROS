
import pytest

from database import UroborosDatabase
from models import Context, TraceRecord


@pytest.fixture
def database(tmp_path):
    db = UroborosDatabase(tmp_path / "pagination.db")
    yield db
    db.close()


def add_trace(database, trace_id, created_at):
    database.save_trace(
        TraceRecord(
            trace_id=trace_id,
            content=f"Memory {trace_id}",
            context=Context({"environment": "test"}),
            source="pagination-test",
            created_at=created_at,
        )
    )


def test_list_traces_supports_offset(database):
    add_trace(database, "trace-1", 1.0)
    add_trace(database, "trace-2", 2.0)
    add_trace(database, "trace-3", 3.0)
    add_trace(database, "trace-4", 4.0)

    first_page = database.list_traces(limit=2, offset=0)
    second_page = database.list_traces(limit=2, offset=2)

    assert [item.trace_id for item in first_page] == [
        "trace-4",
        "trace-3",
    ]

    assert [item.trace_id for item in second_page] == [
        "trace-2",
        "trace-1",
    ]


def test_list_traces_pages_do_not_overlap(database):
    for index in range(7):
        add_trace(
            database,
            f"trace-{index}",
            float(index),
        )

    first_page = database.list_traces(limit=3, offset=0)
    second_page = database.list_traces(limit=3, offset=3)
    third_page = database.list_traces(limit=3, offset=6)

    first_ids = {item.trace_id for item in first_page}
    second_ids = {item.trace_id for item in second_page}
    third_ids = {item.trace_id for item in third_page}

    assert first_ids.isdisjoint(second_ids)
    assert first_ids.isdisjoint(third_ids)
    assert second_ids.isdisjoint(third_ids)

    assert len(first_page + second_page + third_page) == 7


def test_list_traces_has_deterministic_tie_order(database):
    add_trace(database, "trace-a", 10.0)
    add_trace(database, "trace-c", 10.0)
    add_trace(database, "trace-b", 10.0)

    traces = database.list_traces(limit=3)

    assert [item.trace_id for item in traces] == [
        "trace-c",
        "trace-b",
        "trace-a",
    ]


@pytest.mark.parametrize(
    "limit",
    [0, -1, True, 1.5, None],
)
def test_list_traces_rejects_invalid_limit(database, limit):
    with pytest.raises(ValueError):
        database.list_traces(limit=limit)


@pytest.mark.parametrize(
    "offset",
    [-1, True, 1.5, None],
)
def test_list_traces_rejects_invalid_offset(database, offset):
    with pytest.raises(ValueError):
        database.list_traces(offset=offset)


def test_list_traces_returns_empty_page_after_end(database):
    add_trace(database, "trace-1", 1.0)

    assert database.list_traces(limit=10, offset=10) == []
