
from models import Context, TraceRecord
from engine import UroborosEngine


def test_recall_similar_searches_beyond_10000_records(
    tmp_path,
    monkeypatch,
):
    engine = UroborosEngine(
        database_path=str(tmp_path / "full-recall.db")
    )

    try:
        target_context = Context({"environment": "target"})

        traces = [
            TraceRecord(
                trace_id=f"trace-{index:05d}",
                content=f"Memory {index}",
                context=Context({"environment": "other"}),
                source="test",
                created_at=float(index),
            )
            for index in range(10000)
        ]

        traces.append(
            TraceRecord(
                trace_id="trace-target",
                content="The best matching memory",
                context=target_context,
                source="test",
                created_at=10000.0,
            )
        )

        requested_offsets = []

        def fake_list_traces(limit=100, offset=0):
            requested_offsets.append(offset)
            return traces[offset:offset + limit]

        monkeypatch.setattr(
            engine.database,
            "list_traces",
            fake_list_traces,
        )

        results = engine.recall_similar(
            context=target_context,
            limit=1,
        )

        assert len(results) == 1
        assert results[0][0].trace_id == "trace-target"

        assert 10000 in requested_offsets
        assert requested_offsets[-1] == 10001

    finally:
        engine.close()


def test_recall_similar_returns_empty_for_empty_database(
    tmp_path,
):
    engine = UroborosEngine(
        database_path=str(tmp_path / "empty-recall.db")
    )

    try:
        results = engine.recall_similar(
            context=Context({"environment": "test"}),
        )

        assert results == []

    finally:
        engine.close()


def test_recall_similar_keeps_only_requested_number_of_results(
    tmp_path,
):
    engine = UroborosEngine(
        database_path=str(tmp_path / "limited-recall.db")
    )

    try:
        context = Context({"environment": "test"})

        for index in range(20):
            engine.remember(
                content=f"Memory {index}",
                context=context,
                source="test",
            )

        results = engine.recall_similar(
            context=context,
            limit=3,
        )

        assert len(results) == 3
        assert all(weight == 1.0 for _, weight in results)

    finally:
        engine.close()
