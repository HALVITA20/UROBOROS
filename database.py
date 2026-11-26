
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from models import (
    Context,
    EvidenceInput,
    EvidenceRecord,
    TraceRecord,
    new_id,
)
from evidence import validate_evidence


class UroborosDatabase:
    """SQLite persistence layer for UROBOROS."""

    def __init__(self, path: str | Path = "uroboros.db"):
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self.connection:
            self.connection.executescript("""
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    association_id TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    outcome INTEGER,
                    quality REAL NOT NULL,
                    source TEXT NOT NULL,
                    source_family TEXT,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_evidence_association
                ON evidence(association_id);

                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)

    @staticmethod
    def _serialize_context(context: Context) -> str:
        return json.dumps({
            "features": dict(context.features),
            "observed": (
                sorted(context.observed)
                if context.observed is not None
                else None
            ),
        }, ensure_ascii=False, allow_nan=False)

    @staticmethod
    def _deserialize_context(data: str) -> Context:
        obj = json.loads(data)
        observed = obj.get("observed")

        return Context(
            features=obj["features"],
            observed=(
                frozenset(observed)
                if observed is not None
                else None
            ),
        )

    def save_trace(self, trace: TraceRecord) -> str:
        with self.connection:
            self.connection.execute("""
                INSERT INTO traces
                (trace_id, content, context_json, source, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                trace.trace_id,
                trace.content,
                self._serialize_context(trace.context),
                trace.source,
                trace.created_at,
            ))

        return trace.trace_id

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        row = self.connection.execute("""
            SELECT * FROM traces WHERE trace_id = ?
        """, (trace_id,)).fetchone()

        if row is None:
            return None

        return TraceRecord(
            trace_id=row["trace_id"],
            content=row["content"],
            context=self._deserialize_context(row["context_json"]),
            source=row["source"],
            created_at=row["created_at"],
        )

    def list_traces(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TraceRecord]:
        """
        Return traces in deterministic newest-first order.

        Offset enables callers to retrieve the database in pages.
        """

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
        ):
            raise ValueError("limit must be a positive integer")

        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
        ):
            raise ValueError("offset must be a non-negative integer")

        rows = self.connection.execute("""
            SELECT * FROM traces
            ORDER BY created_at DESC, trace_id DESC
            LIMIT ? OFFSET ?
        """, (limit, offset)).fetchall()

        return [
            TraceRecord(
                trace_id=row["trace_id"],
                content=row["content"],
                context=self._deserialize_context(row["context_json"]),
                source=row["source"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def save_evidence(
        self,
        item: EvidenceInput,
        min_quality: float = 0.5,
    ) -> str:
        validate_evidence(item, min_quality=min_quality)

        evidence_id = new_id()
        created_at = time.time()

        with self.connection:
            self.connection.execute("""
                INSERT INTO evidence
                (evidence_id, association_id, context_json,
                 kind, outcome, quality, source, source_family, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evidence_id,
                item.association_id,
                self._serialize_context(item.context),
                item.kind.value,
                (
                    int(item.outcome)
                    if item.outcome is not None
                    else None
                ),
                item.quality,
                item.source,
                item.source_family,
                created_at,
            ))

        return evidence_id

    def list_evidence(
        self,
        association_id: str,
        limit: int = 1000,
    ) -> list[EvidenceRecord]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
        ):
            raise ValueError("limit must be a positive integer")

        rows = self.connection.execute("""
            SELECT * FROM evidence
            WHERE association_id = ?
            ORDER BY created_at ASC, evidence_id ASC
            LIMIT ?
        """, (association_id, limit)).fetchall()

        from models import EvidenceKind

        return [
            EvidenceRecord(
                evidence_id=row["evidence_id"],
                association_id=row["association_id"],
                context=self._deserialize_context(row["context_json"]),
                kind=EvidenceKind(row["kind"]),
                outcome=(
                    bool(row["outcome"])
                    if row["outcome"] is not None
                    else None
                ),
                quality=row["quality"],
                source=row["source"],
                source_family=row["source_family"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def set_metadata(self, key: str, value: Any) -> None:
        with self.connection:
            self.connection.execute("""
                INSERT INTO metadata (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (key, json.dumps(value, ensure_ascii=False)))

    def get_metadata(self, key: str) -> Any | None:
        row = self.connection.execute("""
            SELECT value FROM metadata WHERE key = ?
        """, (key,)).fetchone()

        return json.loads(row["value"]) if row else None

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> UroborosDatabase:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
