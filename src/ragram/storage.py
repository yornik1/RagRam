"""SQLite persistence for RagRam raw messages, chunks, and index runs."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import ChannelRecord, ChunkRecord, IndexRunRecord, MessageRecord

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS channels (
    entity_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    username TEXT,
    type TEXT NOT NULL,
    last_message_date TEXT,
    unread_count INTEGER,
    raw_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    entity_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    sender_id TEXT,
    sender_name TEXT,
    reply_to_id INTEGER,
    forward_info TEXT,
    message_link TEXT,
    text TEXT NOT NULL,
    raw_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (entity_id, message_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_entity_date ON messages(entity_id, date);

CREATE TABLE IF NOT EXISTS index_runs (
    id TEXT PRIMARY KEY,
    entity_id INTEGER NOT NULL,
    scope_type TEXT NOT NULL,
    scope_value TEXT,
    embedding_model TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    last_message_id INTEGER,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_index_runs_entity_model ON index_runs(entity_id, embedding_model);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    entity_id INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    message_id_start INTEGER NOT NULL,
    message_id_end INTEGER NOT NULL,
    date_start TEXT NOT NULL,
    date_end TEXT NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_entity_model ON chunks(entity_id, embedding_model);
"""


def _now() -> datetime:
    return datetime.now(UTC)


def _dt(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _json(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def _loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}


class SQLiteStore:
    """Small SQLite repository with deterministic schema bootstrap."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            self.path.parent.chmod(0o700)
        connection = sqlite3.connect(self.path)
        if os.name == "posix":
            self.path.chmod(0o600)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA_SQL)

    def table_names(self) -> list[str]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return [str(row["name"]) for row in rows]

    def upsert_channel(self, channel: ChannelRecord) -> None:
        now = _dt(_now())
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO channels(entity_id, title, username, type, last_message_date, unread_count, raw_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    title=excluded.title,
                    username=excluded.username,
                    type=excluded.type,
                    last_message_date=excluded.last_message_date,
                    unread_count=excluded.unread_count,
                    raw_json=excluded.raw_json,
                    updated_at=excluded.updated_at
                """,
                (
                    channel.entity_id,
                    channel.title,
                    channel.username,
                    channel.kind,
                    _dt(channel.last_message_date),
                    channel.unread_count,
                    _json(channel.raw),
                    now,
                ),
            )

    def get_channel(self, entity_id: int) -> ChannelRecord | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM channels WHERE entity_id = ?", (entity_id,)).fetchone()
        if row is None:
            return None
        return ChannelRecord(
            entity_id=int(row["entity_id"]),
            title=str(row["title"]),
            username=row["username"],
            kind=str(row["type"]),
            last_message_date=_parse_dt(row["last_message_date"]),
            unread_count=row["unread_count"],
            raw=_loads(row["raw_json"]),
        )

    def channel_count(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) AS count FROM channels").fetchone()["count"])

    def upsert_message(self, message: MessageRecord) -> bool:
        now = _dt(_now())
        with self.connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM messages WHERE entity_id = ? AND message_id = ?",
                (message.entity_id, message.message_id),
            ).fetchone()
            created_at = now
            if exists is not None:
                current = connection.execute(
                    "SELECT created_at FROM messages WHERE entity_id = ? AND message_id = ?",
                    (message.entity_id, message.message_id),
                ).fetchone()
                created_at = current["created_at"]
            connection.execute(
                """
                INSERT INTO messages(
                    entity_id, message_id, date, sender_id, sender_name, reply_to_id,
                    forward_info, message_link, text, raw_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id, message_id) DO UPDATE SET
                    date=excluded.date,
                    sender_id=excluded.sender_id,
                    sender_name=excluded.sender_name,
                    reply_to_id=excluded.reply_to_id,
                    forward_info=excluded.forward_info,
                    message_link=excluded.message_link,
                    text=excluded.text,
                    raw_json=excluded.raw_json,
                    updated_at=excluded.updated_at
                """,
                (
                    message.entity_id,
                    message.message_id,
                    _dt(message.date),
                    message.sender_id,
                    message.sender_name,
                    message.reply_to_id,
                    _json(message.forward_info),
                    message.message_link,
                    message.text,
                    _json(message.raw),
                    created_at,
                    now,
                ),
            )
        return exists is None

    def _message_from_row(self, row: sqlite3.Row) -> MessageRecord:
        parsed_date = _parse_dt(row["date"])
        if parsed_date is None:  # schema prevents this; guard for type checkers.
            raise ValueError("message date is missing")
        return MessageRecord(
            entity_id=int(row["entity_id"]),
            message_id=int(row["message_id"]),
            date=parsed_date,
            text=str(row["text"]),
            sender_id=row["sender_id"],
            sender_name=row["sender_name"],
            reply_to_id=row["reply_to_id"],
            forward_info=_loads(row["forward_info"]),
            message_link=row["message_link"],
            raw=_loads(row["raw_json"]),
        )

    def get_message(self, entity_id: int, message_id: int) -> MessageRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM messages WHERE entity_id = ? AND message_id = ?",
                (entity_id, message_id),
            ).fetchone()
        return self._message_from_row(row) if row is not None else None

    def message_count(self, entity_id: int | None = None) -> int:
        with self.connect() as connection:
            if entity_id is None:
                row = connection.execute("SELECT COUNT(*) AS count FROM messages").fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM messages WHERE entity_id = ?",
                    (entity_id,),
                ).fetchone()
        return int(row["count"])

    def messages_after(self, entity_id: int, *, after_message_id: int | None = None) -> list[MessageRecord]:
        query = "SELECT * FROM messages WHERE entity_id = ?"
        params: list[Any] = [entity_id]
        if after_message_id is not None:
            query += " AND message_id > ?"
            params.append(after_message_id)
        query += " ORDER BY message_id ASC"
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._message_from_row(row) for row in rows]

    def latest_message_id(self, entity_id: int) -> int | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT MAX(message_id) AS latest FROM messages WHERE entity_id = ?",
                (entity_id,),
            ).fetchone()
        return row["latest"]

    def upsert_index_run(self, run: IndexRunRecord) -> None:
        started_at = run.started_at or _now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO index_runs(
                    id, entity_id, scope_type, scope_value, embedding_model,
                    started_at, completed_at, status, last_message_id, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    entity_id=excluded.entity_id,
                    scope_type=excluded.scope_type,
                    scope_value=excluded.scope_value,
                    embedding_model=excluded.embedding_model,
                    completed_at=excluded.completed_at,
                    status=excluded.status,
                    last_message_id=excluded.last_message_id,
                    error=excluded.error
                """,
                (
                    run.id,
                    run.entity_id,
                    run.scope_type,
                    run.scope_value,
                    run.embedding_model,
                    _dt(started_at),
                    _dt(run.completed_at),
                    run.status,
                    run.last_message_id,
                    run.error,
                ),
            )

    def update_index_run(
        self,
        run_id: str,
        *,
        status: str,
        last_message_id: int | None = None,
        error: str | None = None,
    ) -> None:
        completed_at = _dt(_now()) if status in {"complete", "failed"} else None
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE index_runs
                SET status = ?, last_message_id = COALESCE(?, last_message_id), error = ?, completed_at = COALESCE(?, completed_at)
                WHERE id = ?
                """,
                (status, last_message_id, error, completed_at, run_id),
            )


    def latest_failed_index_cursor(
        self,
        *,
        entity_id: int,
        scope_type: str,
        scope_value: str | None,
        embedding_model: str,
    ) -> int | None:
        """Return the newest failed-run cursor for retrying the same indexing scope."""

        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT last_message_id
                FROM index_runs
                WHERE entity_id = ?
                  AND scope_type = ?
                  AND COALESCE(scope_value, '') = COALESCE(?, '')
                  AND embedding_model = ?
                  AND status = 'failed'
                  AND last_message_id IS NOT NULL
                ORDER BY completed_at DESC, started_at DESC
                LIMIT 1
                """,
                (entity_id, scope_type, scope_value, embedding_model),
            ).fetchone()
        return None if row is None else int(row["last_message_id"])

    def get_index_run(self, run_id: str) -> IndexRunRecord | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM index_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return IndexRunRecord(
            id=str(row["id"]),
            entity_id=int(row["entity_id"]),
            scope_type=str(row["scope_type"]),
            scope_value=row["scope_value"],
            embedding_model=str(row["embedding_model"]),
            started_at=_parse_dt(row["started_at"]),
            completed_at=_parse_dt(row["completed_at"]),
            status=str(row["status"]),
            last_message_id=row["last_message_id"],
            error=row["error"],
        )

    def upsert_chunk(self, chunk: ChunkRecord) -> bool:
        now = _dt(_now())
        with self.connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM chunks WHERE chunk_id = ?",
                (chunk.chunk_id,),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO chunks(
                    chunk_id, entity_id, embedding_model, message_id_start, message_id_end,
                    date_start, date_end, text, token_count, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    entity_id=excluded.entity_id,
                    embedding_model=excluded.embedding_model,
                    message_id_start=excluded.message_id_start,
                    message_id_end=excluded.message_id_end,
                    date_start=excluded.date_start,
                    date_end=excluded.date_end,
                    text=excluded.text,
                    token_count=excluded.token_count,
                    metadata_json=excluded.metadata_json
                """,
                (
                    chunk.chunk_id,
                    chunk.entity_id,
                    chunk.embedding_model,
                    chunk.message_id_start,
                    chunk.message_id_end,
                    _dt(chunk.date_start),
                    _dt(chunk.date_end),
                    chunk.text,
                    chunk.token_count,
                    _json(chunk.metadata),
                    now,
                ),
            )
        return exists is None

    def _chunk_from_row(self, row: sqlite3.Row) -> ChunkRecord:
        date_start = _parse_dt(row["date_start"])
        date_end = _parse_dt(row["date_end"])
        if date_start is None or date_end is None:
            raise ValueError("chunk date range is missing")
        return ChunkRecord(
            chunk_id=str(row["chunk_id"]),
            entity_id=int(row["entity_id"]),
            embedding_model=str(row["embedding_model"]),
            message_id_start=int(row["message_id_start"]),
            message_id_end=int(row["message_id_end"]),
            date_start=date_start,
            date_end=date_end,
            text=str(row["text"]),
            token_count=int(row["token_count"]),
            metadata=_loads(row["metadata_json"]),
        )

    def list_chunks(self, *, entity_id: int, embedding_model: str) -> list[ChunkRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM chunks
                WHERE entity_id = ? AND embedding_model = ?
                ORDER BY message_id_start ASC, message_id_end ASC
                """,
                (entity_id, embedding_model),
            ).fetchall()
        return [self._chunk_from_row(row) for row in rows]


    def stale_chunk_ids_for_model(self, *, entity_id: int, embedding_model: str, keep_chunk_ids: set[str]) -> list[str]:
        """Return chunk IDs that are no longer active for an entity/model."""

        with self.connect() as connection:
            rows = connection.execute(
                "SELECT chunk_id FROM chunks WHERE entity_id = ? AND embedding_model = ?",
                (entity_id, embedding_model),
            ).fetchall()
        existing = {str(row["chunk_id"]) for row in rows}
        return sorted(existing - keep_chunk_ids)

    def delete_chunks_by_ids(self, chunk_ids: list[str]) -> int:
        """Delete chunk metadata after vector deletion has succeeded."""

        if not chunk_ids:
            return 0
        with self.connect() as connection:
            connection.executemany("DELETE FROM chunks WHERE chunk_id = ?", [(chunk_id,) for chunk_id in chunk_ids])
        return len(chunk_ids)

    def delete_chunks_for_model(self, *, entity_id: int, embedding_model: str, keep_chunk_ids: set[str] | None = None) -> list[str]:
        """Delete stale chunks for an entity/model and return deleted IDs."""

        with self.connect() as connection:
            rows = connection.execute(
                "SELECT chunk_id FROM chunks WHERE entity_id = ? AND embedding_model = ?",
                (entity_id, embedding_model),
            ).fetchall()
            existing = {str(row["chunk_id"]) for row in rows}
            keep = keep_chunk_ids or set()
            stale = sorted(existing - keep)
            if stale:
                connection.executemany("DELETE FROM chunks WHERE chunk_id = ?", [(chunk_id,) for chunk_id in stale])
        return stale
