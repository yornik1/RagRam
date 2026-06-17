"""Shared lightweight data models for RagRam."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AppPaths:
    """Filesystem paths for RagRam's local-first app home."""

    home: Path
    config_path: Path
    sessions_dir: Path
    data_dir: Path
    sqlite_path: Path
    chroma_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class ScaffoldStatus:
    """Current local status that does not contact external services."""

    app_home: Path
    config_path: Path
    sqlite_path: Path
    has_config: bool
    has_sqlite: bool
    uses_dotenv: bool = False
    no_paid_apis: bool = True


@dataclass(frozen=True)
class PromptSpec:
    """Prompt metadata used to keep secrets hidden in interactive flows."""

    key: str
    message: str
    secret: bool = False


@dataclass(frozen=True)
class ChannelRecord:
    """Raw Telegram channel/group metadata persisted in SQLite."""

    entity_id: int
    title: str
    username: str | None = None
    kind: str = "channel"
    last_message_date: datetime | None = None
    unread_count: int | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class MessageRecord:
    """Raw Telegram text message persisted before embedding."""

    entity_id: int
    message_id: int
    date: datetime
    text: str
    sender_id: str | None = None
    sender_name: str | None = None
    reply_to_id: int | None = None
    forward_info: dict[str, Any] | None = None
    message_link: str | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class IndexRunRecord:
    """Indexing run state used for resumability."""

    id: str
    entity_id: int
    scope_type: str
    scope_value: str | None
    embedding_model: str
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    last_message_id: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class ChunkRecord:
    """Chunk metadata persisted alongside Chroma vectors."""

    chunk_id: str
    entity_id: int
    embedding_model: str
    message_id_start: int
    message_id_end: int
    date_start: datetime
    date_end: datetime
    text: str
    token_count: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class IndexScope:
    """User-selected Telegram message indexing scope."""

    scope_type: str
    scope_value: str | None = None

    @classmethod
    def last_n(cls, count: int) -> "IndexScope":
        if count <= 0:
            raise ValueError("last_n count must be positive")
        return cls("last_n", str(count))

    @classmethod
    def all(cls) -> "IndexScope":
        return cls("all", None)

    @classmethod
    def from_year(cls, year: int) -> "IndexScope":
        return cls("from_year", str(year))

    @classmethod
    def from_date(cls, value: str) -> "IndexScope":
        return cls("from_date", value)

    @property
    def limit(self) -> int | None:
        if self.scope_type == "last_n" and self.scope_value is not None:
            return int(self.scope_value)
        return None

    @property
    def start_date(self) -> datetime | None:
        if self.scope_type == "from_year" and self.scope_value is not None:
            return datetime(int(self.scope_value), 1, 1, tzinfo=UTC)
        if self.scope_type == "from_date" and self.scope_value is not None:
            parsed = datetime.fromisoformat(self.scope_value)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed
        return None


@dataclass(frozen=True)
class IngestionResult:
    """Summary of a raw Telegram message ingestion run."""

    fetched_messages: int
    saved_messages: int
    last_message_id: int | None = None
