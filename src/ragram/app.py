"""Top-level application orchestration for RagRam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .chunking import chunk_messages
from .config import default_config, ensure_app_layout, load_config, save_config, scaffold_status
from .models import AppPaths, IndexRunRecord, IndexScope, IngestionResult, ScaffoldStatus
from .progress import NullProgressReporter, ProgressReporter
from .storage import SQLiteStore
from .telegram_client import TelegramFloodWait, iter_text_message_records


@dataclass(frozen=True)
class StartPlan:
    """Description of the local start command plan."""

    no_ui: bool
    ui_port: int
    status: ScaffoldStatus
    paths: AppPaths


@dataclass(frozen=True)
class IndexingPipelineResult:
    """Summary of converting stored raw messages into local vector chunks."""

    raw_messages: int
    chunks_created: int
    chunks_indexed: int


def build_start_plan(*, no_ui: bool, ui_port: int) -> StartPlan:
    """Create local app layout, config, SQLite schema, and build a start plan."""

    paths = ensure_app_layout()
    store = SQLiteStore(paths.sqlite_path)
    store.initialize()
    if not paths.config_path.exists():
        save_config(default_config(), paths.config_path)
    else:
        # Normalize older configs to the current schema without dropping values.
        save_config(load_config(paths.config_path), paths.config_path)
    return StartPlan(no_ui=no_ui, ui_port=ui_port, status=scaffold_status(paths.home), paths=paths)


def index_stored_messages(
    store: SQLiteStore,
    *,
    vector_store: Any,
    entity_id: int,
    embedding_model: str,
    progress: ProgressReporter | None = None,
) -> IndexingPipelineResult:
    """Chunk raw SQLite messages, persist chunks, and upsert them into Chroma.

    This function is intentionally deterministic and fakeable so CLI smoke tests
    can exercise the end-to-end indexing path without Telegram, Chroma, or paid
    APIs.
    """

    reporter = progress or NullProgressReporter()
    messages = store.messages_after(entity_id)
    reporter.start(total=len(messages))
    chunks = chunk_messages(messages, entity_id=entity_id, embedding_model=embedding_model)
    active_ids = {chunk.chunk_id for chunk in chunks}
    stale_ids = store.stale_chunk_ids_for_model(entity_id=entity_id, embedding_model=embedding_model, keep_chunk_ids=active_ids)
    if stale_ids:
        delete_chunks = getattr(vector_store, "delete_chunks", None)
        if delete_chunks is None:
            raise TypeError("vector_store must implement delete_chunks for stale chunk reconciliation")
        delete_chunks(stale_ids, entity_id=entity_id)
        store.delete_chunks_by_ids(stale_ids)
    for chunk in chunks:
        store.upsert_chunk(chunk)
    chunks_indexed = vector_store.upsert_chunks(chunks, entity_id=entity_id) if chunks else 0
    reporter.finish(chunks_indexed)
    return IndexingPipelineResult(
        raw_messages=len(messages),
        chunks_created=len(chunks),
        chunks_indexed=chunks_indexed,
    )


async def ingest_text_messages(
    client: Any,
    store: SQLiteStore,
    *,
    entity: Any,
    entity_id: int,
    scope: IndexScope,
    run_id: str,
    embedding_model: str,
    channel_username: str | None = None,
    progress: ProgressReporter | None = None,
) -> IngestionResult:
    """Fetch Telegram text messages and persist raw rows before indexing.

    The function is intentionally raw-first and dedupe-safe. Re-running with the
    same messages updates existing rows and reports only newly inserted rows as
    saved, which makes interrupted ingestion safe to retry.
    """

    reporter = progress or NullProgressReporter()
    resume_cursor = store.latest_failed_index_cursor(
        entity_id=entity_id,
        scope_type=scope.scope_type,
        scope_value=scope.scope_value,
        embedding_model=embedding_model,
    )
    store.upsert_index_run(
        IndexRunRecord(
            id=run_id,
            entity_id=entity_id,
            scope_type=scope.scope_type,
            scope_value=scope.scope_value,
            embedding_model=embedding_model,
            status="running",
        )
    )
    reporter.start(total=scope.limit)

    fetched = 0
    saved = 0
    last_message_id: int | None = None
    try:
        async for message in iter_text_message_records(
            client,
            entity,
            entity_id=entity_id,
            scope=scope,
            channel_username=channel_username,
            resume_before_message_id=resume_cursor,
        ):
            if resume_cursor is not None and message.message_id >= resume_cursor and store.get_message(entity_id, message.message_id) is not None:
                continue
            fetched += 1
            inserted = store.upsert_message(message)
            if inserted:
                saved += 1
            last_message_id = message.message_id
            store.update_index_run(run_id, status="running", last_message_id=last_message_id)
            reporter.advance(1)
    except TelegramFloodWait as exc:
        store.update_index_run(run_id, status="failed", last_message_id=last_message_id, error=str(exc))
        reporter.fail(str(exc))
        raise
    except Exception as exc:
        store.update_index_run(run_id, status="failed", last_message_id=last_message_id, error=str(exc))
        reporter.fail(str(exc))
        raise

    store.update_index_run(run_id, status="complete", last_message_id=last_message_id, error=None)
    reporter.finish(saved)
    return IngestionResult(fetched_messages=fetched, saved_messages=saved, last_message_id=last_message_id)
