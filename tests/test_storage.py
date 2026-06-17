from __future__ import annotations

import json
from datetime import UTC, datetime

from ragram.models import ChannelRecord, ChunkRecord, IndexRunRecord, MessageRecord
from ragram.storage import SQLiteStore


def make_store(tmp_path):
    store = SQLiteStore(tmp_path / "ragram.sqlite")
    store.initialize()
    return store


def test_initialize_creates_schema_tables(tmp_path):
    store = make_store(tmp_path)

    tables = set(store.table_names())

    assert {"channels", "messages", "index_runs", "chunks"}.issubset(tables)


def test_upsert_channel_preserves_metadata_and_updates_existing_row(tmp_path):
    store = make_store(tmp_path)
    first = ChannelRecord(
        entity_id=100,
        title="Old title",
        username="old",
        kind="channel",
        last_message_date=datetime(2024, 1, 1, tzinfo=UTC),
        unread_count=2,
        raw={"a": 1},
    )
    second = ChannelRecord(
        entity_id=100,
        title="New title",
        username="new",
        kind="megagroup",
        last_message_date=datetime(2025, 1, 1, tzinfo=UTC),
        unread_count=0,
        raw={"b": 2},
    )

    store.upsert_channel(first)
    store.upsert_channel(second)
    loaded = store.get_channel(100)

    assert loaded is not None
    assert loaded.title == "New title"
    assert loaded.username == "new"
    assert loaded.kind == "megagroup"
    assert loaded.raw == {"b": 2}
    assert store.channel_count() == 1


def test_upsert_message_dedupes_by_entity_and_message_id_and_preserves_metadata(tmp_path):
    store = make_store(tmp_path)
    original = MessageRecord(
        entity_id=100,
        message_id=5,
        date=datetime(2025, 1, 1, 12, 0, tzinfo=UTC),
        text="Первый текст",
        sender_id="42",
        sender_name="Alice",
        reply_to_id=4,
        forward_info={"from": "source"},
        message_link="https://t.me/example/5",
        raw={"views": 10},
    )
    updated = MessageRecord(
        entity_id=100,
        message_id=5,
        date=datetime(2025, 1, 1, 12, 0, tzinfo=UTC),
        text="Обновленный текст",
        sender_id="42",
        sender_name="Alice",
        reply_to_id=4,
        forward_info={"from": "source2"},
        message_link="https://t.me/example/5",
        raw={"views": 11},
    )

    inserted_first = store.upsert_message(original)
    inserted_second = store.upsert_message(updated)
    loaded = store.get_message(100, 5)

    assert inserted_first is True
    assert inserted_second is False
    assert store.message_count(100) == 1
    assert loaded is not None
    assert loaded.text == "Обновленный текст"
    assert loaded.forward_info == {"from": "source2"}
    assert loaded.raw == {"views": 11}


def test_messages_after_cursor_supports_resumable_ingestion(tmp_path):
    store = make_store(tmp_path)
    for message_id in [1, 2, 3]:
        store.upsert_message(
            MessageRecord(
                entity_id=100,
                message_id=message_id,
                date=datetime(2025, 1, message_id, tzinfo=UTC),
                text=f"message {message_id}",
            )
        )

    assert [message.message_id for message in store.messages_after(100, after_message_id=1)] == [2, 3]
    assert store.latest_message_id(100) == 3


def test_index_run_lifecycle_records_resume_cursor_and_error(tmp_path):
    store = make_store(tmp_path)
    run = IndexRunRecord(
        id="run-1",
        entity_id=100,
        scope_type="last_n",
        scope_value="1000",
        embedding_model="ai-forever/ru-en-RoSBERTa",
        status="running",
    )

    store.upsert_index_run(run)
    store.update_index_run("run-1", status="failed", last_message_id=55, error="FloodWait 17")
    loaded = store.get_index_run("run-1")

    assert loaded is not None
    assert loaded.status == "failed"
    assert loaded.last_message_id == 55
    assert loaded.error == "FloodWait 17"


def test_chunk_records_are_deduped_and_queryable_by_model(tmp_path):
    store = make_store(tmp_path)
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        entity_id=100,
        embedding_model="intfloat/multilingual-e5-small",
        message_id_start=1,
        message_id_end=3,
        date_start=datetime(2025, 1, 1, tzinfo=UTC),
        date_end=datetime(2025, 1, 3, tzinfo=UTC),
        text="chunk text",
        token_count=123,
        metadata={"message_ids": [1, 2, 3]},
    )

    assert store.upsert_chunk(chunk) is True
    assert store.upsert_chunk(chunk) is False
    chunks = store.list_chunks(entity_id=100, embedding_model="intfloat/multilingual-e5-small")

    assert len(chunks) == 1
    assert chunks[0].metadata == {"message_ids": [1, 2, 3]}
