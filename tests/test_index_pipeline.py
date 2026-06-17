from __future__ import annotations

from datetime import UTC, datetime

from ragram.app import index_stored_messages
from ragram.models import MessageRecord
from ragram.storage import SQLiteStore


class FakeVectorStore:
    def __init__(self):
        self.calls = []
        self.deletes = []

    def upsert_chunks(self, chunks, *, entity_id):
        self.calls.append({"chunks": chunks, "entity_id": entity_id})
        return len(chunks)

    def delete_chunks(self, chunk_ids, *, entity_id):
        self.deletes.append({"chunk_ids": chunk_ids, "entity_id": entity_id})
        return len(chunk_ids)


def test_index_stored_messages_chunks_raw_messages_and_upserts_sqlite_and_vector_store(tmp_path):
    store = SQLiteStore(tmp_path / "ragram.sqlite")
    store.initialize()
    for message_id in range(1, 5):
        store.upsert_message(
            MessageRecord(
                entity_id=100,
                message_id=message_id,
                date=datetime(2025, 1, message_id, tzinfo=UTC),
                text=f"сообщение {message_id} с текстом",
            )
        )
    vector_store = FakeVectorStore()

    result = index_stored_messages(
        store,
        vector_store=vector_store,
        entity_id=100,
        embedding_model="intfloat/multilingual-e5-small",
    )

    assert result.raw_messages == 4
    assert result.chunks_created >= 1
    assert result.chunks_indexed == result.chunks_created
    assert store.list_chunks(entity_id=100, embedding_model="intfloat/multilingual-e5-small")
    assert vector_store.calls[0]["entity_id"] == 100


def test_index_stored_messages_reconciles_stale_chunks(tmp_path):
    store = SQLiteStore(tmp_path / "ragram.sqlite")
    store.initialize()
    store.upsert_message(MessageRecord(entity_id=100, message_id=1, date=datetime(2025, 1, 1, tzinfo=UTC), text="new text"))
    stale = MessageRecord(entity_id=100, message_id=99, date=datetime(2025, 1, 2, tzinfo=UTC), text="stale")
    # Seed a stale chunk by indexing a temporary stale message, then remove it from raw source.
    store.upsert_message(stale)
    stale_vector = FakeVectorStore()
    index_stored_messages(store, vector_store=stale_vector, entity_id=100, embedding_model="model-a")
    with store.connect() as connection:
        connection.execute("DELETE FROM messages WHERE entity_id = ? AND message_id = ?", (100, 99))

    result = index_stored_messages(store, vector_store=FakeVectorStore(), entity_id=100, embedding_model="model-a")

    assert result.chunks_created == 1
    chunks = store.list_chunks(entity_id=100, embedding_model="model-a")
    assert len(chunks) == 1
    assert chunks[0].message_id_start == 1


def test_index_stored_messages_requires_vector_delete_contract_for_stale_chunks(tmp_path):
    store = SQLiteStore(tmp_path / "ragram.sqlite")
    store.initialize()
    store.upsert_message(MessageRecord(entity_id=100, message_id=1, date=datetime(2025, 1, 1, tzinfo=UTC), text="one"))
    stale_vector = FakeVectorStore()
    index_stored_messages(store, vector_store=stale_vector, entity_id=100, embedding_model="model-a")
    with store.connect() as connection:
        connection.execute("DELETE FROM messages WHERE entity_id = ?", (100,))

    try:
        index_stored_messages(store, vector_store=type("NoDeleteVectorStore", (), {"upsert_chunks": lambda self, chunks, *, entity_id: len(chunks)})(), entity_id=100, embedding_model="model-a")
    except TypeError as exc:
        assert "delete_chunks" in str(exc)
    else:
        raise AssertionError("stale vector cleanup must require an explicit delete_chunks contract")


def test_index_stored_messages_deletes_vectors_before_sqlite_metadata(tmp_path):
    store = SQLiteStore(tmp_path / "ragram.sqlite")
    store.initialize()
    store.upsert_message(MessageRecord(entity_id=100, message_id=1, date=datetime(2025, 1, 1, tzinfo=UTC), text="one"))
    vector = FakeVectorStore()
    index_stored_messages(store, vector_store=vector, entity_id=100, embedding_model="model-a")
    with store.connect() as connection:
        connection.execute("DELETE FROM messages WHERE entity_id = ?", (100,))

    class FailingDeleteVectorStore(FakeVectorStore):
        def delete_chunks(self, chunk_ids, *, entity_id):
            raise RuntimeError("vector delete failed")

    try:
        index_stored_messages(store, vector_store=FailingDeleteVectorStore(), entity_id=100, embedding_model="model-a")
    except RuntimeError as exc:
        assert "vector delete failed" in str(exc)
    else:
        raise AssertionError("vector delete failure should stop SQLite metadata deletion")

    assert store.list_chunks(entity_id=100, embedding_model="model-a")
