from __future__ import annotations

import sys
import types
from datetime import UTC, datetime

from ragram.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODEL_CHOICES,
    EmbeddingProvider,
    FakeEmbeddingProvider,
    model_slug,
)
from ragram.models import ChunkRecord
from ragram.vector_store import ChromaVectorStore, RetrievalResult, collection_name


def chunk(chunk_id: str, text: str = "hello world") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        entity_id=100,
        embedding_model="ai-forever/ru-en-RoSBERTa",
        message_id_start=1,
        message_id_end=2,
        date_start=datetime(2025, 1, 1, tzinfo=UTC),
        date_end=datetime(2025, 1, 2, tzinfo=UTC),
        text=text,
        token_count=5,
        metadata={"message_ids": [1, 2]},
    )


class FakeCollection:
    def __init__(self, name):
        self.name = name
        self.upserts = []
        self.items = {}

    def upsert(self, *, ids, embeddings, documents, metadatas):
        self.upserts.append({"ids": ids, "embeddings": embeddings, "documents": documents, "metadatas": metadatas})
        for idx, item_id in enumerate(ids):
            self.items[item_id] = {
                "embedding": embeddings[idx],
                "document": documents[idx],
                "metadata": metadatas[idx],
            }

    def query(self, *, query_embeddings, n_results):
        ids = list(self.items)[:n_results]
        return {
            "ids": [ids],
            "documents": [[self.items[item_id]["document"] for item_id in ids]],
            "metadatas": [[self.items[item_id]["metadata"] for item_id in ids]],
            "distances": [[0.1 + index for index, _ in enumerate(ids)]],
        }

    def count(self):
        return len(self.items)


class FakeChromaClient:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, name, metadata=None):
        self.collections.setdefault(name, FakeCollection(name))
        return self.collections[name]


def test_embedding_defaults_match_mvp_choices():
    assert DEFAULT_EMBEDDING_MODEL == "ai-forever/ru-en-RoSBERTa"
    assert EMBEDDING_MODEL_CHOICES == (
        "ai-forever/ru-en-RoSBERTa",
        "BAAI/bge-m3",
        "intfloat/multilingual-e5-small",
    )


def test_model_slug_and_collection_name_are_deterministic():
    assert model_slug("ai-forever/ru-en-RoSBERTa") == "ai_forever_ru_en_rosberta"
    assert model_slug("BAAI/bge-m3") == "baai_bge_m3"
    assert collection_name(entity_id=-100123, embedding_model="BAAI/bge-m3") == "telegram_-100123_baai_bge_m3"


def test_fake_embedding_provider_is_deterministic_and_extensible():
    provider: EmbeddingProvider = FakeEmbeddingProvider(model_name="fake-model", dimensions=4)

    first = provider.embed_documents(["hello", "world"])
    second = provider.embed_query("hello")

    assert provider.model_name == "fake-model"
    assert len(first) == 2
    assert len(first[0]) == 4
    assert first[0] == second


def test_vector_store_upserts_chunks_into_model_specific_collection():
    client = FakeChromaClient()
    provider = FakeEmbeddingProvider(model_name="ai-forever/ru-en-RoSBERTa", dimensions=3)
    store = ChromaVectorStore(client=client, embedding_provider=provider)

    store.upsert_chunks([chunk("c1", "Привет мир"), chunk("c2", "hello world")], entity_id=100)

    name = collection_name(entity_id=100, embedding_model=provider.model_name)
    collection = client.collections[name]
    assert collection.count() == 2
    assert collection.upserts[0]["ids"] == ["c1", "c2"]
    assert collection.upserts[0]["metadatas"][0]["message_id_start"] == 1
    assert collection.upserts[0]["metadatas"][0]["message_ids"] == "[1, 2]"


def test_vector_store_query_returns_structured_results_with_sources():
    client = FakeChromaClient()
    provider = FakeEmbeddingProvider(model_name="model-a", dimensions=3)
    store = ChromaVectorStore(client=client, embedding_provider=provider)
    store.upsert_chunks([chunk("c1", "context one"), chunk("c2", "context two")], entity_id=100)

    results = store.query(entity_id=100, query="question", top_k=1)

    assert results == [
        RetrievalResult(
            chunk_id="c1",
            text="context one",
            metadata=client.collections[collection_name(entity_id=100, embedding_model="model-a")].items["c1"]["metadata"],
            distance=0.1,
        )
    ]


def test_model_change_uses_separate_collection_for_reindexing():
    client = FakeChromaClient()
    store_a = ChromaVectorStore(client=client, embedding_provider=FakeEmbeddingProvider(model_name="model-a", dimensions=3))
    store_b = ChromaVectorStore(client=client, embedding_provider=FakeEmbeddingProvider(model_name="model-b", dimensions=3))

    store_a.upsert_chunks([chunk("c1")], entity_id=100)
    store_b.upsert_chunks([chunk("c1")], entity_id=100)

    assert set(client.collections) == {
        collection_name(entity_id=100, embedding_model="model-a"),
        collection_name(entity_id=100, embedding_model="model-b"),
    }


def test_persistent_chroma_client_disables_anonymized_telemetry(tmp_path, monkeypatch):
    captured = {}

    class FakeSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_chromadb = types.SimpleNamespace(
        PersistentClient=lambda **kwargs: captured.setdefault("client_kwargs", kwargs) or object()
    )
    fake_config = types.SimpleNamespace(Settings=FakeSettings)
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)
    monkeypatch.setitem(sys.modules, "chromadb.config", fake_config)

    ChromaVectorStore(
        embedding_provider=FakeEmbeddingProvider(model_name="model-a", dimensions=3),
        persist_directory=tmp_path,
    )

    settings = captured["client_kwargs"]["settings"]
    assert captured["client_kwargs"]["path"] == str(tmp_path)
    assert settings.kwargs["anonymized_telemetry"] is False
