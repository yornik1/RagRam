"""Chroma vector-store integration for RagRam."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .embeddings import EmbeddingProvider, model_slug
from .models import ChunkRecord


def collection_name(*, entity_id: int, embedding_model: str) -> str:
    """Return deterministic Chroma collection name for entity/model pair."""

    return f"telegram_{entity_id}_{model_slug(embedding_model)}"


@dataclass(frozen=True)
class RetrievalResult:
    """Structured retrieval result returned by vector search."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None = None


class ChromaVectorStore:
    """Thin wrapper around Chroma collections with explicit embeddings."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        client: Any | None = None,
        persist_directory: Path | str | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.client = client or self._create_persistent_client(persist_directory)

    def _create_persistent_client(self, persist_directory: Path | str | None) -> Any:
        if persist_directory is None:
            raise ValueError("persist_directory is required when client is not supplied")
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:  # pragma: no cover - exercised only without optional extra.
            raise RuntimeError(
                "chromadb is required for persistent vector storage. "
                "Install RagRam with: pip install -e '.[local]'"
            ) from exc
        return chromadb.PersistentClient(path=str(persist_directory), settings=Settings(anonymized_telemetry=False))

    def collection(self, *, entity_id: int):
        return self.client.get_or_create_collection(
            name=collection_name(entity_id=entity_id, embedding_model=self.embedding_provider.model_name),
            metadata={"embedding_model": self.embedding_provider.model_name},
        )

    def upsert_chunks(self, chunks: list[ChunkRecord], *, entity_id: int) -> int:
        if not chunks:
            return 0
        embeddings = self.embedding_provider.embed_documents([chunk.text for chunk in chunks])
        collection = self.collection(entity_id=entity_id)
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[self._metadata(chunk) for chunk in chunks],
        )
        return len(chunks)

    def query(self, *, entity_id: int, query: str, top_k: int = 8) -> list[RetrievalResult]:
        embedding = self.embedding_provider.embed_query(query)
        collection = self.collection(entity_id=entity_id)
        raw = collection.query(query_embeddings=[embedding], n_results=top_k)
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0] if raw.get("distances") else [None] * len(ids)
        results: list[RetrievalResult] = []
        for index, item_id in enumerate(ids):
            raw_distance = distances[index]
            distance = None if raw_distance is None else float(raw_distance)
            results.append(
                RetrievalResult(
                    chunk_id=str(item_id),
                    text=str(documents[index]),
                    metadata=dict(metadatas[index] or {}),
                    distance=distance,
                )
            )
        return results


    def delete_chunks(self, chunk_ids: list[str], *, entity_id: int) -> int:
        """Delete stale chunk vectors when the backing collection supports it."""

        if not chunk_ids:
            return 0
        collection = self.collection(entity_id=entity_id)
        delete = getattr(collection, "delete", None)
        if delete is None:
            raise RuntimeError("Chroma collection does not support delete; stale chunk reconciliation cannot continue")
        delete(ids=chunk_ids)
        return len(chunk_ids)

    def count(self, *, entity_id: int) -> int:
        return int(self.collection(entity_id=entity_id).count())

    def _metadata(self, chunk: ChunkRecord) -> dict[str, Any]:
        metadata = {
            "entity_id": chunk.entity_id,
            "embedding_model": chunk.embedding_model,
            "message_id_start": chunk.message_id_start,
            "message_id_end": chunk.message_id_end,
            "date_start": chunk.date_start.isoformat(),
            "date_end": chunk.date_end.isoformat(),
            "token_count": chunk.token_count,
        }
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[key] = value
            else:
                metadata[key] = json.dumps(value, ensure_ascii=False)
        return metadata
