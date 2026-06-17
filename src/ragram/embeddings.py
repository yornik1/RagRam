"""Embedding provider interfaces for RagRam."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

DEFAULT_EMBEDDING_MODEL = "ai-forever/ru-en-RoSBERTa"
EMBEDDING_MODEL_CHOICES = (
    DEFAULT_EMBEDDING_MODEL,
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-small",
)


class EmbeddingProvider(Protocol):
    """Minimal embedding provider interface used by vector indexing."""

    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def model_slug(model_name: str) -> str:
    """Return a Chroma-safe deterministic slug for an embedding model name."""

    slug = re.sub(r"[^0-9A-Za-z]+", "_", model_name).strip("_").lower()
    return slug or "model"


class SentenceTransformerEmbeddingProvider:
    """sentence-transformers-backed embedding provider loaded lazily."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - exercised only without optional extra.
                raise RuntimeError(
                    "sentence-transformers is required for local embeddings. "
                    "Install RagRam with: pip install -e '.[local]'"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return [list(map(float, item)) for item in embeddings]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class FakeEmbeddingProvider:
    """Deterministic offline embedding provider for tests."""

    def __init__(self, model_name: str = "fake-embedding", dimensions: int = 8):
        self.model_name = model_name
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(f"{self.model_name}\n{text}".encode("utf-8")).digest()
        values = []
        for index in range(self.dimensions):
            byte = digest[index % len(digest)]
            values.append(round(byte / 255.0, 6))
        return values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
