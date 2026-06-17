"""Grounded retrieval-augmented answering for RagRam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .llm import LLMProvider
from .vector_store import RetrievalResult

INSUFFICIENT_CONTEXT_RU = "Недостаточно контекста в найденных сообщениях, чтобы ответить на вопрос."
INSUFFICIENT_CONTEXT_EN = "The retrieved messages do not contain enough context to answer the question."
DEFAULT_MAX_CONTEXT_TOKENS = 6000
MODEL_CONTEXT_BUDGETS = {"qwen3:4b": 6000, "qwen3:8b": 10000}


def context_budget_for_model(model: str) -> int:
    """Return a conservative local prompt context budget for the answer model."""

    return MODEL_CONTEXT_BUDGETS.get(model, MODEL_CONTEXT_BUDGETS["qwen3:4b"])


class Retriever(Protocol):
    """Protocol implemented by vector stores used for retrieval."""

    def query(self, *, entity_id: int, query: str, top_k: int) -> list[RetrievalResult]:
        """Return the top matching chunks for a channel/entity."""
        ...


@dataclass(frozen=True)
class GroundedAnswer:
    """Answer plus source metadata shown in the UI."""

    answer: str
    sources: list[dict]
    raw_context: str = ""


def format_sources(contexts: list[RetrievalResult]) -> list[dict]:
    """Convert retrieval results into UI-friendly source dictionaries."""

    sources: list[dict] = []
    for result in contexts:
        metadata = result.metadata
        sources.append(
            {
                "chunk_id": result.chunk_id,
                "message_id_start": metadata.get("message_id_start"),
                "message_id_end": metadata.get("message_id_end"),
                "date_start": metadata.get("date_start"),
                "date_end": metadata.get("date_end"),
                "distance": result.distance,
            }
        )
    return sources


def _approx_words(text: str) -> int:
    return len(text.split())


def build_context_block(contexts: list[RetrievalResult], *, max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS) -> str:
    """Render retrieved chunks into a compact source-labelled context block."""

    blocks: list[str] = []
    remaining = max_context_tokens
    truncated = False
    for result in contexts:
        if remaining <= 0:
            truncated = True
            break
        metadata = result.metadata
        source_header = (
            f"Source {result.chunk_id} "
            f"(messages {metadata.get('message_id_start')}–{metadata.get('message_id_end')}, "
            f"dates {metadata.get('date_start')}–{metadata.get('date_end')})"
        )
        words = result.text.split()
        if len(words) > remaining:
            text = " ".join(words[:remaining])
            truncated = True
        else:
            text = result.text
        blocks.append(f"{source_header}\n{text}")
        remaining -= _approx_words(text)
    if truncated:
        blocks.append("[context truncated to fit local model budget]")
    return "\n\n".join(blocks)


def build_grounded_prompt(*, question: str, contexts: list[RetrievalResult], top_k: int = 8, max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS) -> str:
    """Build a conservative prompt for source-grounded local answering."""

    context_block = build_context_block(contexts[:top_k], max_context_tokens=max_context_tokens)
    return f"""You are RagRam, a local Telegram channel question-answering assistant.
Use the rules below strictly:
- answer only from the retrieved context.
- If the context is insufficient, say that the retrieved messages do not contain enough information.
- Answer in the language of the question.
- Preserve uncertainty and do not invent facts, dates, names, or numbers.
- Cite source ids naturally when useful.

Question:
{question}

Retrieved context:
{context_block}

Answer:"""


def _looks_english(text: str) -> bool:
    latin = sum("a" <= char.lower() <= "z" for char in text)
    cyrillic = sum("а" <= char.lower() <= "я" or char.lower() == "ё" for char in text)
    return latin > cyrillic


class RagService:
    """Orchestrates retrieval, grounded prompt construction, and local generation."""

    def __init__(self, *, retriever: Retriever, llm: LLMProvider, entity_id: int, answer_model: str):
        self.retriever = retriever
        self.llm = llm
        self.entity_id = entity_id
        self.answer_model = answer_model

    def answer(self, question: str, *, top_k: int = 8) -> GroundedAnswer:
        """Answer a user question from retrieved Telegram chunks."""

        contexts = self.retriever.query(entity_id=self.entity_id, query=question, top_k=top_k)
        if not contexts:
            message = INSUFFICIENT_CONTEXT_EN if _looks_english(question) else INSUFFICIENT_CONTEXT_RU
            return GroundedAnswer(answer=message, sources=[])

        prompt = build_grounded_prompt(
            question=question,
            contexts=contexts,
            top_k=top_k,
            max_context_tokens=context_budget_for_model(self.answer_model),
        )
        answer = self.llm.generate(model=self.answer_model, prompt=prompt)
        return GroundedAnswer(
            answer=answer,
            sources=format_sources(contexts),
            raw_context=build_context_block(contexts, max_context_tokens=context_budget_for_model(self.answer_model)),
        )
