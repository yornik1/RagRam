"""Message chunking for RagRam's Telegram RAG index."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .models import ChunkRecord, MessageRecord

_TOKEN_RE = re.compile(r"[\wА-Яа-яЁё]+|[^\s]", re.UNICODE)


@dataclass(frozen=True)
class ChunkingOptions:
    """Configurable token window for grouping Telegram messages."""

    target_min_tokens: int = 400
    target_max_tokens: int = 900

    def __post_init__(self) -> None:
        if self.target_min_tokens <= 0:
            raise ValueError("target_min_tokens must be positive")
        if self.target_max_tokens < self.target_min_tokens:
            raise ValueError("target_max_tokens must be >= target_min_tokens")


def approximate_token_count(text: str) -> int:
    """Return a deterministic lightweight token estimate for Russian/English text."""

    return len(_TOKEN_RE.findall(text))


def deterministic_chunk_id(
    *,
    entity_id: int,
    embedding_model: str,
    message_id_start: int,
    message_id_end: int,
    text: str,
) -> str:
    """Create a stable chunk ID from entity/model/message range/content."""

    digest = hashlib.sha256(
        "\n".join(
            [
                str(entity_id),
                embedding_model,
                str(message_id_start),
                str(message_id_end),
                text,
            ]
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"tg_{entity_id}_{message_id_start}_{message_id_end}_{digest}"


def _format_message(message: MessageRecord) -> str:
    sender = f" @{message.sender_name}" if message.sender_name else ""
    return f"[{message.message_id} {message.date.isoformat()}{sender}] {message.text.strip()}"


def _build_chunk(
    messages: list[MessageRecord],
    *,
    entity_id: int,
    embedding_model: str,
) -> ChunkRecord:
    if not messages:
        raise ValueError("cannot build chunk from no messages")
    text = "\n".join(_format_message(message) for message in messages)
    message_ids = [message.message_id for message in messages]
    dates = [message.date for message in messages]
    return ChunkRecord(
        chunk_id=deterministic_chunk_id(
            entity_id=entity_id,
            embedding_model=embedding_model,
            message_id_start=min(message_ids),
            message_id_end=max(message_ids),
            text=text,
        ),
        entity_id=entity_id,
        embedding_model=embedding_model,
        message_id_start=min(message_ids),
        message_id_end=max(message_ids),
        date_start=min(dates),
        date_end=max(dates),
        text=text,
        token_count=approximate_token_count(text),
        metadata={
            "message_ids": message_ids,
            "message_count": len(messages),
            "date_start": min(dates).isoformat(),
            "date_end": max(dates).isoformat(),
        },
    )


def chunk_messages(
    messages: list[MessageRecord],
    *,
    entity_id: int,
    embedding_model: str,
    options: ChunkingOptions | None = None,
) -> list[ChunkRecord]:
    """Group Telegram messages into source-preserving RAG chunks.

    Short nearby messages are accumulated until the target minimum is reached or
    adding the next message would exceed the target maximum. Messages whose own
    content fits the target window are emitted as individual chunks to preserve
    long channel posts.
    """

    opts = options or ChunkingOptions()
    sorted_messages = sorted(
        (message for message in messages if message.text.strip()),
        key=lambda item: (item.date, item.message_id),
    )
    chunks: list[ChunkRecord] = []
    current: list[MessageRecord] = []
    current_tokens = 0

    def flush_current() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append(_build_chunk(current, entity_id=entity_id, embedding_model=embedding_model))
            current = []
            current_tokens = 0

    for message in sorted_messages:
        message_tokens = approximate_token_count(message.text)
        if message_tokens >= opts.target_min_tokens:
            flush_current()
            chunks.append(_build_chunk([message], entity_id=entity_id, embedding_model=embedding_model))
            continue

        if current and current_tokens + message_tokens > opts.target_max_tokens:
            flush_current()

        current.append(message)
        current_tokens += message_tokens

        if current_tokens >= opts.target_min_tokens:
            flush_current()

    flush_current()
    return chunks
