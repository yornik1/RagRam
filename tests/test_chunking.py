from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ragram.chunking import ChunkingOptions, approximate_token_count, chunk_messages, deterministic_chunk_id
from ragram.models import MessageRecord


def msg(message_id: int, text: str) -> MessageRecord:
    return MessageRecord(
        entity_id=100,
        message_id=message_id,
        date=datetime(2025, 1, 1, 12, 0, tzinfo=UTC) + timedelta(minutes=message_id),
        text=text,
        sender_name="Alice",
    )


def test_approximate_token_count_handles_russian_and_english_text():
    assert approximate_token_count("Hello world, this is RagRam") >= 5
    assert approximate_token_count("Привет мир это тест RagRam") >= 5


def test_short_nearby_messages_are_grouped_with_ranges_and_metadata():
    messages = [
        msg(1, "one two three"),
        msg(2, "four five six"),
        msg(3, "seven eight nine"),
        msg(4, "ten eleven twelve"),
    ]

    chunks = chunk_messages(
        messages,
        entity_id=100,
        embedding_model="intfloat/multilingual-e5-small",
        options=ChunkingOptions(target_min_tokens=8, target_max_tokens=12),
    )

    assert len(chunks) == 2
    assert chunks[0].message_id_start == 1
    assert chunks[0].message_id_end == 3
    assert chunks[0].metadata["message_ids"] == [1, 2, 3]
    assert chunks[0].date_start == messages[0].date
    assert chunks[0].date_end == messages[2].date
    assert "one two three" in chunks[0].text
    assert chunks[1].metadata["message_ids"] == [4]


def test_long_post_is_preserved_as_individual_chunk_when_suitable():
    long_text = " ".join(f"word{i}" for i in range(18))
    messages = [msg(1, "short one"), msg(2, long_text), msg(3, "short two")]

    chunks = chunk_messages(
        messages,
        entity_id=100,
        embedding_model="model-a",
        options=ChunkingOptions(target_min_tokens=5, target_max_tokens=20),
    )

    long_chunks = [chunk for chunk in chunks if chunk.message_id_start == 2]
    assert len(long_chunks) == 1
    assert long_chunks[0].message_id_end == 2
    assert long_text in long_chunks[0].text


def test_deterministic_chunk_ids_are_stable_and_include_embedding_model():
    messages = [msg(1, "same text"), msg(2, "more text")]

    first = chunk_messages(messages, entity_id=100, embedding_model="model-a", options=ChunkingOptions(target_min_tokens=3, target_max_tokens=20))
    second = chunk_messages(messages, entity_id=100, embedding_model="model-a", options=ChunkingOptions(target_min_tokens=3, target_max_tokens=20))
    other_model = chunk_messages(messages, entity_id=100, embedding_model="model-b", options=ChunkingOptions(target_min_tokens=3, target_max_tokens=20))

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.chunk_id for chunk in first] != [chunk.chunk_id for chunk in other_model]
    assert first[0].chunk_id == deterministic_chunk_id(
        entity_id=100,
        embedding_model="model-a",
        message_id_start=1,
        message_id_end=2,
        text=first[0].text,
    )


def test_default_options_match_mvp_target_window():
    options = ChunkingOptions()

    assert options.target_min_tokens == 400
    assert options.target_max_tokens == 900
