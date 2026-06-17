from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from ragram.app import ingest_text_messages
from ragram.models import IndexScope
from ragram.progress import InMemoryProgressReporter, RateSnapshot
from ragram.storage import SQLiteStore
from ragram.telegram_client import TelegramFloodWait, iter_text_messages


class FakeAsyncIter:
    def __init__(self, items):
        self.items = list(items)

    def __aiter__(self):
        self._iter = iter(self.items)
        return self

    async def __anext__(self):
        try:
            item = next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClient:
    def __init__(self, messages):
        self.messages = messages
        self.iter_calls = []

    def iter_messages(self, entity, **kwargs):
        self.iter_calls.append((entity, kwargs))
        return FakeAsyncIter(self.messages)


@dataclass
class FakeMessage:
    id: int
    date: datetime
    text: str | None = None
    message: str | None = None
    sender_id: int | None = None
    reply_to_msg_id: int | None = None


def run(coro):
    return asyncio.run(coro)


def require(value):
    assert value is not None
    return value


def test_index_scope_factories_and_start_dates():
    assert IndexScope.last_n(1000).scope_type == "last_n"
    assert IndexScope.last_n(1000).limit == 1000
    assert IndexScope.all().limit is None
    assert IndexScope.from_year(2024).start_date == datetime(2024, 1, 1, tzinfo=UTC)
    assert IndexScope.from_date("2024-05-06").start_date == datetime(2024, 5, 6, tzinfo=UTC)


def test_iter_text_messages_uses_limit_and_skips_empty_text():
    client = FakeClient(
        [
            FakeMessage(3, datetime(2025, 1, 3, tzinfo=UTC), text="new"),
            FakeMessage(2, datetime(2025, 1, 2, tzinfo=UTC), text=""),
            FakeMessage(1, datetime(2025, 1, 1, tzinfo=UTC), message="old"),
        ]
    )

    result = run(iter_text_messages(client, "entity", entity_id=100, scope=IndexScope.last_n(2)))

    assert [message.message_id for message in result] == [3, 1]
    assert result[0].text == "new"
    assert client.iter_calls == [("entity", {"limit": 2})]


def test_iter_text_messages_from_date_stops_after_older_messages():
    client = FakeClient(
        [
            FakeMessage(3, datetime(2025, 1, 3, tzinfo=UTC), text="keep"),
            FakeMessage(2, datetime(2024, 1, 2, tzinfo=UTC), text="also keep"),
            FakeMessage(1, datetime(2023, 12, 31, tzinfo=UTC), text="too old"),
        ]
    )

    result = run(iter_text_messages(client, "entity", entity_id=100, scope=IndexScope.from_year(2024)))

    assert [message.message_id for message in result] == [3, 2]


def test_ingest_text_messages_persists_raw_messages_and_completes_run(tmp_path):
    store = SQLiteStore(tmp_path / "ragram.sqlite")
    store.initialize()
    progress = InMemoryProgressReporter()
    client = FakeClient(
        [
            FakeMessage(2, datetime(2025, 1, 2, tzinfo=UTC), text="Второе", sender_id=10),
            FakeMessage(1, datetime(2025, 1, 1, tzinfo=UTC), text="Первое", reply_to_msg_id=0),
        ]
    )

    result = run(
        ingest_text_messages(
            client,
            store,
            entity="entity",
            entity_id=100,
            channel_username="example",
            scope=IndexScope.last_n(1000),
            run_id="run-1",
            embedding_model="model-a",
            progress=progress,
        )
    )

    assert result.saved_messages == 2
    assert store.message_count(100) == 2
    assert require(store.get_message(100, 2)).message_link == "https://t.me/example/2"
    assert require(store.get_index_run("run-1")).status == "complete"
    assert require(store.get_index_run("run-1")).last_message_id == 1
    assert progress.events[-1] == ("finish", 2)


def test_ingest_text_messages_is_resume_safe_via_dedupe(tmp_path):
    store = SQLiteStore(tmp_path / "ragram.sqlite")
    store.initialize()
    client = FakeClient([FakeMessage(1, datetime(2025, 1, 1, tzinfo=UTC), text="same")])

    first = run(ingest_text_messages(client, store, entity="entity", entity_id=100, scope=IndexScope.last_n(10), run_id="r1", embedding_model="m"))
    second = run(ingest_text_messages(client, store, entity="entity", entity_id=100, scope=IndexScope.last_n(10), run_id="r2", embedding_model="m"))

    assert first.saved_messages == 1
    assert second.saved_messages == 0
    assert store.message_count(100) == 1


def test_ingest_text_messages_marks_run_failed_on_flood_wait(tmp_path):
    store = SQLiteStore(tmp_path / "ragram.sqlite")
    store.initialize()
    client = FakeClient([TelegramFloodWait(seconds=17, operation="iter_messages")])

    with pytest.raises(TelegramFloodWait):
        run(ingest_text_messages(client, store, entity="entity", entity_id=100, scope=IndexScope.all(), run_id="run-fail", embedding_model="m"))

    run_record = store.get_index_run("run-fail")
    assert require(run_record).status == "failed"
    assert require(run_record).error == "Telegram FloodWait during iter_messages: wait 17 seconds"


def test_rate_snapshot_reports_rates_and_eta():
    snapshot = RateSnapshot(completed=50, total=100, elapsed_seconds=10)

    assert snapshot.items_per_second == 5
    assert snapshot.remaining == 50
    assert snapshot.eta_seconds == 10


def test_ingest_text_messages_persists_before_later_flood_wait(tmp_path):
    store = SQLiteStore(tmp_path / "ragram.sqlite")
    store.initialize()
    client = FakeClient([
        FakeMessage(2, datetime(2025, 1, 2, tzinfo=UTC), text="saved before flood"),
        TelegramFloodWait(seconds=19, operation="iter_messages"),
    ])

    with pytest.raises(TelegramFloodWait):
        run(ingest_text_messages(client, store, entity="entity", entity_id=100, scope=IndexScope.all(), run_id="run-mid-fail", embedding_model="m"))

    assert store.message_count(100) == 1
    assert require(store.get_message(100, 2)).text == "saved before flood"
    run_record = store.get_index_run("run-mid-fail")
    assert require(run_record).status == "failed"
    assert require(run_record).last_message_id == 2


def test_ingest_text_messages_resumes_failed_scope_from_cursor(tmp_path):
    store = SQLiteStore(tmp_path / "ragram.sqlite")
    store.initialize()
    first_client = FakeClient([
        FakeMessage(5, datetime(2025, 1, 5, tzinfo=UTC), text="five"),
        FakeMessage(4, datetime(2025, 1, 4, tzinfo=UTC), text="four"),
        TelegramFloodWait(seconds=7, operation="iter_messages"),
    ])
    with pytest.raises(TelegramFloodWait):
        run(ingest_text_messages(first_client, store, entity="entity", entity_id=100, scope=IndexScope.all(), run_id="run-fail", embedding_model="m"))

    second_client = FakeClient([
        FakeMessage(5, datetime(2025, 1, 5, tzinfo=UTC), text="five again"),
        FakeMessage(4, datetime(2025, 1, 4, tzinfo=UTC), text="four again"),
        FakeMessage(3, datetime(2025, 1, 3, tzinfo=UTC), text="three"),
    ])
    result = run(ingest_text_messages(second_client, store, entity="entity", entity_id=100, scope=IndexScope.all(), run_id="run-resume", embedding_model="m"))

    assert result.fetched_messages == 1
    assert result.saved_messages == 1
    assert require(store.get_message(100, 5)).text == "five"
    assert require(store.get_message(100, 3)).text == "three"


def test_ingest_text_messages_resumes_last_n_scope_from_cursor(tmp_path):
    store = SQLiteStore(tmp_path / "ragram.sqlite")
    store.initialize()
    first_client = FakeClient([
        FakeMessage(5, datetime(2025, 1, 5, tzinfo=UTC), text="five"),
        TelegramFloodWait(seconds=7, operation="iter_messages"),
    ])
    with pytest.raises(TelegramFloodWait):
        run(ingest_text_messages(first_client, store, entity="entity", entity_id=100, scope=IndexScope.last_n(1000), run_id="run-lastn-fail", embedding_model="m"))

    second_client = FakeClient([
        FakeMessage(5, datetime(2025, 1, 5, tzinfo=UTC), text="five again"),
        FakeMessage(4, datetime(2025, 1, 4, tzinfo=UTC), text="four"),
    ])
    result = run(ingest_text_messages(second_client, store, entity="entity", entity_id=100, scope=IndexScope.last_n(1000), run_id="run-lastn-resume", embedding_model="m"))

    assert second_client.iter_calls == [("entity", {"limit": 1000, "max_id": 5})]
    assert result.fetched_messages == 1
    assert require(store.get_message(100, 5)).text == "five"
    assert require(store.get_message(100, 4)).text == "four"
