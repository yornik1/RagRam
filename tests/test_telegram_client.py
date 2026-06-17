from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from ragram.telegram_client import (
    TelegramLoginCodeInvalid,
    TelegramDialog,
    TelegramFloodWait,
    ensure_telegram_login,
    filter_dialogs,
    list_accessible_dialogs,
    normalize_entity_input,
    resolve_dialog_choice,
)


class FakeSessionPasswordNeededError(Exception):
    pass


class FakePhoneCodeInvalidError(Exception):
    pass


class FakeFloodWaitError(Exception):
    def __init__(self, seconds: int):
        super().__init__(f"wait {seconds}")
        self.seconds = seconds


class FakeClient:
    def __init__(self, *, authorized=False, dialogs=None, entity=None, password_needed=False, flood_wait=None, invalid_code_attempts=0):
        self.authorized = authorized
        self.dialogs = dialogs or []
        self.entity = entity
        self.password_needed = password_needed
        self.flood_wait = flood_wait
        self.invalid_code_attempts = invalid_code_attempts
        self.connected = False
        self.sent_code_to = None
        self.sign_in_calls = []
        self.get_entity_calls = []

    async def connect(self):
        self.connected = True

    async def is_user_authorized(self):
        return self.authorized

    async def send_code_request(self, phone):
        self.sent_code_to = phone
        return type("SentCode", (), {"phone_code_hash": "hash-123"})()

    async def sign_in(self, **kwargs):
        self.sign_in_calls.append(kwargs)
        if self.invalid_code_attempts > 0 and "code" in kwargs:
            self.invalid_code_attempts -= 1
            raise FakePhoneCodeInvalidError()
        if self.password_needed and "password" not in kwargs:
            raise FakeSessionPasswordNeededError()
        self.authorized = True
        return type("User", (), {"id": 42})()

    def iter_dialogs(self, **kwargs):
        if self.flood_wait is not None:
            raise self.flood_wait
        return FakeAsyncIter(self.dialogs)

    async def get_entity(self, value):
        self.get_entity_calls.append(value)
        return self.entity


class FakeAsyncIter:
    def __init__(self, items):
        self.items = list(items)

    def __aiter__(self):
        self._iter = iter(self.items)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


@dataclass
class FakeEntity:
    id: int
    title: str
    username: str | None = None
    broadcast: bool = False
    megagroup: bool = False
    is_group: bool = False


@dataclass
class FakeMessage:
    date: datetime | None


@dataclass
class FakeDialog:
    entity: FakeEntity
    title: str
    date: datetime | None
    unread_count: int = 0
    is_channel: bool = False
    is_group: bool = False

    @property
    def message(self):
        return FakeMessage(self.date)


def test_login_reuses_authorized_session_without_sending_code():
    async def scenario():
        client = FakeClient(authorized=True)

        result = await ensure_telegram_login(
        client,
        phone="+15550000000",
        code_callback=lambda: "11111",
        password_callback=lambda: "secret",
    )

        assert result.reused_session is True
        assert client.connected is True
        assert client.sent_code_to is None
        assert client.sign_in_calls == []

    asyncio.run(scenario())


def test_login_uses_code_hash_and_2fa_password_when_required():
    async def scenario():
        client = FakeClient(password_needed=True)

        result = await ensure_telegram_login(
        client,
        phone="+15550000000",
        code_callback=lambda: "22222",
        password_callback=lambda: "2fa-secret",
        password_needed_error_types=(FakeSessionPasswordNeededError,),
    )

        assert result.reused_session is False
        assert result.required_2fa is True
        assert client.sent_code_to == "+15550000000"
        assert client.sign_in_calls[0] == {
        "phone": "+15550000000",
        "code": "22222",
        "phone_code_hash": "hash-123",
    }
        assert client.sign_in_calls[1] == {"password": "2fa-secret"}

    asyncio.run(scenario())


def test_login_retries_invalid_code_before_success():
    async def scenario():
        client = FakeClient(invalid_code_attempts=1)
        codes = iter(["11111", "22222"])

        result = await ensure_telegram_login(
            client,
            phone="+15550000000",
            code_callback=lambda: next(codes),
            password_callback=lambda: "secret",
            code_invalid_error_types=(FakePhoneCodeInvalidError,),
        )

        assert result.reused_session is False
        assert [call["code"] for call in client.sign_in_calls] == ["11111", "22222"]

    asyncio.run(scenario())


def test_login_raises_domain_error_after_invalid_code_retries():
    async def scenario():
        client = FakeClient(invalid_code_attempts=3)
        codes = iter(["11111", "22222", "33333"])

        with pytest.raises(TelegramLoginCodeInvalid):
            await ensure_telegram_login(
                client,
                phone="+15550000000",
                code_callback=lambda: next(codes),
                password_callback=lambda: "secret",
                code_invalid_error_types=(FakePhoneCodeInvalidError,),
                code_attempts=3,
            )

        assert len(client.sign_in_calls) == 3

    asyncio.run(scenario())


def test_list_accessible_dialogs_filters_channels_groups_and_sorts_by_date():
    async def scenario():
        older = datetime(2024, 1, 1, tzinfo=UTC)
        newer = datetime(2025, 1, 1, tzinfo=UTC)
        dialogs = [
        FakeDialog(FakeEntity(1, "User chat"), "User chat", newer),
        FakeDialog(FakeEntity(2, "Old group", is_group=True), "Old group", older, unread_count=3, is_group=True),
        FakeDialog(FakeEntity(3, "New channel", "new", broadcast=True), "New channel", newer, is_channel=True),
        FakeDialog(FakeEntity(4, "Mega", "mega", megagroup=True), "Mega", None, is_channel=True),
    ]

        result = await list_accessible_dialogs(FakeClient(dialogs=dialogs))

        assert [dialog.title for dialog in result] == ["New channel", "Old group", "Mega"]
        assert result[0].kind == "channel"
        assert result[1].kind == "group"
        assert result[1].unread_count == 3
        assert result[2].last_message_date is None

    asyncio.run(scenario())


def test_filter_dialogs_matches_title_username_and_entity_id():
    dialogs = [
        TelegramDialog(entity_id=10, title="Новости Python", username="py_news", kind="channel"),
        TelegramDialog(entity_id=20, title="Design", username=None, kind="group"),
    ]

    assert [item.entity_id for item in filter_dialogs(dialogs, "python")] == [10]
    assert [item.entity_id for item in filter_dialogs(dialogs, "py_news")] == [10]
    assert [item.entity_id for item in filter_dialogs(dialogs, "20")] == [20]
    assert filter_dialogs(dialogs, "") == dialogs


def test_normalize_entity_input_accepts_username_urls_and_ids():
    assert normalize_entity_input(" @example ") == "example"
    assert normalize_entity_input("https://t.me/example") == "example"
    assert normalize_entity_input("https://telegram.me/example") == "example"
    assert normalize_entity_input("-100123") == -100123


def test_resolve_dialog_choice_prefers_known_title_username_id_then_get_entity():
    async def scenario():
        dialogs = [TelegramDialog(entity_id=10, title="Exact Title", username="known", kind="channel")]

        assert (await resolve_dialog_choice(FakeClient(), "Exact Title", dialogs)).entity_id == 10
        assert (await resolve_dialog_choice(FakeClient(), "@known", dialogs)).entity_id == 10
        assert (await resolve_dialog_choice(FakeClient(), "10", dialogs)).entity_id == 10

        client = FakeClient(entity=FakeEntity(99, "Resolved", "resolved", broadcast=True))
        resolved = await resolve_dialog_choice(client, "https://t.me/resolved", dialogs)

        assert resolved.entity_id == 99
        assert resolved.username == "resolved"
        assert client.get_entity_calls == ["resolved"]

    asyncio.run(scenario())


def test_list_dialogs_converts_flood_wait_to_safe_domain_error():
    async def scenario():
        with pytest.raises(TelegramFloodWait) as raised:
            await list_accessible_dialogs(FakeClient(flood_wait=FakeFloodWaitError(17)))

        assert raised.value.seconds == 17
        assert raised.value.operation == "list_dialogs"

    asyncio.run(scenario())
