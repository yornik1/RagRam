from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from ragram.telegram_client import (
    TelegramLoginCodeInvalid,
    TelegramLoginCodeRequestFailed,
    TelegramLoginCodeResendRequested,
    TelegramQRLoginTimeout,
    TelegramDialog,
    TelegramFloodWait,
    describe_login_code_delivery,
    ensure_telegram_login,
    ensure_telegram_qr_login,
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


class FakeSentCodeTypeApp:
    def __init__(self, length: int):
        self.length = length


class FakeClient:
    def __init__(
        self,
        *,
        authorized=False,
        dialogs=None,
        entity=None,
        password_needed=False,
        flood_wait=None,
        invalid_code_attempts=0,
        send_code_errors=None,
    ):
        self.authorized = authorized
        self.dialogs = dialogs or []
        self.entity = entity
        self.password_needed = password_needed
        self.flood_wait = flood_wait
        self.invalid_code_attempts = invalid_code_attempts
        self.send_code_errors = list(send_code_errors or [])
        self.connected = False
        self.sent_code_to = None
        self.send_code_calls = []
        self.sign_in_calls = []
        self.get_entity_calls = []

    async def connect(self):
        self.connected = True

    async def is_user_authorized(self):
        return self.authorized

    async def send_code_request(self, phone):
        self.sent_code_to = phone
        self.send_code_calls.append(phone)
        if self.send_code_errors:
            error = self.send_code_errors.pop(0)
            if error is not None:
                raise error
        return type(
            "SentCode",
            (),
            {"phone_code_hash": "hash-123", "type": FakeSentCodeTypeApp(5), "next_type": None, "timeout": None},
        )()

    async def sign_in(self, **kwargs):
        self.sign_in_calls.append(kwargs)
        if self.invalid_code_attempts > 0 and "code" in kwargs:
            self.invalid_code_attempts -= 1
            raise FakePhoneCodeInvalidError()
        if self.password_needed and "password" not in kwargs:
            raise FakeSessionPasswordNeededError()
        self.authorized = True
        return type("User", (), {"id": 42})()

    async def qr_login(self, ignored_ids=None):
        return FakeQRLogin()

    def iter_dialogs(self, **kwargs):
        if self.flood_wait is not None:
            raise self.flood_wait
        return FakeAsyncIter(self.dialogs)

    async def get_entity(self, value):
        self.get_entity_calls.append(value)
        return self.entity


class FakeQRLogin:
    def __init__(self, *, fail_timeout: bool = False, expired_once: bool = False):
        self.url = "tg://login?token=fake"
        self.expires = datetime.now(UTC) + timedelta(minutes=5)
        self.wait_calls = []
        self.fail_timeout = fail_timeout
        self.expired_once = expired_once
        self.recreate_calls = 0
        if expired_once:
            self.url = "tg://login?token=expired"
            self.expires = datetime.now(UTC) - timedelta(seconds=1)

    async def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.fail_timeout or self.expired_once:
            self.expired_once = False
            raise TimeoutError()
        return type("User", (), {"id": 42})()

    async def recreate(self):
        self.recreate_calls += 1
        self.expired_once = False
        self.url = "tg://login?token=fresh"
        self.expires = datetime.now(UTC) + timedelta(minutes=5)


class FakeQRClient(FakeClient):
    def __init__(self, *, authorized=False, qr_login=None):
        super().__init__(authorized=authorized)
        self.qr_login_obj = qr_login or FakeQRLogin()
        self.qr_login_calls = 0

    async def qr_login(self, ignored_ids=None):
        self.qr_login_calls += 1
        return self.qr_login_obj


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
        invalid_attempts = []

        result = await ensure_telegram_login(
            client,
            phone="+15550000000",
            code_callback=lambda: next(codes),
            password_callback=lambda: "secret",
            code_invalid_callback=lambda attempt, attempts: invalid_attempts.append((attempt, attempts)),
            code_invalid_error_types=(FakePhoneCodeInvalidError,),
        )

        assert result.reused_session is False
        assert [call["code"] for call in client.sign_in_calls] == ["11111", "22222"]
        assert invalid_attempts == [(1, 3)]

    asyncio.run(scenario())


def test_login_can_resend_code_before_sign_in():
    async def scenario():
        client = FakeClient()
        values = iter([TelegramLoginCodeResendRequested(), "22222"])
        deliveries = []

        def code_callback():
            value = next(values)
            if isinstance(value, BaseException):
                raise value
            return value

        result = await ensure_telegram_login(
            client,
            phone="+15550000000",
            code_callback=code_callback,
            password_callback=lambda: "secret",
            code_sent_callback=deliveries.append,
            code_invalid_error_types=(FakePhoneCodeInvalidError,),
        )

        assert result.reused_session is False
        assert client.send_code_calls == ["+15550000000", "+15550000000"]
        assert [call["code"] for call in client.sign_in_calls] == ["22222"]
        assert [delivery.operation for delivery in deliveries] == ["send_login_code", "resend_login_code"]
        assert deliveries[0].delivery_method == "FakeSentCodeTypeApp"

    asyncio.run(scenario())


def test_login_resend_failure_is_domain_error_without_raw_resend_exception():
    async def scenario():
        client = FakeClient(
            send_code_errors=[
                None,
                RuntimeError(
                    "Returned when all available options for this type of number were already used "
                    "(caused by ResendCodeRequest)"
                ),
            ]
        )

        with pytest.raises(TelegramLoginCodeRequestFailed) as raised:
            await ensure_telegram_login(
                client,
                phone="+15550000000",
                code_callback=lambda: (_ for _ in ()).throw(TelegramLoginCodeResendRequested()),
                password_callback=lambda: "secret",
                code_invalid_error_types=(FakePhoneCodeInvalidError,),
            )

        assert "refused to send another code right now" in str(raised.value)
        assert "all available options" not in str(raised.value).casefold()
        assert client.send_code_calls == ["+15550000000", "+15550000000"]

    asyncio.run(scenario())


def test_describe_login_code_delivery_maps_safe_telegram_metadata():
    sent_code_type_app = type("SentCodeTypeApp", (), {"length": 5})
    code_type_sms = type("CodeTypeSms", (), {})
    sent_code = type(
        "SentCode",
        (),
        {
            "type": sent_code_type_app(),
            "next_type": code_type_sms(),
            "timeout": 60,
        },
    )()

    delivery = describe_login_code_delivery(sent_code, operation="send_login_code")

    assert delivery.delivery_method == "Telegram app/service message"
    assert delivery.next_method == "SMS"
    assert delivery.timeout_seconds == 60
    assert delivery.code_length == 5


def test_login_send_code_flood_wait_is_domain_flood_wait():
    async def scenario():
        client = FakeClient(send_code_errors=[FakeFloodWaitError(42)])

        with pytest.raises(TelegramFloodWait) as raised:
            await ensure_telegram_login(
                client,
                phone="+15550000000",
                code_callback=lambda: "11111",
                password_callback=lambda: "secret",
                code_invalid_error_types=(FakePhoneCodeInvalidError,),
            )

        assert raised.value.seconds == 42
        assert raised.value.operation == "send_login_code"

    asyncio.run(scenario())


def test_qr_login_displays_challenge_and_waits_for_scan():
    async def scenario():
        qr = FakeQRLogin()
        client = FakeQRClient(qr_login=qr)
        challenges = []

        result = await ensure_telegram_qr_login(
            client,
            qr_callback=challenges.append,
            password_callback=lambda: "2fa",
            timeout_seconds=123,
        )

        assert result.reused_session is False
        assert client.connected is True
        assert client.qr_login_calls == 1
        assert challenges[0].url == "tg://login?token=fake"
        assert qr.wait_calls[0] == pytest.approx(123, abs=0.1)

    asyncio.run(scenario())


def test_qr_login_refreshes_expired_qr_before_total_timeout():
    async def scenario():
        qr = FakeQRLogin(expired_once=True)
        client = FakeQRClient(qr_login=qr)
        challenges = []

        result = await ensure_telegram_qr_login(
            client,
            qr_callback=challenges.append,
            password_callback=lambda: "2fa",
            timeout_seconds=30,
        )

        assert result.reused_session is False
        assert qr.recreate_calls == 1
        assert [challenge.url for challenge in challenges] == [
            "tg://login?token=expired",
            "tg://login?token=fresh",
        ]
        assert len(qr.wait_calls) == 1

    asyncio.run(scenario())


def test_qr_login_timeout_is_domain_error():
    async def scenario():
        client = FakeQRClient(qr_login=FakeQRLogin(fail_timeout=True))

        with pytest.raises(TelegramQRLoginTimeout):
            await ensure_telegram_qr_login(
                client,
                qr_callback=lambda challenge: None,
                password_callback=lambda: "2fa",
                timeout_seconds=1,
            )

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
