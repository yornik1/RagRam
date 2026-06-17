"""Telethon integration boundaries for RagRam.

This module keeps the real Telethon dependency behind small, fakeable async
functions. Tests use fake clients so no Telegram credentials or network access
are needed.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar

try:  # pragma: no cover - import presence is covered by package smoke tests.
    from telethon import TelegramClient as TelethonTelegramClient
    from telethon import errors as telethon_errors
except Exception:  # pragma: no cover - lets status/import work if dependency is absent.
    TelethonTelegramClient = None  # type: ignore[assignment]
    telethon_errors = None  # type: ignore[assignment]

from .config import TelegramConfig
from .models import AppPaths, IndexScope, MessageRecord

T = TypeVar("T")


class TelegramClientProtocol(Protocol):
    async def connect(self) -> Any: ...

    async def is_user_authorized(self) -> bool: ...

    async def send_code_request(self, phone: str) -> Any: ...

    async def sign_in(self, **kwargs: Any) -> Any: ...

    def iter_dialogs(self, **kwargs: Any) -> Any: ...

    async def get_entity(self, value: Any) -> Any: ...


@dataclass(frozen=True)
class LoginResult:
    """Result of ensuring a Telethon user session is authorized."""

    reused_session: bool
    required_2fa: bool = False


@dataclass(frozen=True)
class LoginCodeDelivery:
    """Safe, user-displayable details about a Telegram login-code request."""

    operation: str
    delivery_method: str
    next_method: str | None = None
    timeout_seconds: int | None = None
    code_length: int | None = None


@dataclass(frozen=True)
class TelegramDialog:
    """Displayable accessible Telegram channel/group info."""

    entity_id: int
    title: str
    kind: str
    username: str | None = None
    last_message_date: datetime | None = None
    unread_count: int | None = None
    raw: Any = None


class TelegramFloodWait(RuntimeError):
    """Domain error used when Telegram asks us to wait before retrying."""

    def __init__(self, seconds: int, operation: str):
        self.seconds = seconds
        self.operation = operation
        super().__init__(f"Telegram FloodWait during {operation}: wait {seconds} seconds")


class TelegramConfigurationError(RuntimeError):
    """Raised when Telegram config is incomplete for a real client."""


class DialogNotFoundError(ValueError):
    """Raised when a custom dialog choice cannot be resolved."""


class TelegramLoginCodeInvalid(RuntimeError):
    """Raised when Telegram rejects the login code after retries."""


class TelegramLoginCodeResendRequested(RuntimeError):
    """Raised by the UI callback when the user asks Telegram to resend a code."""


class TelegramLoginCodeRequestFailed(RuntimeError):
    """Raised when Telegram rejects or fails a login-code send/resend request."""

    def __init__(self, operation: str, exc: BaseException):
        self.operation = operation
        self.original_error_type = exc.__class__.__name__
        self.original_message = str(exc)
        if "all available options" in self.original_message.casefold():
            message = (
                "Telegram accepted the earlier login-code request, but it refused to send another code right now: "
                "all delivery options for this phone number are already used. Wait a few minutes, check every logged-in "
                "Telegram app/session for a service login message, or quit with q and re-check the saved phone/api_id/api_hash."
            )
        else:
            message = f"Telegram could not {operation.replace('_', ' ')}: {self.original_message}"
        super().__init__(message)


def create_telegram_client(config: TelegramConfig, paths: AppPaths) -> Any:
    """Create a real Telethon client for the configured local session path."""

    if TelethonTelegramClient is None:
        raise TelegramConfigurationError("Telethon is not installed")
    if config.api_id is None or not config.api_hash:
        raise TelegramConfigurationError("Telegram api_id and api_hash are required")

    session_path = paths.sessions_dir / config.session_name
    return TelethonTelegramClient(str(session_path), int(config.api_id), config.api_hash)


def _default_password_error_types() -> tuple[type[BaseException], ...]:
    if telethon_errors is None:
        return ()
    error_type = getattr(telethon_errors, "SessionPasswordNeededError", None)
    return (error_type,) if isinstance(error_type, type) else ()


def _default_code_invalid_error_types() -> tuple[type[BaseException], ...]:
    if telethon_errors is None:
        return ()
    error_type = getattr(telethon_errors, "PhoneCodeInvalidError", None)
    return (error_type,) if isinstance(error_type, type) else ()


def _sent_code_method_label(value: Any) -> str | None:
    if value is None:
        return None
    name = value.__class__.__name__
    labels = {
        "SentCodeTypeApp": "Telegram app/service message",
        "SentCodeTypeSms": "SMS",
        "SentCodeTypeCall": "phone call",
        "SentCodeTypeFlashCall": "flash call",
        "SentCodeTypeMissedCall": "missed call",
        "SentCodeTypeEmailCode": "email",
        "SentCodeTypeFirebaseSms": "Firebase/SMS",
        "SentCodeTypeFragmentSms": "Fragment SMS",
        "SentCodeTypeSmsPhrase": "SMS phrase",
        "SentCodeTypeSmsWord": "SMS word",
        "CodeTypeSms": "SMS",
        "CodeTypeCall": "phone call",
        "CodeTypeFlashCall": "flash call",
        "CodeTypeMissedCall": "missed call",
        "CodeTypeFragmentSms": "Fragment SMS",
    }
    return labels.get(name, name)


def describe_login_code_delivery(sent_code: Any, *, operation: str) -> LoginCodeDelivery:
    """Return safe diagnostics from a Telethon SentCode object."""

    code_type = getattr(sent_code, "type", None)
    return LoginCodeDelivery(
        operation=operation,
        delivery_method=_sent_code_method_label(code_type) or sent_code.__class__.__name__,
        next_method=_sent_code_method_label(getattr(sent_code, "next_type", None)),
        timeout_seconds=getattr(sent_code, "timeout", None),
        code_length=getattr(code_type, "length", None),
    )


async def _send_login_code_request(client: TelegramClientProtocol, phone: str, *, operation: str) -> Any:
    """Request a Telegram login code and convert low-level failures."""

    try:
        return await client.send_code_request(phone)
    except Exception as exc:
        if _is_flood_wait(exc):
            _raise_domain_flood_wait(exc, operation)
        raise TelegramLoginCodeRequestFailed(operation, exc) from exc


async def ensure_telegram_login(
    client: TelegramClientProtocol,
    *,
    phone: str,
    code_callback: Callable[[], str | Awaitable[str]],
    password_callback: Callable[[], str | Awaitable[str]],
    code_sent_callback: Callable[[LoginCodeDelivery], Any | Awaitable[Any]] | None = None,
    code_invalid_callback: Callable[[int, int], Any | Awaitable[Any]] | None = None,
    password_needed_error_types: tuple[type[BaseException], ...] | None = None,
    code_invalid_error_types: tuple[type[BaseException], ...] | None = None,
    code_attempts: int = 3,
) -> LoginResult:
    """Connect and authorize a Telegram user session, reusing it when possible."""

    await client.connect()
    if await client.is_user_authorized():
        return LoginResult(reused_session=True)

    sent_code = await _send_login_code_request(client, phone, operation="send_login_code")
    phone_code_hash = getattr(sent_code, "phone_code_hash", None)
    if code_sent_callback is not None:
        callback_result = code_sent_callback(describe_login_code_delivery(sent_code, operation="send_login_code"))
        if inspect.isawaitable(callback_result):
            await callback_result

    password_errors = password_needed_error_types or _default_password_error_types()
    code_invalid_errors = code_invalid_error_types or _default_code_invalid_error_types()
    attempts = max(1, code_attempts)
    attempt = 1
    while attempt <= attempts:
        try:
            code = code_callback()
            if inspect.isawaitable(code):
                code = await code
        except TelegramLoginCodeResendRequested:
            sent_code = await _send_login_code_request(client, phone, operation="resend_login_code")
            phone_code_hash = getattr(sent_code, "phone_code_hash", None)
            if code_sent_callback is not None:
                callback_result = code_sent_callback(describe_login_code_delivery(sent_code, operation="resend_login_code"))
                if inspect.isawaitable(callback_result):
                    await callback_result
            continue
        sign_in_kwargs: dict[str, Any] = {"phone": phone, "code": code}
        if phone_code_hash:
            sign_in_kwargs["phone_code_hash"] = phone_code_hash
        try:
            await client.sign_in(**sign_in_kwargs)
            return LoginResult(reused_session=False)
        except code_invalid_errors as exc:  # type: ignore[misc]
            if attempt >= attempts:
                raise TelegramLoginCodeInvalid(
                    "Telegram login code was invalid or expired. Check the latest code in Telegram and run ragram start again."
                ) from exc
            if code_invalid_callback is not None:
                callback_result = code_invalid_callback(attempt, attempts)
                if inspect.isawaitable(callback_result):
                    await callback_result
            attempt += 1
            continue
        except password_errors:  # type: ignore[misc]
            password = password_callback()
            if inspect.isawaitable(password):
                password = await password
            await client.sign_in(password=password)
            return LoginResult(reused_session=False, required_2fa=True)

    raise TelegramLoginCodeInvalid("Telegram login code was invalid or expired.")


def _entity_kind(dialog: Any, entity: Any) -> str | None:
    if bool(getattr(entity, "broadcast", False)):
        return "channel"
    if bool(getattr(entity, "megagroup", False)):
        return "megagroup"
    if bool(getattr(dialog, "is_group", False)) or bool(getattr(entity, "is_group", False)):
        return "group"
    if bool(getattr(dialog, "is_channel", False)):
        return "channel"
    return None


def _dialog_date(dialog: Any) -> datetime | None:
    message = getattr(dialog, "message", None)
    message_date = getattr(message, "date", None)
    return message_date or getattr(dialog, "date", None)


def dialog_from_entity(entity: Any, *, raw: Any = None, date: datetime | None = None, unread_count: int | None = None) -> TelegramDialog:
    """Convert a Telethon-like entity into RagRam dialog info."""

    kind = _entity_kind(raw, entity) or "channel"
    title = getattr(entity, "title", None) or getattr(raw, "title", None) or str(getattr(entity, "id", "unknown"))
    return TelegramDialog(
        entity_id=int(getattr(entity, "id")),
        title=title,
        username=getattr(entity, "username", None),
        kind=kind,
        last_message_date=date,
        unread_count=unread_count,
        raw=raw or entity,
    )


def dialog_from_telethon_dialog(dialog: Any) -> TelegramDialog | None:
    """Convert a Telethon dialog into RagRam dialog info, skipping user chats."""

    entity = getattr(dialog, "entity", dialog)
    kind = _entity_kind(dialog, entity)
    if kind is None:
        return None
    title = getattr(dialog, "title", None) or getattr(entity, "title", None) or str(getattr(entity, "id", "unknown"))
    return TelegramDialog(
        entity_id=int(getattr(entity, "id")),
        title=title,
        username=getattr(entity, "username", None),
        kind=kind,
        last_message_date=_dialog_date(dialog),
        unread_count=getattr(dialog, "unread_count", None),
        raw=dialog,
    )


def _is_flood_wait(exc: BaseException) -> bool:
    if telethon_errors is not None:
        flood_wait_type = getattr(telethon_errors, "FloodWaitError", None)
        if isinstance(flood_wait_type, type) and isinstance(exc, flood_wait_type):
            return True
    return hasattr(exc, "seconds") and "FloodWait" in exc.__class__.__name__


def _raise_domain_flood_wait(exc: BaseException, operation: str) -> None:
    seconds = int(getattr(exc, "seconds", 0))
    raise TelegramFloodWait(seconds=seconds, operation=operation) from exc


async def list_accessible_dialogs(client: TelegramClientProtocol, *, limit: int | None = None) -> list[TelegramDialog]:
    """List accessible channels/groups sorted by last message date descending."""

    try:
        iterator = client.iter_dialogs(limit=limit)
        dialogs: list[TelegramDialog] = []
        async for raw_dialog in iterator:
            dialog = dialog_from_telethon_dialog(raw_dialog)
            if dialog is not None:
                dialogs.append(dialog)
    except Exception as exc:
        if _is_flood_wait(exc):
            _raise_domain_flood_wait(exc, "list_dialogs")
        raise

    return sorted(
        dialogs,
        key=lambda item: (item.last_message_date is not None, item.last_message_date or datetime.min),
        reverse=True,
    )


def filter_dialogs(dialogs: Iterable[TelegramDialog], query: str) -> list[TelegramDialog]:
    """Filter dialogs by title, username, kind, or entity ID."""

    items = list(dialogs)
    needle = query.strip().casefold()
    if not needle:
        return items
    return [
        dialog
        for dialog in items
        if needle in dialog.title.casefold()
        or (dialog.username is not None and needle in dialog.username.casefold())
        or needle in str(dialog.entity_id)
        or needle in dialog.kind.casefold()
    ]


def normalize_entity_input(value: str) -> str | int:
    """Normalize a custom channel/group input for matching or Telethon lookup."""

    normalized = value.strip()
    for prefix in ("https://t.me/", "http://t.me/", "https://telegram.me/", "http://telegram.me/"):
        if normalized.casefold().startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    normalized = normalized.strip().strip("/")
    if normalized.startswith("@"):
        normalized = normalized[1:]
    if normalized.lstrip("-").isdigit():
        return int(normalized)
    return normalized


def _matches_known_dialog(dialog: TelegramDialog, normalized: str | int, original: str) -> bool:
    if isinstance(normalized, int):
        return dialog.entity_id == normalized
    folded = normalized.casefold()
    original_folded = original.strip().casefold()
    return (
        dialog.title.casefold() == original_folded
        or dialog.title.casefold() == folded
        or (dialog.username is not None and dialog.username.casefold() == folded)
    )


async def resolve_dialog_choice(
    client: TelegramClientProtocol,
    value: str,
    known_dialogs: Iterable[TelegramDialog],
) -> TelegramDialog:
    """Resolve a custom title/username/URL/entity ID into a dialog."""

    normalized = normalize_entity_input(value)
    for dialog in known_dialogs:
        if _matches_known_dialog(dialog, normalized, value):
            return dialog

    try:
        entity = await client.get_entity(normalized)
    except Exception as exc:
        if _is_flood_wait(exc):
            _raise_domain_flood_wait(exc, "resolve_entity")
        raise DialogNotFoundError(f"Could not resolve Telegram entity: {value}") from exc
    return dialog_from_entity(entity, raw=entity)


def message_record_from_telethon(
    message: Any,
    *,
    entity_id: int,
    channel_username: str | None = None,
) -> MessageRecord | None:
    """Convert a Telethon-like message into a raw text MessageRecord."""

    text = getattr(message, "text", None) or getattr(message, "message", None)
    if not text or not str(text).strip():
        return None

    message_id = int(getattr(message, "id"))
    date = getattr(message, "date")
    if not isinstance(date, datetime):
        raise ValueError("Telegram message date is required")

    reply_to_id = getattr(message, "reply_to_msg_id", None)
    if reply_to_id is None:
        reply_to = getattr(message, "reply_to", None)
        reply_to_id = getattr(reply_to, "reply_to_msg_id", None)

    forward = getattr(message, "fwd_from", None)
    forward_info = None if forward is None else {"repr": repr(forward)}
    sender = getattr(message, "sender", None)
    sender_name = getattr(sender, "title", None) or getattr(sender, "username", None)
    sender_id = getattr(message, "sender_id", None)
    message_link = f"https://t.me/{channel_username}/{message_id}" if channel_username else None

    return MessageRecord(
        entity_id=entity_id,
        message_id=message_id,
        date=date,
        text=str(text),
        sender_id=str(sender_id) if sender_id is not None else None,
        sender_name=sender_name,
        reply_to_id=reply_to_id,
        forward_info=forward_info,
        message_link=message_link,
        raw={"id": message_id, "date": date.isoformat()},
    )


async def iter_text_message_records(
    client: Any,
    entity: Any,
    *,
    entity_id: int,
    scope: IndexScope,
    channel_username: str | None = None,
    resume_before_message_id: int | None = None,
):
    """Stream text-only Telegram messages for a selected scope."""

    kwargs: dict[str, Any] = {}
    if scope.limit is not None:
        kwargs["limit"] = scope.limit
    if resume_before_message_id is not None:
        kwargs["max_id"] = resume_before_message_id

    start_date = scope.start_date
    try:
        async for message in client.iter_messages(entity, **kwargs):
            date = getattr(message, "date", None)
            if start_date is not None and isinstance(date, datetime) and date < start_date:
                break
            record = message_record_from_telethon(
                message,
                entity_id=entity_id,
                channel_username=channel_username,
            )
            if record is not None:
                yield record
    except TelegramFloodWait:
        raise
    except Exception as exc:
        if _is_flood_wait(exc):
            _raise_domain_flood_wait(exc, "iter_messages")
        raise


async def iter_text_messages(
    client: Any,
    entity: Any,
    *,
    entity_id: int,
    scope: IndexScope,
    channel_username: str | None = None,
) -> list[MessageRecord]:
    """Fetch text-only Telegram messages for a selected scope."""

    return [
        record
        async for record in iter_text_message_records(
            client,
            entity,
            entity_id=entity_id,
            scope=scope,
            channel_username=channel_username,
        )
    ]
