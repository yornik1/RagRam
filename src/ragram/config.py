"""Local configuration and app-directory management for RagRam."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

from .models import AppPaths, PromptSpec, ScaffoldStatus

APP_DIR_NAME = ".ragram"
CONFIG_FILE_NAME = "config.toml"
SQLITE_FILE_NAME = "ragram.sqlite"


@dataclass
class TelegramConfig:
    api_id: int | None = None
    api_hash: str | None = None
    phone: str | None = None
    session_name: str = "telegram"


@dataclass
class ChannelConfig:
    entity_id: int | None = None
    access_hash: str | None = None
    title: str | None = None
    username: str | None = None


@dataclass
class IndexingConfig:
    scope_type: str = "last_n"
    scope_value: str = "1000"
    embedding_model: str = "ai-forever/ru-en-RoSBERTa"
    answer_model: str = "qwen3:4b"
    summarization_model: str = "qwen3:4b"


@dataclass
class UiConfig:
    port: int = 8501


@dataclass
class RagRamConfig:
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    ui: UiConfig = field(default_factory=UiConfig)


def default_app_home() -> Path:
    """Return RagRam's default local app directory without creating it."""

    override = os.environ.get("RAGRAM_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / APP_DIR_NAME


def app_paths(app_home: Path | None = None) -> AppPaths:
    """Build all RagRam local paths without creating them."""

    home = (app_home or default_app_home()).expanduser()
    data_dir = home / "data"
    return AppPaths(
        home=home,
        config_path=home / CONFIG_FILE_NAME,
        sessions_dir=home / "sessions",
        data_dir=data_dir,
        sqlite_path=data_dir / SQLITE_FILE_NAME,
        chroma_dir=data_dir / "chroma",
        logs_dir=home / "logs",
    )


def _mkdir_private(path: Path) -> None:
    """Create a directory and restrict it to the current user on POSIX."""

    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        path.chmod(0o700)


def ensure_app_layout(app_home: Path | None = None) -> AppPaths:
    """Create RagRam's local app layout idempotently and return its paths."""

    paths = app_paths(app_home)
    for directory in (paths.home, paths.sessions_dir, paths.data_dir, paths.chroma_dir, paths.logs_dir):
        _mkdir_private(directory)
    paths.sqlite_path.touch(exist_ok=True)
    if os.name == "posix":
        paths.sqlite_path.chmod(0o600)
    return paths


def default_config() -> RagRamConfig:
    """Return default MVP config values."""

    return RagRamConfig()


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        return {}
    return value


def load_config(path: Path | None = None) -> RagRamConfig:
    """Load config TOML, returning defaults when the file is missing."""

    config_path = path or app_paths().config_path
    if not config_path.exists():
        return default_config()

    data = tomllib.loads(config_path.read_text())
    telegram = _section(data, "telegram")
    channel = _section(data, "channel")
    indexing = _section(data, "indexing")
    ui = _section(data, "ui")

    defaults = default_config()
    return RagRamConfig(
        telegram=TelegramConfig(
            api_id=telegram.get("api_id", defaults.telegram.api_id),
            api_hash=telegram.get("api_hash", defaults.telegram.api_hash),
            phone=telegram.get("phone", defaults.telegram.phone),
            session_name=telegram.get("session_name", defaults.telegram.session_name),
        ),
        channel=ChannelConfig(
            entity_id=channel.get("entity_id", defaults.channel.entity_id),
            access_hash=channel.get("access_hash", defaults.channel.access_hash),
            title=channel.get("title", defaults.channel.title),
            username=channel.get("username", defaults.channel.username),
        ),
        indexing=IndexingConfig(
            scope_type=indexing.get("scope_type", defaults.indexing.scope_type),
            scope_value=str(indexing.get("scope_value", defaults.indexing.scope_value)),
            embedding_model=indexing.get("embedding_model", defaults.indexing.embedding_model),
            answer_model=indexing.get("answer_model", defaults.indexing.answer_model),
            summarization_model=indexing.get(
                "summarization_model",
                defaults.indexing.summarization_model,
            ),
        ),
        ui=UiConfig(port=int(ui.get("port", defaults.ui.port))),
    )


def _without_none(value: Any) -> Any:
    """Return dataclass/dict/list data with None values omitted for TOML."""

    if isinstance(value, dict):
        return {key: _without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


def save_config(config: RagRamConfig, path: Path | None = None) -> None:
    """Persist config TOML locally with user-only permissions."""

    config_path = path or app_paths().config_path
    _mkdir_private(config_path.parent)
    temp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    temp_path.write_text(tomli_w.dumps(_without_none(asdict(config))))
    if os.name == "posix":
        temp_path.chmod(0o600)
    temp_path.replace(config_path)
    if os.name == "posix":
        config_path.chmod(0o600)


def scaffold_status(app_home: Path | None = None) -> ScaffoldStatus:
    """Return local status without contacting Telegram, Ollama, Chroma, or UI."""

    paths = app_paths(app_home)
    return ScaffoldStatus(
        app_home=paths.home,
        config_path=paths.config_path,
        sqlite_path=paths.sqlite_path,
        has_config=paths.config_path.exists(),
        has_sqlite=paths.sqlite_path.exists(),
    )


def credential_prompt_specs() -> list[PromptSpec]:
    """Return Telegram credential prompt metadata with hidden secret fields."""

    return [
        PromptSpec("api_id", "Telegram api_id"),
        PromptSpec("api_hash", "Telegram api_hash", secret=True),
        PromptSpec("phone", "Telegram phone number"),
        PromptSpec("code", "Telegram login code"),
        PromptSpec("password", "Telegram 2FA password", secret=True),
    ]
