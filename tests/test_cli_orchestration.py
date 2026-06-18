from __future__ import annotations

from datetime import UTC, datetime

from typer.testing import CliRunner

from ragram.cli import _normalize_login_code, app
from ragram.config import ChannelConfig, RagRamConfig, TelegramConfig, save_config
from ragram.models import ChunkRecord, MessageRecord
from ragram.storage import SQLiteStore

runner = CliRunner()


class FakeVectorStore:
    def __init__(self):
        self.upserts = []

    def upsert_chunks(self, chunks, *, entity_id):
        self.upserts.append({"chunks": chunks, "entity_id": entity_id})
        return len(chunks)


class FakeAuthClient:
    def __init__(self, authorized: bool = False):
        self.authorized = authorized
        self.connected = False

    async def connect(self):
        self.connected = True

    async def is_user_authorized(self):
        return self.authorized

    async def disconnect(self):
        self.connected = False


def test_login_code_normalization_accepts_human_separators_and_rejects_app_token_shape():
    assert _normalize_login_code("1 2 3 4 5") == "12345"
    assert _normalize_login_code("1-2-3-4-5") == "12345"
    assert not _normalize_login_code("8cGyeMmwPVY").isdigit()


def test_start_no_ui_creates_config_and_initializes_sqlite(tmp_path, monkeypatch):
    home = tmp_path / "ragram-home"
    monkeypatch.setenv("RAGRAM_HOME", str(home))

    result = runner.invoke(app, ["start", "--no-ui"])

    assert result.exit_code == 0
    assert (home / "config.toml").exists()
    assert (home / "data" / "ragram.sqlite").exists()
    assert "Telegram MTProto access needs api_id and api_hash" in result.stdout
    assert "Open RagRam" not in result.stdout


def test_start_launches_ui_and_prints_url_when_channel_is_configured(tmp_path, monkeypatch):
    home = tmp_path / "ragram-home"
    monkeypatch.setenv("RAGRAM_HOME", str(home))
    config = RagRamConfig(channel=ChannelConfig(entity_id=100, title="Test Channel", username="test"))
    save_config(config, home / "config.toml")
    launches = []

    def fake_launch_streamlit_ui(*, paths, port):
        from ragram.ui import build_ui_launch_plan

        plan = build_ui_launch_plan(paths=paths, port=port)
        launches.append(plan)
        return plan

    monkeypatch.setattr("ragram.cli.launch_streamlit_ui", fake_launch_streamlit_ui)

    result = runner.invoke(app, ["start", "--ui-port", "8601"])

    assert result.exit_code == 0
    assert launches[0].url == "http://localhost:8601"
    assert "Open RagRam: http://localhost:8601" in result.stdout


def test_start_interactive_ready_index_skips_setup_and_launches_ui(tmp_path, monkeypatch):
    home = tmp_path / "ragram-home"
    monkeypatch.setenv("RAGRAM_HOME", str(home))
    monkeypatch.setattr("ragram.cli._is_interactive", lambda: True)
    config = RagRamConfig(channel=ChannelConfig(entity_id=100, title="Ready Channel", username="ready"))
    save_config(config, home / "config.toml")
    store = SQLiteStore(home / "data" / "ragram.sqlite")
    store.initialize()
    store.upsert_chunk(
        ChunkRecord(
            chunk_id="ready-1",
            entity_id=100,
            embedding_model=config.indexing.embedding_model,
            message_id_start=1,
            message_id_end=1,
            date_start=datetime(2025, 1, 1, tzinfo=UTC),
            date_end=datetime(2025, 1, 1, tzinfo=UTC),
            text="ready context",
            token_count=2,
            metadata={},
        )
    )
    launches = []

    async def fail_setup(*args, **kwargs):
        raise AssertionError("interactive setup should not run when local index is ready")

    def fake_launch_streamlit_ui(*, paths, port):
        from ragram.ui import build_ui_launch_plan

        plan = build_ui_launch_plan(paths=paths, port=port)
        launches.append(plan)
        return plan

    monkeypatch.setattr("ragram.cli._run_interactive_start", fail_setup)
    monkeypatch.setattr("ragram.cli.launch_streamlit_ui", fake_launch_streamlit_ui)

    result = runner.invoke(app, ["start", "--ui-port", "8602"])

    assert result.exit_code == 0
    assert launches[0].url == "http://localhost:8602"
    assert "Existing local index found; skipping setup" in result.stdout
    assert "Open RagRam: http://localhost:8602" in result.stdout


def test_restart_clear_data_requires_confirmation_and_recreates_layout(tmp_path, monkeypatch):
    home = tmp_path / "ragram-home"
    monkeypatch.setenv("RAGRAM_HOME", str(home))
    store = SQLiteStore(home / "data" / "ragram.sqlite")
    store.initialize()
    store.upsert_message(MessageRecord(entity_id=100, message_id=1, date=datetime(2025, 1, 1, tzinfo=UTC), text="old"))
    chroma_dir = home / "data" / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    (chroma_dir / "index.bin").write_text("old")

    cancelled = runner.invoke(app, ["restart", "--clear-data"], input="n\n")
    assert cancelled.exit_code == 0
    assert "cancelled" in cancelled.stdout.lower()
    assert store.message_count() == 1

    confirmed = runner.invoke(app, ["restart", "--clear-data"], input="y\n")
    assert confirmed.exit_code == 0
    assert "Local raw data and Chroma index cleared" in confirmed.stdout
    assert SQLiteStore(home / "data" / "ragram.sqlite").message_count() == 0
    assert chroma_dir.is_dir()
    assert not (chroma_dir / "index.bin").exists()


def test_status_reports_channel_counts_models_session_and_ui(tmp_path, monkeypatch):
    home = tmp_path / "ragram-home"
    monkeypatch.setenv("RAGRAM_HOME", str(home))
    config = RagRamConfig(channel=ChannelConfig(entity_id=100, title="Test Channel", username="test"))
    save_config(config, home / "config.toml")
    session_file = home / "sessions" / "telegram.session"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text("session")
    store = SQLiteStore(home / "data" / "ragram.sqlite")
    store.initialize()
    store.upsert_message(MessageRecord(entity_id=100, message_id=1, date=datetime(2025, 1, 1, tzinfo=UTC), text="raw"))

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Telegram session" in result.stdout
    assert "Selected channel" in result.stdout
    assert "Test Channel" in result.stdout
    assert "Raw messages" in result.stdout
    assert "Embedding model" in result.stdout
    assert "Answer model" in result.stdout
    assert "UI URL" in result.stdout


def test_restart_guided_reconfigure_models_without_clearing_data(tmp_path, monkeypatch):
    home = tmp_path / "ragram-home"
    monkeypatch.setenv("RAGRAM_HOME", str(home))
    config = RagRamConfig(channel=ChannelConfig(entity_id=100, title="Old", username="old"))
    save_config(config, home / "config.toml")

    result = runner.invoke(app, ["restart", "--reconfigure"], input="n\ny\nintfloat/multilingual-e5-small\nqwen3:8b\nqwen3:8b\nn\nn\n")

    assert result.exit_code == 0
    assert "Reconfigured local model choices" in result.stdout
    from ragram.config import load_config

    loaded = load_config(home / "config.toml")
    assert loaded.indexing.embedding_model == "intfloat/multilingual-e5-small"
    assert loaded.indexing.answer_model == "qwen3:8b"


def test_channel_search_filters_dialog_choices(monkeypatch):
    from ragram.cli import _filter_dialog_choices
    from ragram.telegram_client import TelegramDialog

    dialogs = [TelegramDialog(entity_id=1, title="Python News", kind="channel"), TelegramDialog(entity_id=2, title="Other", kind="group")]

    filtered = _filter_dialog_choices(dialogs, "python")

    assert len(filtered) == 1
    assert filtered[0].entity_id == 1


def test_start_interactive_first_run_with_fakes_configures_channel_and_models(tmp_path, monkeypatch):
    from ragram.config import load_config
    from ragram.telegram_client import LoginResult, TelegramDialog

    home = tmp_path / "ragram-home"
    monkeypatch.setenv("RAGRAM_HOME", str(home))
    monkeypatch.setattr("ragram.cli._is_interactive", lambda: True)

    class Prompt:
        def __init__(self, value=None, choices=None, message=""):
            self.value = value
            self.choices = choices or []
            self.message = message

        def execute(self):
            if self.value == "__first_choice__":
                return self.choices[0]["value"]
            return self.value

    class FakeInquirer:
        def text(self, *, message, default=None):
            values = {
                "Telegram api_id (digits only):": "12345",
                "Telegram phone number (international format, e.g. +15551234567):": "+15550000000",
                "Filter channels/groups by title (optional; press Enter to show all):": "python",
                "How many recent messages?": "25",
            }
            return Prompt(values.get(message, default or ""), message=message)

        def secret(self, *, message):
            return Prompt("hash-secret", message=message)

        def select(self, *, message, choices, default=None):
            values = {
                "Telegram login method:": "code",
                "Indexing scope:": "last_n",
                "Embedding model:": "intfloat/multilingual-e5-small",
                "Answer model (Ollama):": "qwen3:8b",
                "Summarization model (Ollama):": "qwen3:8b",
            }
            if message == "Choose one Telegram channel/group:":
                return Prompt("__first_choice__", choices=choices, message=message)
            return Prompt(values[message], choices=choices, message=message)

    async def fake_login(*args, **kwargs):
        return LoginResult(reused_session=False)

    async def fake_dialogs(*args, **kwargs):
        return [
            TelegramDialog(entity_id=100, title="Python News", kind="channel", username="python"),
            TelegramDialog(entity_id=200, title="Other", kind="group"),
        ]

    async def fake_ingest(*args, **kwargs):
        class Result:
            fetched_messages = 0
            saved_messages = 0
            last_message_id = None
        return Result()

    class FakeVectorStore:
        def __init__(self, **kwargs):
            pass

        def upsert_chunks(self, chunks, *, entity_id):
            return len(chunks)

    monkeypatch.setattr("ragram.cli._load_inquirer", lambda: FakeInquirer())
    monkeypatch.setattr("ragram.cli.create_telegram_client", lambda config, paths: FakeAuthClient())
    monkeypatch.setattr("ragram.cli.ensure_telegram_login", fake_login)
    monkeypatch.setattr("ragram.cli.list_accessible_dialogs", fake_dialogs)
    monkeypatch.setattr("ragram.cli.ingest_text_messages", fake_ingest)
    monkeypatch.setattr("ragram.cli.SentenceTransformerEmbeddingProvider", lambda model_name: object())
    monkeypatch.setattr("ragram.cli.ChromaVectorStore", FakeVectorStore)
    monkeypatch.setattr("ragram.cli.launch_streamlit_ui", lambda *, paths, port: type("Plan", (), {"url": f"http://localhost:{port}"})())
    monkeypatch.setattr("ragram.cli.OllamaClient", lambda: type("Client", (), {"model_status": lambda self, models: type("Status", (), {"running": True, "missing_models": ()})()})())

    result = runner.invoke(app, ["start", "--no-ui"])

    assert result.exit_code == 0
    loaded = load_config(home / "config.toml")
    assert loaded.telegram.api_id == 12345
    assert loaded.telegram.api_hash == "hash-secret"
    assert loaded.channel.entity_id == 100
    assert loaded.indexing.scope_value == "25"
    assert loaded.indexing.embedding_model == "intfloat/multilingual-e5-small"
    assert loaded.indexing.answer_model == "qwen3:8b"


def test_start_can_reenter_saved_telegram_credentials_before_login(tmp_path, monkeypatch):
    from ragram.config import load_config
    from ragram.telegram_client import LoginResult, TelegramDialog

    home = tmp_path / "ragram-home"
    monkeypatch.setenv("RAGRAM_HOME", str(home))
    monkeypatch.setattr("ragram.cli._is_interactive", lambda: True)
    save_config(
        RagRamConfig(telegram=TelegramConfig(api_id=111, api_hash="old-hash", phone="+10000000000")),
        home / "config.toml",
    )
    session_file = home / "sessions" / "telegram.session"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text("unauthorized partial session")

    class Prompt:
        def __init__(self, value=None, choices=None):
            self.value = value
            self.choices = choices or []

        def execute(self):
            if self.value == "__first_choice__":
                return self.choices[0]["value"]
            return self.value

    class FakeInquirer:
        def confirm(self, *, message, default=True):
            return Prompt(False)

        def text(self, *, message, default=None):
            values = {
                "Telegram api_id (digits only):": "222",
                "Telegram phone number (international format, e.g. +15551234567):": "+15551234567",
                "Filter channels/groups by title (optional; press Enter to show all):": "",
                "How many recent messages?": "10",
            }
            return Prompt(values.get(message, default or ""))

        def secret(self, *, message):
            return Prompt("new-hash")

        def select(self, *, message, choices, default=None):
            values = {
                "Telegram login method:": "code",
                "Indexing scope:": "last_n",
                "Embedding model:": "intfloat/multilingual-e5-small",
                "Answer model (Ollama):": "qwen3:4b",
                "Summarization model (Ollama):": "qwen3:4b",
            }
            if message == "Choose one Telegram channel/group:":
                return Prompt("__first_choice__", choices=choices)
            return Prompt(values[message], choices=choices)

    async def fake_login(*args, **kwargs):
        return LoginResult(reused_session=False)

    async def fake_dialogs(*args, **kwargs):
        return [TelegramDialog(entity_id=100, title="Python News", kind="channel", username="python")]

    async def fake_ingest(*args, **kwargs):
        class Result:
            fetched_messages = 0
            saved_messages = 0
            last_message_id = None
        return Result()

    class FakeVectorStore:
        def __init__(self, **kwargs):
            pass

        def upsert_chunks(self, chunks, *, entity_id):
            return len(chunks)

    created_clients = []

    def fake_create_client(config, paths):
        created_clients.append((config.api_id, config.api_hash, config.phone))
        return FakeAuthClient()

    monkeypatch.setattr("ragram.cli._load_inquirer", lambda: FakeInquirer())
    monkeypatch.setattr("ragram.cli.create_telegram_client", fake_create_client)
    monkeypatch.setattr("ragram.cli.ensure_telegram_login", fake_login)
    monkeypatch.setattr("ragram.cli.list_accessible_dialogs", fake_dialogs)
    monkeypatch.setattr("ragram.cli.ingest_text_messages", fake_ingest)
    monkeypatch.setattr("ragram.cli.SentenceTransformerEmbeddingProvider", lambda model_name: object())
    monkeypatch.setattr("ragram.cli.ChromaVectorStore", FakeVectorStore)
    monkeypatch.setattr("ragram.cli.OllamaClient", lambda: type("Client", (), {"model_status": lambda self, models: type("Status", (), {"running": True, "missing_models": ()})()})())

    result = runner.invoke(app, ["start", "--no-ui"])

    assert result.exit_code == 0
    assert created_clients[0] == (222, "new-hash", None)
    loaded = load_config(home / "config.toml")
    assert loaded.telegram.api_id == 222
    assert loaded.telegram.phone == "+15551234567"


def test_start_qr_login_does_not_require_phone_number(tmp_path, monkeypatch):
    from ragram.config import load_config
    from ragram.telegram_client import LoginResult, TelegramDialog

    home = tmp_path / "ragram-home"
    monkeypatch.setenv("RAGRAM_HOME", str(home))
    monkeypatch.setattr("ragram.cli._is_interactive", lambda: True)

    class Prompt:
        def __init__(self, value=None, choices=None):
            self.value = value
            self.choices = choices or []

        def execute(self):
            if self.value == "__first_choice__":
                return self.choices[0]["value"]
            return self.value

    class FakeInquirer:
        def text(self, *, message, default=None):
            values = {
                "Telegram api_id (digits only):": "222",
                "Filter channels/groups by title (optional; press Enter to show all):": "",
                "How many recent messages?": "10",
            }
            if message == "Telegram phone number (international format, e.g. +15551234567):":
                raise AssertionError("QR login should not ask for phone")
            return Prompt(values.get(message, default or ""))

        def secret(self, *, message):
            return Prompt("new-hash")

        def select(self, *, message, choices, default=None):
            values = {
                "Telegram login method:": "qr",
                "Indexing scope:": "last_n",
                "Embedding model:": "intfloat/multilingual-e5-small",
                "Answer model (Ollama):": "qwen3:4b",
                "Summarization model (Ollama):": "qwen3:4b",
            }
            if message == "Choose one Telegram channel/group:":
                return Prompt("__first_choice__", choices=choices)
            return Prompt(values[message], choices=choices)

    async def fake_qr_login(*args, **kwargs):
        kwargs["qr_callback"](type("Challenge", (), {"url": "tg://login?token=fake", "expires_at": None})())
        return LoginResult(reused_session=False)

    async def fake_dialogs(*args, **kwargs):
        return [TelegramDialog(entity_id=100, title="Python News", kind="channel", username="python")]

    async def fake_ingest(*args, **kwargs):
        class Result:
            fetched_messages = 0
            saved_messages = 0
            last_message_id = None
        return Result()

    class FakeVectorStore:
        def __init__(self, **kwargs):
            pass

        def upsert_chunks(self, chunks, *, entity_id):
            return len(chunks)

    monkeypatch.setattr("ragram.cli._load_inquirer", lambda: FakeInquirer())
    monkeypatch.setattr("ragram.cli._open_file_if_supported", lambda path: False)
    monkeypatch.setattr("ragram.cli.create_telegram_client", lambda config, paths: FakeAuthClient())
    monkeypatch.setattr("ragram.cli.ensure_telegram_qr_login", fake_qr_login)
    monkeypatch.setattr("ragram.cli.list_accessible_dialogs", fake_dialogs)
    monkeypatch.setattr("ragram.cli.ingest_text_messages", fake_ingest)
    monkeypatch.setattr("ragram.cli.SentenceTransformerEmbeddingProvider", lambda model_name: object())
    monkeypatch.setattr("ragram.cli.ChromaVectorStore", FakeVectorStore)
    monkeypatch.setattr("ragram.cli.OllamaClient", lambda: type("Client", (), {"model_status": lambda self, models: type("Status", (), {"running": True, "missing_models": ()})()})())

    result = runner.invoke(app, ["start", "--no-ui"])

    assert result.exit_code == 0
    assert "QR image saved:" in result.stdout
    assert (home / "qr-login.png").exists()
    loaded = load_config(home / "config.toml")
    assert loaded.telegram.api_id == 222
    assert loaded.telegram.phone is None


def test_restart_clear_data_recreates_sqlite_private_permissions(tmp_path, monkeypatch):
    import os
    import stat

    home = tmp_path / "ragram-home"
    monkeypatch.setenv("RAGRAM_HOME", str(home))
    SQLiteStore(home / "data" / "ragram.sqlite").initialize()

    result = runner.invoke(app, ["restart", "--clear-data"], input="y\n")

    assert result.exit_code == 0
    if os.name == "posix":
        assert stat.S_IMODE((home / "data" / "ragram.sqlite").stat().st_mode) == 0o600


def test_restart_reconfigure_channel_and_scope(tmp_path, monkeypatch):
    from ragram.config import load_config

    home = tmp_path / "ragram-home"
    monkeypatch.setenv("RAGRAM_HOME", str(home))
    config = RagRamConfig(channel=ChannelConfig(entity_id=100, title="Old", username="old"))
    save_config(config, home / "config.toml")

    result = runner.invoke(
        app,
        ["restart", "--reconfigure"],
        input="n\nn\ny\ny\nfrom_year\n2024\nn\n",
    )

    assert result.exit_code == 0
    loaded = load_config(home / "config.toml")
    assert loaded.channel.entity_id is None
    assert loaded.indexing.scope_type == "from_year"
    assert loaded.indexing.scope_value == "2024"


def test_status_reports_actual_ollama_model_availability(tmp_path, monkeypatch):
    home = tmp_path / "ragram-home"
    monkeypatch.setenv("RAGRAM_HOME", str(home))
    save_config(RagRamConfig(), home / "config.toml")

    class FakeOllama:
        def model_status(self, models):
            return type("Status", (), {"running": True, "missing_models": ("qwen3:8b",), "available_models": ("qwen3:4b",)})()

    monkeypatch.setattr("ragram.cli.OllamaClient", lambda: FakeOllama())

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Ollama" in result.stdout
    assert "running" in result.stdout
    assert "Missing local models" in result.stdout
    assert "qwen3:8b" in result.stdout
