from pathlib import Path

from ragram.config import (
    RagRamConfig,
    credential_prompt_specs,
    default_config,
    ensure_app_layout,
    load_config,
    save_config,
    scaffold_status,
)


def test_ensure_app_layout_creates_expected_local_files(tmp_path):
    paths = ensure_app_layout(tmp_path / ".ragram")

    assert paths.home == tmp_path / ".ragram"
    assert paths.config_path == paths.home / "config.toml"
    assert paths.sessions_dir.is_dir()
    assert paths.data_dir.is_dir()
    assert paths.sqlite_path == paths.data_dir / "ragram.sqlite"
    assert paths.sqlite_path.exists()
    assert paths.chroma_dir.is_dir()
    assert paths.logs_dir.is_dir()


def test_config_roundtrip_preserves_model_and_channel_choices(tmp_path):
    paths = ensure_app_layout(tmp_path / ".ragram")
    config = default_config()
    config.telegram.api_id = 12345
    config.telegram.api_hash = "hash-secret"
    config.telegram.phone = "+15550000000"
    config.channel.entity_id = 777
    config.channel.title = "Тестовый канал"
    config.indexing.embedding_model = "intfloat/multilingual-e5-small"
    config.indexing.answer_model = "qwen3:8b"
    config.indexing.summarization_model = "qwen3:4b"
    config.ui.port = 8600

    save_config(config, paths.config_path)
    loaded = load_config(paths.config_path)

    assert isinstance(loaded, RagRamConfig)
    assert loaded.telegram.api_id == 12345
    assert loaded.telegram.api_hash == "hash-secret"
    assert loaded.channel.entity_id == 777
    assert loaded.channel.title == "Тестовый канал"
    assert loaded.indexing.embedding_model == "intfloat/multilingual-e5-small"
    assert loaded.indexing.answer_model == "qwen3:8b"
    assert loaded.indexing.summarization_model == "qwen3:4b"
    assert loaded.ui.port == 8600


def test_load_config_missing_file_returns_defaults(tmp_path):
    loaded = load_config(tmp_path / "missing.toml")

    assert loaded.indexing.scope_type == "last_n"
    assert loaded.indexing.scope_value == "1000"
    assert loaded.indexing.embedding_model == "ai-forever/ru-en-RoSBERTa"
    assert loaded.indexing.answer_model == "qwen3:4b"
    assert loaded.indexing.summarization_model == "qwen3:4b"
    assert loaded.ui.port == 8501


def test_scaffold_status_uses_ragram_home_without_dotenv(monkeypatch, tmp_path):
    home = tmp_path / "custom-ragram"
    monkeypatch.setenv("RAGRAM_HOME", str(home))

    status = scaffold_status()

    assert status.app_home == home
    assert status.config_path == home / "config.toml"
    assert status.sqlite_path == home / "data" / "ragram.sqlite"
    assert status.uses_dotenv is False


def test_credential_prompt_specs_mark_secrets_as_hidden():
    specs = {spec.key: spec for spec in credential_prompt_specs()}

    assert specs["api_id"].secret is False
    assert specs["api_hash"].secret is True
    assert specs["phone"].secret is False
    assert specs["code"].secret is False
    assert specs["password"].secret is True


def test_app_layout_and_config_use_restrictive_permissions(tmp_path):
    import os
    import stat

    paths = ensure_app_layout(tmp_path / ".ragram")
    config = default_config()
    config.telegram.api_hash = "secret"
    save_config(config, paths.config_path)

    if os.name == "posix":
        assert stat.S_IMODE(paths.home.stat().st_mode) == 0o700
        assert stat.S_IMODE(paths.sessions_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(paths.config_path.stat().st_mode) == 0o600
