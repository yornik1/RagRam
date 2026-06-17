from typer.testing import CliRunner

from ragram.cli import app


runner = CliRunner()


def test_package_exposes_cli_app():
    assert app.info.name == "ragram"


def test_status_command_runs_without_external_services():
    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "RagRam status" in result.stdout
    assert "No paid APIs" in result.stdout
    assert "SQLite" in result.stdout


def test_start_no_ui_runs_without_external_services(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGRAM_HOME", str(tmp_path / "ragram-home"))

    result = runner.invoke(app, ["start", "--no-ui"])

    assert result.exit_code == 0
    assert "RagRam start" in result.stdout
    assert "no paid APIs" in result.stdout


def test_root_version_option_runs_without_command():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "RagRam 0.1.0" in result.stdout


def test_start_no_ui_creates_local_app_layout(tmp_path, monkeypatch):
    home = tmp_path / "ragram-home"
    monkeypatch.setenv("RAGRAM_HOME", str(home))

    result = runner.invoke(app, ["start", "--no-ui"])

    assert result.exit_code == 0
    assert (home / "config.toml").parent.exists()
    assert (home / "sessions").is_dir()
    assert (home / "data" / "ragram.sqlite").exists()
    assert (home / "data" / "chroma").is_dir()
    assert (home / "logs").is_dir()
