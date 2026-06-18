from __future__ import annotations

import runpy
import signal
import subprocess
from pathlib import Path

from ragram.config import ChannelConfig, IndexingConfig, RagRamConfig, UiConfig, app_paths
from ragram.rag import GroundedAnswer
from ragram.ui import (
    build_ui_launch_plan,
    build_ui_state,
    first_available_port,
    stop_existing_streamlit_ui,
    ui_pid_path,
    render_answer_payload,
    source_label,
)


def configured_app(tmp_path: Path) -> RagRamConfig:
    return RagRamConfig(
        channel=ChannelConfig(entity_id=123, title="Русский канал", username="ruchannel"),
        indexing=IndexingConfig(
            embedding_model="ai-forever/ru-en-RoSBERTa",
            answer_model="qwen3:4b",
            summarization_model="qwen3:4b",
        ),
        ui=UiConfig(port=8501),
    )


def test_streamlit_entrypoint_imports_when_run_as_script():
    # Streamlit executes this file by path, not as `ragram.streamlit_app`;
    # absolute imports must work without package-relative import context.
    runpy.run_path("src/ragram/streamlit_app.py", run_name="streamlit_app_test")


def test_build_ui_launch_plan_uses_streamlit_localhost_and_printable_url(tmp_path):
    paths = app_paths(tmp_path / "home")

    plan = build_ui_launch_plan(paths=paths, port=8601)

    assert plan.url == "http://localhost:8601"
    assert plan.command[:3] == [plan.python_executable, "-m", "streamlit"]
    assert "run" in plan.command
    assert str(plan.app_module_path).endswith("streamlit_app.py")
    assert "--server.port" in plan.command
    assert "8601" in plan.command
    assert "--server.fileWatcherType" in plan.command
    assert "none" in plan.command
    assert plan.environment["RAGRAM_HOME"] == str(paths.home)


def test_first_available_port_skips_busy_ports(monkeypatch):
    busy = {8501, 8502}

    def fake_port_available(port, *, host="127.0.0.1"):
        return port not in busy

    monkeypatch.setattr("ragram.ui.port_available", fake_port_available)

    assert first_available_port(8501) == 8503


def test_build_ui_state_exposes_selected_channel_and_models(tmp_path):
    state = build_ui_state(configured_app(tmp_path))

    assert state.channel_title == "Русский канал"
    assert state.channel_username == "ruchannel"
    assert state.entity_id == 123
    assert state.embedding_model == "ai-forever/ru-en-RoSBERTa"
    assert state.answer_model == "qwen3:4b"
    assert state.summarization_model == "qwen3:4b"


def test_source_label_shows_message_ids_dates_and_distance():
    label = source_label(
        {
            "chunk_id": "c1",
            "message_id_start": 10,
            "message_id_end": 12,
            "date_start": "2025-01-01T00:00:00+00:00",
            "date_end": "2025-01-02T00:00:00+00:00",
            "distance": 0.23456,
        }
    )

    assert "c1" in label
    assert "messages 10–12" in label
    assert "2025-01-01" in label
    assert "2025-01-02" in label
    assert "distance 0.235" in label


def test_render_answer_payload_hides_or_shows_raw_context():
    answer = GroundedAnswer(
        answer="Ответ из источников",
        sources=[{"chunk_id": "c1", "message_id_start": 10, "message_id_end": 12}],
        raw_context="сырой контекст",
    )

    hidden = render_answer_payload(answer, show_raw_context=False)
    shown = render_answer_payload(answer, show_raw_context=True)

    assert hidden["answer"] == "Ответ из источников"
    assert hidden["sources"][0]["chunk_id"] == "c1"
    assert hidden["raw_context"] is None
    assert shown["raw_context"] == "сырой контекст"


def test_launch_streamlit_ui_uses_fake_popen_and_returns_url(tmp_path, monkeypatch):
    from ragram.ui import launch_streamlit_ui

    monkeypatch.setattr("ragram.ui.port_available", lambda port, *, host="127.0.0.1": True)
    monkeypatch.setattr("ragram.ui._list_ragram_streamlit_processes", lambda: [])
    paths = app_paths(tmp_path / "home")
    calls = []

    class FakeProcess:
        pid = 12345

    def fake_popen(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return FakeProcess()

    plan = launch_streamlit_ui(paths=paths, port=8502, popen=fake_popen)

    assert plan.url == "http://localhost:8502"
    assert calls[0]["command"] == plan.command
    assert calls[0]["env"]["RAGRAM_HOME"] == str(paths.home)
    assert calls[0]["stderr"] == subprocess.STDOUT
    assert calls[0]["start_new_session"] is True
    assert (paths.logs_dir / "streamlit-8502.log").exists()
    assert ui_pid_path(paths).exists()
    assert '"pid": 12345' in ui_pid_path(paths).read_text()


def test_launch_streamlit_ui_stops_existing_ragram_processes(tmp_path, monkeypatch):
    from ragram.ui import launch_streamlit_ui

    monkeypatch.setattr("ragram.ui.port_available", lambda port, *, host="127.0.0.1": True)
    monkeypatch.setattr("ragram.ui._list_ragram_streamlit_processes", lambda: [(111, 8501, "old"), (222, 8502, "old")])
    monkeypatch.setattr("ragram.ui._process_alive", lambda pid: False)
    killed = []

    def fake_kill(pid, sig):
        killed.append((pid, sig))

    class FakeProcess:
        pid = 333

    monkeypatch.setattr("ragram.ui.os.kill", fake_kill)
    paths = app_paths(tmp_path / "home")

    launch_streamlit_ui(paths=paths, port=8501, popen=lambda command, **kwargs: FakeProcess())

    assert killed == [(111, signal.SIGTERM), (222, signal.SIGTERM)]
    assert '"pid": 333' in ui_pid_path(paths).read_text()


def test_stop_existing_streamlit_ui_uses_live_pid_file(tmp_path, monkeypatch):
    paths = app_paths(tmp_path / "home")
    paths.logs_dir.mkdir(parents=True)
    ui_pid_path(paths).write_text('{"pid": 444, "port": 8501}')
    monkeypatch.setattr("ragram.ui._list_ragram_streamlit_processes", lambda: [])
    monkeypatch.setattr("ragram.ui._process_alive", lambda pid: True)
    killed = []
    monkeypatch.setattr("ragram.ui.os.kill", lambda pid, sig: killed.append((pid, sig)))

    stopped = stop_existing_streamlit_ui(paths=paths, wait_seconds=0)

    assert stopped == [444]
    assert killed == [(444, signal.SIGTERM), (444, signal.SIGKILL)]
    assert not ui_pid_path(paths).exists()
