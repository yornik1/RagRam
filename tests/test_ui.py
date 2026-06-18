from __future__ import annotations

import runpy
import subprocess
from pathlib import Path

from ragram.config import ChannelConfig, IndexingConfig, RagRamConfig, UiConfig, app_paths
from ragram.rag import GroundedAnswer
from ragram.ui import (
    build_ui_launch_plan,
    build_ui_state,
    first_available_port,
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
    paths = app_paths(tmp_path / "home")
    calls = []

    def fake_popen(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return object()

    plan = launch_streamlit_ui(paths=paths, port=8502, popen=fake_popen)

    assert plan.url == "http://localhost:8502"
    assert calls[0]["command"] == plan.command
    assert calls[0]["env"]["RAGRAM_HOME"] == str(paths.home)
    assert calls[0]["stderr"] == subprocess.STDOUT
    assert calls[0]["start_new_session"] is True
    assert (paths.logs_dir / "streamlit-8502.log").exists()
