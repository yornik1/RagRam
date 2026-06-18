"""Local Streamlit UI helpers for RagRam."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import RagRamConfig
from .models import AppPaths
from .rag import GroundedAnswer


@dataclass(frozen=True)
class UiLaunchPlan:
    """Subprocess launch details for RagRam's local UI."""

    port: int
    url: str
    command: list[str]
    app_module_path: Path
    python_executable: str
    environment: dict[str, str]


@dataclass(frozen=True)
class UiState:
    """Channel/model state displayed at the top of the local UI."""

    channel_title: str
    channel_username: str | None
    entity_id: int | None
    embedding_model: str
    answer_model: str
    summarization_model: str


DEFAULT_TOP_K = 8
MIN_TOP_K = 1
MAX_TOP_K = 20


def port_available(port: int, *, host: str = "127.0.0.1") -> bool:
    """Return whether a localhost TCP port can be bound for the UI."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def first_available_port(start_port: int, *, host: str = "127.0.0.1", limit: int = 50) -> int:
    """Return the requested port or the next nearby free localhost port."""

    for port in range(start_port, min(65535, start_port + limit) + 1):
        if port_available(port, host=host):
            return port
    raise RuntimeError(f"No free local UI port found near {start_port}.")


def streamlit_app_path() -> Path:
    """Return the package-local Streamlit app entrypoint."""

    return Path(__file__).with_name("streamlit_app.py")


def build_ui_launch_plan(*, paths: AppPaths, port: int) -> UiLaunchPlan:
    """Build a deterministic localhost Streamlit launch plan."""

    app_path = streamlit_app_path()
    python_executable = sys.executable
    url = f"http://localhost:{port}"
    command = [
        python_executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        "localhost",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
    ]
    environment = dict(os.environ)
    environment["RAGRAM_HOME"] = str(paths.home)
    return UiLaunchPlan(
        port=port,
        url=url,
        command=command,
        app_module_path=app_path,
        python_executable=python_executable,
        environment=environment,
    )


def launch_streamlit_ui(*, paths: AppPaths, port: int, popen: Any = subprocess.Popen) -> UiLaunchPlan:
    """Launch the local Streamlit UI in a detached background process."""

    plan = build_ui_launch_plan(paths=paths, port=first_available_port(port))
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = paths.logs_dir / f"streamlit-{plan.port}.log"
    try:
        log_file = log_path.open("ab")
        popen(
            plan.command,
            env=plan.environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )  # noqa: S603 - command is package-local and deterministic.
    except FileNotFoundError as exc:  # pragma: no cover - defensive around broken Python executable.
        raise RuntimeError("Could not launch Streamlit with the current Python executable.") from exc
    return plan


def build_ui_state(config: RagRamConfig) -> UiState:
    """Extract selected channel and model choices for display."""

    return UiState(
        channel_title=config.channel.title or "No channel selected",
        channel_username=config.channel.username,
        entity_id=config.channel.entity_id,
        embedding_model=config.indexing.embedding_model,
        answer_model=config.indexing.answer_model,
        summarization_model=config.indexing.summarization_model,
    )


def source_label(source: Mapping[str, Any]) -> str:
    """Render a compact source label with dates, message IDs, and distance."""

    chunk_id = source.get("chunk_id", "unknown")
    start = source.get("message_id_start", "?")
    end = source.get("message_id_end", start)
    date_start = str(source.get("date_start", "?")).split("T", 1)[0]
    date_end = str(source.get("date_end", date_start)).split("T", 1)[0]
    distance = source.get("distance")
    distance_part = "" if distance is None else f", distance {float(distance):.3f}"
    return f"Source {chunk_id}: messages {start}–{end}, dates {date_start}–{date_end}{distance_part}"


def render_answer_payload(answer: GroundedAnswer, *, show_raw_context: bool) -> dict[str, Any]:
    """Prepare answer data for Streamlit rendering and tests."""

    return {
        "answer": answer.answer,
        "sources": answer.sources,
        "raw_context": answer.raw_context if show_raw_context else None,
    }
