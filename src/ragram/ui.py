"""Local Streamlit UI helpers for RagRam."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
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



def ui_pid_path(paths: AppPaths) -> Path:
    """Return the pid-file path for the detached local UI process."""

    return paths.logs_dir / "streamlit.pid"


def _extract_streamlit_port(command: str) -> int | None:
    parts = command.split()
    try:
        index = parts.index("--server.port")
    except ValueError:
        return None
    if index + 1 >= len(parts):
        return None
    try:
        return int(parts[index + 1])
    except ValueError:
        return None


def _list_ragram_streamlit_processes(*, app_path: Path | None = None) -> list[tuple[int, int | None, str]]:
    """Return running RagRam Streamlit processes as (pid, port, command)."""

    target = str(app_path or streamlit_app_path())
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    processes: list[tuple[int, int | None, str]] = []
    current_pid = os.getpid()
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        if " -m streamlit run " not in f" {command} " or target not in command:
            continue
        processes.append((pid, _extract_streamlit_port(command), command))
    return processes


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def stop_existing_streamlit_ui(*, paths: AppPaths, wait_seconds: float = 2.0) -> list[int]:
    """Stop existing RagRam UI processes before starting a new one."""

    stopped: list[int] = []
    pid_file = ui_pid_path(paths)
    candidate_pids = {pid for pid, _, _ in _list_ragram_streamlit_processes()}
    if pid_file.exists():
        try:
            payload = json.loads(pid_file.read_text())
            pid = int(payload.get("pid"))
            if _process_alive(pid):
                candidate_pids.add(pid)
        except Exception:
            pass
    for pid in sorted(candidate_pids):
        try:
            os.kill(pid, signal.SIGTERM)
            stopped.append(pid)
        except OSError:
            continue
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline and any(_process_alive(pid) for pid in stopped):
        time.sleep(0.05)
    for pid in stopped:
        if _process_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    if pid_file.exists():
        pid_file.unlink(missing_ok=True)
    return stopped


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
    """Launch exactly one detached local Streamlit UI process."""

    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    stop_existing_streamlit_ui(paths=paths)
    plan = build_ui_launch_plan(paths=paths, port=first_available_port(port))
    log_path = paths.logs_dir / f"streamlit-{plan.port}.log"
    try:
        with log_path.open("ab") as log_file:
            process = popen(
                plan.command,
                env=plan.environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )  # noqa: S603 - command is package-local and deterministic.
    except FileNotFoundError as exc:  # pragma: no cover - defensive around broken Python executable.
        raise RuntimeError("Could not launch Streamlit with the current Python executable.") from exc
    pid = getattr(process, "pid", None)
    if pid is not None:
        ui_pid_path(paths).write_text(
            json.dumps({"pid": pid, "port": plan.port, "url": plan.url, "command": plan.command}, ensure_ascii=False)
        )
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
