"""Progress helpers for RagRam ingestion and indexing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rich.progress import BarColumn, Progress, TaskID, TaskProgressColumn, TextColumn, TimeRemainingColumn


@dataclass(frozen=True)
class RateSnapshot:
    """Simple rate/ETA calculation for progress reporting."""

    completed: int
    total: int | None
    elapsed_seconds: float

    @property
    def items_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.completed / self.elapsed_seconds

    @property
    def remaining(self) -> int | None:
        if self.total is None:
            return None
        return max(self.total - self.completed, 0)

    @property
    def eta_seconds(self) -> float | None:
        remaining = self.remaining
        rate = self.items_per_second
        if remaining is None or rate <= 0:
            return None
        return remaining / rate


class ProgressReporter(Protocol):
    def start(self, *, total: int | None = None) -> None: ...

    def advance(self, count: int = 1) -> None: ...

    def finish(self, completed: int) -> None: ...

    def fail(self, error: str) -> None: ...


class NullProgressReporter:
    """No-op progress reporter for tests and non-interactive paths."""

    def start(self, *, total: int | None = None) -> None:
        return None

    def advance(self, count: int = 1) -> None:
        return None

    def finish(self, completed: int) -> None:
        return None

    def fail(self, error: str) -> None:
        return None


class InMemoryProgressReporter:
    """Progress reporter used by tests to assert event order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int | str | None]] = []

    def start(self, *, total: int | None = None) -> None:
        self.events.append(("start", total))

    def advance(self, count: int = 1) -> None:
        self.events.append(("advance", count))

    def finish(self, completed: int) -> None:
        self.events.append(("finish", completed))

    def fail(self, error: str) -> None:
        self.events.append(("fail", error))


class RichProgressReporter:
    """Rich-backed progress reporter with ETA for interactive ingestion."""

    def __init__(self, description: str = "Fetching messages") -> None:
        self.description = description
        self._progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )
        self._task_id: TaskID | None = None

    def start(self, *, total: int | None = None) -> None:
        self._progress.start()
        self._task_id = self._progress.add_task(self.description, total=total)

    def advance(self, count: int = 1) -> None:
        if self._task_id is not None:
            self._progress.advance(self._task_id, count)

    def finish(self, completed: int) -> None:
        if self._task_id is not None:
            self._progress.update(self._task_id, completed=completed)
        self._progress.stop()

    def fail(self, error: str) -> None:
        self._progress.stop()
