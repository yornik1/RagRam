"""Local LLM integration for RagRam.

The MVP only supports free/local providers. Ollama is the default answer and
summarization backend because it provides a simple localhost API and keeps all
question answering on the user's machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import requests

DEFAULT_ANSWER_MODEL = "qwen3:4b"
DEFAULT_BETTER_ANSWER_MODEL = "qwen3:8b"
ANSWER_MODEL_CHOICES = (DEFAULT_ANSWER_MODEL, DEFAULT_BETTER_ANSWER_MODEL)
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class LLMProvider(Protocol):
    """Protocol implemented by local answer-generation providers."""

    def generate(self, *, model: str, prompt: str) -> str:
        """Generate a non-streaming answer from a local model."""
        ...


@dataclass(frozen=True)
class OllamaModelStatus:
    """Result of checking Ollama availability and local model presence."""

    running: bool
    available_models: tuple[str, ...]
    missing_models: tuple[str, ...]


def pull_commands_for_missing_models(models: Iterable[str]) -> list[str]:
    """Return exact shell commands users can run to install Ollama models."""

    return [f"ollama pull {model}" for model in models]


class OllamaClient:
    """Small client for Ollama's local HTTP API.

    The implementation deliberately avoids SDK dependencies so that the base
    RagRam install stays lightweight. It uses Ollama's documented `/api/tags`
    and `/api/generate` endpoints.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        session: Any | None = None,
        health_timeout: int = 5,
        generate_timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests
        self.health_timeout = health_timeout
        self.generate_timeout = generate_timeout

    def model_status(self, required_models: Iterable[str]) -> OllamaModelStatus:
        """Check whether Ollama is reachable and which required models are missing."""

        required = tuple(required_models)
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=self.health_timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return OllamaModelStatus(running=False, available_models=(), missing_models=required)

        available = tuple(
            str(model_name)
            for item in payload.get("models", [])
            if (model_name := item.get("name") or item.get("model"))
        )
        available_set = set(available)
        missing = tuple(model for model in required if model not in available_set)
        return OllamaModelStatus(running=True, available_models=available, missing_models=missing)

    def generate(self, *, model: str, prompt: str) -> str:
        """Generate a complete non-streaming answer with a local Ollama model."""

        response = self.session.post(
            f"{self.base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=self.generate_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("response", "")).strip()
