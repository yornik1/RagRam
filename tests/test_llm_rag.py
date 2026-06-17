from __future__ import annotations

from ragram.llm import (
    DEFAULT_ANSWER_MODEL,
    DEFAULT_BETTER_ANSWER_MODEL,
    OllamaClient,
    OllamaModelStatus,
    pull_commands_for_missing_models,
)
from ragram.rag import GroundedAnswer, RagService, build_grounded_prompt, format_sources
from ragram.vector_store import RetrievalResult


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.get_payload = {"models": []}
        self.post_payload = {"response": "Ответ", "eval_count": 20, "eval_duration": 2_000_000_000}
        self.get_calls = []
        self.post_calls = []

    def get(self, url, timeout):
        self.get_calls.append((url, timeout))
        return FakeResponse(self.get_payload)

    def post(self, url, json, timeout):
        self.post_calls.append((url, json, timeout))
        return FakeResponse(self.post_payload)


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def query(self, *, entity_id, query, top_k):
        self.calls.append({"entity_id": entity_id, "query": query, "top_k": top_k})
        return self.results


class FakeLLM:
    def __init__(self, answer="Согласно источникам, проект локальный."):
        self.answer = answer
        self.prompts = []

    def generate(self, *, model, prompt):
        self.prompts.append({"model": model, "prompt": prompt})
        return self.answer


def retrieval(chunk_id="c1", text="RagRam хранит данные локально."):
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        metadata={
            "message_id_start": 10,
            "message_id_end": 12,
            "date_start": "2025-01-01T00:00:00+00:00",
            "date_end": "2025-01-02T00:00:00+00:00",
        },
        distance=0.2,
    )


def test_ollama_defaults_and_pull_commands():
    assert DEFAULT_ANSWER_MODEL == "qwen3:4b"
    assert DEFAULT_BETTER_ANSWER_MODEL == "qwen3:8b"
    assert pull_commands_for_missing_models(["qwen3:4b", "qwen3:8b"]) == [
        "ollama pull qwen3:4b",
        "ollama pull qwen3:8b",
    ]


def test_ollama_model_status_detects_running_and_missing_models():
    session = FakeSession()
    session.get_payload = {"models": [{"name": "qwen3:4b"}, {"model": "other:latest"}]}
    client = OllamaClient(base_url="http://localhost:11434", session=session)

    status = client.model_status(["qwen3:4b", "qwen3:8b"])

    assert status == OllamaModelStatus(running=True, available_models=("qwen3:4b", "other:latest"), missing_models=("qwen3:8b",))
    assert session.get_calls == [("http://localhost:11434/api/tags", 5)]


def test_ollama_generate_uses_non_streaming_api_and_returns_text():
    session = FakeSession()
    client = OllamaClient(base_url="http://localhost:11434", session=session)

    answer = client.generate(model="qwen3:4b", prompt="Answer from context")

    assert answer == "Ответ"
    assert session.post_calls == [
        (
            "http://localhost:11434/api/generate",
            {"model": "qwen3:4b", "prompt": "Answer from context", "stream": False},
            120,
        )
    ]


def test_build_grounded_prompt_requires_context_language_and_uncertainty():
    prompt = build_grounded_prompt(question="Что такое RagRam?", contexts=[retrieval()], top_k=8)

    assert "answer only from the retrieved context" in prompt
    assert "If the context is insufficient" in prompt
    assert "Answer in the language of the question" in prompt
    assert "Что такое RagRam?" in prompt
    assert "RagRam хранит данные локально" in prompt
    assert "Source c1" in prompt


def test_format_sources_preserves_dates_and_message_ids():
    assert format_sources([retrieval()]) == [
        {
            "chunk_id": "c1",
            "message_id_start": 10,
            "message_id_end": 12,
            "date_start": "2025-01-01T00:00:00+00:00",
            "date_end": "2025-01-02T00:00:00+00:00",
            "distance": 0.2,
        }
    ]


def test_rag_service_retrieves_top_k_and_returns_grounded_answer_with_sources():
    retriever = FakeRetriever([retrieval()])
    llm = FakeLLM()
    service = RagService(retriever=retriever, llm=llm, entity_id=100, answer_model="qwen3:4b")

    answer = service.answer("Почему это локально?", top_k=3)

    assert isinstance(answer, GroundedAnswer)
    assert answer.answer == "Согласно источникам, проект локальный."
    assert answer.sources[0]["message_id_start"] == 10
    assert retriever.calls == [{"entity_id": 100, "query": "Почему это локально?", "top_k": 3}]
    assert llm.prompts[0]["model"] == "qwen3:4b"
    assert "Почему это локально?" in llm.prompts[0]["prompt"]


def test_rag_service_short_circuits_when_no_context():
    service = RagService(retriever=FakeRetriever([]), llm=FakeLLM(), entity_id=100, answer_model="qwen3:4b")

    answer = service.answer("Что случилось?", top_k=8)

    assert answer.answer == "Недостаточно контекста в найденных сообщениях, чтобы ответить на вопрос."
    assert answer.sources == []


def test_rag_service_no_context_uses_english_for_english_question():
    service = RagService(retriever=FakeRetriever([]), llm=FakeLLM(), entity_id=100, answer_model="qwen3:4b")

    answer = service.answer("What happened?", top_k=8)

    assert answer.answer == "The retrieved messages do not contain enough context to answer the question."
