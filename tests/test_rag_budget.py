from __future__ import annotations

from ragram.rag import build_grounded_prompt
from ragram.vector_store import RetrievalResult


def test_grounded_prompt_limits_context_size():
    contexts = [
        RetrievalResult(chunk_id=f"c{i}", text="word " * 1000, metadata={"message_id_start": i, "message_id_end": i})
        for i in range(5)
    ]

    prompt = build_grounded_prompt(question="What?", contexts=contexts, top_k=5, max_context_tokens=120)

    assert len(prompt.split()) < 300
    assert "[context truncated to fit local model budget]" in prompt


def test_model_context_budget_is_model_specific():
    from ragram.rag import context_budget_for_model

    assert context_budget_for_model("qwen3:8b") > context_budget_for_model("qwen3:4b")
    assert context_budget_for_model("unknown-local") == context_budget_for_model("qwen3:4b")
