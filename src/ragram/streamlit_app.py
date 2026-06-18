"""Streamlit entrypoint for RagRam's local UI.

This module is intentionally thin: it loads local config, constructs local-only
providers, and renders question answering over the selected Telegram channel.
"""

from __future__ import annotations

from ragram.config import app_paths, load_config
from ragram.embeddings import SentenceTransformerEmbeddingProvider
from ragram.llm import ANSWER_MODEL_CHOICES, OllamaClient
from ragram.rag import RagService
from ragram.ui import DEFAULT_TOP_K, MAX_TOP_K, MIN_TOP_K, build_ui_state, source_label
from ragram.vector_store import ChromaVectorStore


def _provider_error_message(exc: Exception) -> str:
    return (
        "Local vector/embedding dependencies are not ready. "
        "Install local extras with `pip install -e '.[local]'`, then ensure the selected embedding model is available. "
        f"Details: {exc}"
    )


def main() -> None:  # pragma: no cover - rendered by Streamlit, unit-tested through helpers.
    import streamlit as st

    paths = app_paths()
    config = load_config(paths.config_path)
    state = build_ui_state(config)

    st.set_page_config(page_title="RagRam", page_icon="🦙", layout="wide")
    st.title("RagRam")
    st.caption("Local-first Telegram channel Q&A. No paid APIs are used.")

    with st.sidebar:
        st.header("Selected channel")
        st.write(f"**Title:** {state.channel_title}")
        st.write(f"**Username:** {state.channel_username or '—'}")
        st.write(f"**Entity ID:** {state.entity_id or '—'}")
        st.divider()
        answer_model = st.selectbox(
            "Answer model",
            ANSWER_MODEL_CHOICES,
            index=ANSWER_MODEL_CHOICES.index(state.answer_model)
            if state.answer_model in ANSWER_MODEL_CHOICES
            else 0,
        )
        top_k = st.slider("Retrieved chunks", min_value=MIN_TOP_K, max_value=MAX_TOP_K, value=DEFAULT_TOP_K)
        show_raw_context = st.checkbox("Show raw context", value=False)
        st.write(f"**Embedding:** {state.embedding_model}")
        st.write(f"**Summarization:** {state.summarization_model}")

    if state.entity_id is None:
        st.warning("No Telegram channel is selected yet. Run `ragram start` to configure one.")
        return

    question = st.text_area("Question", placeholder="Например: какие основные темы обсуждались в канале?", height=90)
    ask = st.button("Ask RagRam", type="primary", disabled=not question.strip())

    if not ask:
        st.info("Ask a question to retrieve local Telegram context and generate a grounded answer.")
        return

    status = st.status("Starting local RAG pipeline…", expanded=True)
    try:
        status.write("Loading local embedding model and opening Chroma index…")
        embedding_provider = SentenceTransformerEmbeddingProvider(state.embedding_model)
        vector_store = ChromaVectorStore(embedding_provider=embedding_provider, persist_directory=paths.chroma_dir)
        service = RagService(
            retriever=vector_store,
            llm=OllamaClient(),
            entity_id=state.entity_id,
            answer_model=answer_model,
        )
        status.write(f"Retrieving top {top_k} chunks from the local Telegram index…")
        grounded = service.answer_stream(question.strip(), top_k=top_k)
    except Exception as exc:
        status.update(label="Local RAG pipeline failed", state="error")
        st.error(_provider_error_message(exc))
        return

    sources = grounded.sources
    st.subheader("Answer")
    if sources:
        status.update(label="Streaming answer from local Ollama…", state="running", expanded=True)
        st.caption("Generating locally with Ollama… answer text appears as soon as tokens arrive.")
        st.write_stream(grounded.answer_chunks)
        status.update(label="Answer generated", state="complete", expanded=False)
    else:
        st.write("".join(grounded.answer_chunks))
        status.update(label="No relevant chunks found", state="complete", expanded=False)

    st.subheader("Sources")
    if not sources:
        st.write("No sources returned.")
    for index, source in enumerate(sources, start=1):
        with st.expander(f"{index}. {source_label(source)}", expanded=index <= 3):
            st.json(source)

    if show_raw_context and grounded.raw_context:
        st.subheader("Raw context")
        st.text(grounded.raw_context)


if __name__ == "__main__":  # pragma: no cover
    main()
