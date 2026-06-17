# ADR 0001: RagRam MVP Stack

## Status

Accepted for MVP planning.

## Decision

Build RagRam as a Python package using Typer, InquirerPy, Rich, Telethon, SQLite, Chroma, sentence-transformers, Ollama, and Streamlit.

## Rationale

Python keeps the CLI, ingestion, embeddings, storage, and local UI in one runtime. Streamlit is the fastest path to a usable local product UI with selectors, text inputs, and expandable source context. Telethon is the direct MTProto client requested by the brief. SQLite and Chroma provide local-first persistence without a server. Ollama and sentence-transformers keep the MVP free/local.

## Consequences

- Users install with `pip install -e .` for development.
- First startup may download embedding models through sentence-transformers if not cached.
- Ollama must be installed separately and run locally.
- UI customization is limited by Streamlit but enough for MVP.
- Later productization can replace Streamlit with FastAPI + frontend without changing storage/RAG boundaries.
