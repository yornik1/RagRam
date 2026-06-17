# RagRam Architecture Spec

## Final stack choice

- Language/package: Python 3.11+ with `src/` layout.
- CLI: Typer for commands plus InquirerPy for guided prompts.
- Terminal output/progress: Rich.
- Telegram: Telethon directly.
- Local UI: Streamlit.
- Raw storage: SQLite.
- Vector DB: Chroma persistent local storage.
- Embeddings: sentence-transformers through a small provider interface.
- Answer/summarization: Ollama local HTTP API through a small LLM interface.
- Config: TOML at `~/.ragram/config.toml`; no `.env`.

## UI choice

Use Streamlit for MVP. It is faster than building a FastAPI + HTML frontend, has built-in form controls, can be launched as a local process, and is enough for a single-user local RAG UI. FastAPI can be added later if RagRam needs a stable API or custom frontend.

Node is not chosen. It would add a second runtime and packaging path without clear MVP benefit.

## Core module boundaries

| File | Responsibility |
| --- | --- |
| `src/ragram/cli.py` | Typer commands, top-level command UX, prompt orchestration. |
| `src/ragram/app.py` | Application service that coordinates setup, ingestion, indexing, and UI launch. |
| `src/ragram/config.py` | App paths, TOML config model, load/save, defaults. |
| `src/ragram/telegram_client.py` | Telethon session/login, dialog listing, entity resolution, message iteration. |
| `src/ragram/storage.py` | SQLite schema and repositories for channels, messages, chunks, indexing runs. |
| `src/ragram/chunking.py` | Message-to-chunk grouping and token approximation. |
| `src/ragram/embeddings.py` | Embedding provider interface and sentence-transformers implementation. |
| `src/ragram/vector_store.py` | Chroma collection naming, upserts, retrieval. |
| `src/ragram/llm.py` | Ollama client, health/model checks, prompt completion. |
| `src/ragram/rag.py` | Retrieval + prompt assembly + answer/source result object. |
| `src/ragram/progress.py` | Rich progress helpers and rate/ETA formatting. |
| `src/ragram/models.py` | Shared dataclasses/enums typed across modules. |
| `src/ragram/ui.py` and `src/ragram/streamlit_app.py` | Streamlit UI entrypoint. |

The requested flat module layout is preserved for the MVP. If files become too large, split subpackages later without changing public interfaces.

## Runtime data flow

```text
ragram start
  -> config/bootstrap ~/.ragram
  -> Telegram login/session check
  -> dialog discovery and channel selection
  -> indexing scope/model prompts
  -> Telethon message iterator
  -> SQLite raw message upsert
  -> chunking from stored raw messages
  -> sentence-transformers embeddings
  -> Chroma upsert
  -> Streamlit UI launch
  -> RAG question: Chroma retrieval -> grounded prompt -> Ollama answer -> sources
```

## Storage layout

```text
~/.ragram/
  config.toml
  sessions/
    telegram.session
  data/
    ragram.sqlite
    chroma/
  logs/
```

## Config shape

```toml
[telegram]
api_id = 123456
api_hash = "stored-locally"
phone = "+10000000000"
session_name = "telegram"

[channel]
entity_id = 123456789
access_hash = "optional-if-needed"
title = "Example Channel"
username = "example"

[indexing]
scope_type = "last_n"
scope_value = "1000"
embedding_model = "ai-forever/ru-en-RoSBERTa"
answer_model = "qwen3:4b"
summarization_model = "qwen3:4b"

[ui]
port = 8501
```

`api_hash` is stored locally because Telethon needs it, but it must never be echoed after entry. The config file remains user-local, not committed.

## Data model

SQLite tables:

### `channels`

- `entity_id INTEGER PRIMARY KEY`
- `title TEXT NOT NULL`
- `username TEXT`
- `type TEXT`
- `last_message_date TEXT`
- `unread_count INTEGER`
- `raw_json TEXT`
- `updated_at TEXT NOT NULL`

### `messages`

- `entity_id INTEGER NOT NULL`
- `message_id INTEGER NOT NULL`
- `date TEXT NOT NULL`
- `sender_id TEXT`
- `sender_name TEXT`
- `reply_to_id INTEGER`
- `forward_info TEXT`
- `message_link TEXT`
- `text TEXT NOT NULL`
- `raw_json TEXT`
- `created_at TEXT NOT NULL`
- `updated_at TEXT NOT NULL`
- primary key: `(entity_id, message_id)`

### `index_runs`

- `id TEXT PRIMARY KEY`
- `entity_id INTEGER NOT NULL`
- `scope_type TEXT NOT NULL`
- `scope_value TEXT`
- `embedding_model TEXT NOT NULL`
- `started_at TEXT NOT NULL`
- `completed_at TEXT`
- `status TEXT NOT NULL`
- `last_message_id INTEGER`
- `error TEXT`

### `chunks`

- `chunk_id TEXT PRIMARY KEY`
- `entity_id INTEGER NOT NULL`
- `embedding_model TEXT NOT NULL`
- `message_id_start INTEGER NOT NULL`
- `message_id_end INTEGER NOT NULL`
- `date_start TEXT NOT NULL`
- `date_end TEXT NOT NULL`
- `text TEXT NOT NULL`
- `token_count INTEGER NOT NULL`
- `metadata_json TEXT NOT NULL`
- `created_at TEXT NOT NULL`

## Chroma collections

Collection names are deterministic:

```text
telegram_{entity_id}_{embedding_model_slug}
```

`embedding_model_slug` replaces non-alphanumeric characters with underscores and lowercases when safe. Changing the embedding model creates a different collection and allows re-indexing without destroying the previous index.

## Model choices

Embeddings:

1. Recommended default: `ai-forever/ru-en-RoSBERTa`.
2. Fallback: `BAAI/bge-m3`.
3. Lightweight fallback: `intfloat/multilingual-e5-small`.

Answer model:

1. Default: Ollama `qwen3:4b`.
2. Better local option: Ollama `qwen3:8b`.

Summarization model defaults to the selected answer model but is stored separately.

If Ollama is unavailable or a model is missing, print:

```bash
ollama pull qwen3:4b
ollama pull qwen3:8b
```

## Telegram login limitations

- Telegram API credentials (`api_id`, `api_hash`) must be created by the user at Telegram's developer portal; RagRam cannot generate them.
- Telegram login may require an SMS/app code and possibly a 2FA password.
- FloodWait exceptions must be respected by sleeping or stopping safely; bypassing rate limits is not allowed.
- Some private groups/channels may not expose message links or sender details.
- Access is limited to what the user's Telegram account can access.
- Session files are sensitive local credentials and should stay under `~/.ragram/sessions/`.

## Resumability

Raw messages are persisted before embedding. Indexing runs record progress. If ingestion or embedding is interrupted, the next run can skip existing `(entity_id, message_id)` rows and continue with missing chunks/index entries.

## Test strategy

- Unit tests for config paths, TOML roundtrip, chunking, collection names, prompt assembly.
- SQLite tests with temporary database files.
- Telethon tests use fakes/mocks, not real Telegram credentials.
- Ollama tests use fake HTTP/client responses.
- Chroma tests can use temporary persistent directories; heavier tests may be marked integration.
- CLI tests use Typer's test runner and monkeypatched prompt answers.

## Risks

| Risk | Mitigation |
| --- | --- |
| Heavy embedding models are slow or fail on user machine. | Keep model choices explicit; provide lightweight fallback; lazy-load models. |
| Telegram FloodWait interrupts indexing. | Catch FloodWait, show wait time, persist progress before sleeping/exit. |
| Telethon auth is hard to test. | Isolate client wrapper and test with fake client/session interfaces. |
| Streamlit launch can be environment-specific. | Print command/URL; support `--no-ui`; keep UI separate from indexing. |
| Chroma dependency/API changes. | Wrap Chroma in `vector_store.py` and test collection naming/upsert/retrieve through interface. |
| Local Ollama model missing. | Health check and exact pull commands before answering. |
