# RagRam Implementation Plan

## Plan mode

This plan decomposes the MVP into simple, sequential Ultragoal stories. Each story should be independently testable. Cheap/fast agents should only take bounded docs, scaffold, pure-function, or smoke-test tasks; risky integration tasks stay with stronger owner/reviewer lanes.

## Delegation policy

- Cheap/fast candidates: docs, package scaffold, config path helpers, CLI help/status wiring, pure chunking, prompt formatting tests, smoke scripts.
- Stronger-agent candidates: Telethon auth/session, resumable ingestion, Chroma integration, Ollama integration, UI orchestration, final e2e debugging.
- Do not let parallel agents edit the same files in the same round.
- Use fakes/mocks for Telegram and Ollama in automated tests; never require real credentials in CI/local verification.

## Story breakdown

### G001 Planning docs and product spec

**Status:** current story.  
**Files:**

- `docs/specs/product-spec.md`
- `docs/specs/architecture.md`
- `docs/specs/acceptance-matrix.md`
- `docs/adr/0001-mvp-stack.md`
- `docs/plans/implementation-plan.md`
- `docs/superpowers/specs/2026-06-17-ragram-mvp-design.md`
- `docs/superpowers/plans/2026-06-17-ragram-mvp-implementation.md`

**Validation:** docs exist, contain no placeholder markers, and every brief acceptance item maps to a planned story.

**Cheap agent:** yes, docs/planner review only.

### G002 Python package scaffold

**Files:**

- Create `pyproject.toml`
- Create `src/ragram/__init__.py`
- Create `src/ragram/cli.py`
- Create `src/ragram/models.py`
- Create empty MVP module files requested by brief
- Create `tests/test_cli_import.py`

**Steps:**

1. Add package metadata and dependencies in `pyproject.toml` with a console script `ragram = "ragram.cli:app"`.
2. Add minimal Typer app with `start`, `restart`, and `status` commands that can print placeholder planned behavior without contacting Telegram.
3. Add import/smoke tests.
4. Run `python -m pip install -e .` and `pytest`.

**Validation:** install succeeds; `ragram --help` and `ragram status` work.

**Cheap agent:** yes.

### G003 Local config and app directories

**Files:** `src/ragram/config.py`, `src/ragram/models.py`, `tests/test_config.py`.

**Steps:**

1. Implement app path resolution with home override for tests.
2. Implement idempotent creation of `~/.ragram`, `sessions`, `data`, `data/chroma`, `logs`, and SQLite parent path.
3. Implement TOML config load/save with defaults and no `.env` dependency.
4. Add status object for config/session/index paths.

**Validation:** temp-home tests prove layout and TOML roundtrip.

**Cheap agent:** yes.

### G004 Telegram auth and channel selection

**Files:** `src/ragram/telegram_client.py`, `src/ragram/cli.py`, `tests/test_telegram_client.py`.

**Steps:**

1. Wrap Telethon client creation around configured `api_id`, `api_hash`, and session path.
2. Implement interactive API credential collection with hidden `api_hash`.
3. Implement login method choice with QR login recommended first, Telegram app code fallback, and hidden 2FA password.
4. Implement dialog listing with title, username, type, last message date, unread count, entity ID.
5. Implement search/filter/custom entity selection.
6. Add faked Telethon tests for sorting, filtering, session reuse, and auth branches.

**Validation:** tests do not contact Telegram; manual real login remains a later smoke step.

**Cheap agent:** no.

### G005 SQLite raw message storage

**Files:** `src/ragram/storage.py`, `tests/test_storage.py`.

**Steps:**

1. Create SQLite schema and migrations/bootstrap.
2. Add channel upsert.
3. Add message upsert with `(entity_id, message_id)` primary key.
4. Add indexing run record/update methods.
5. Add chunk record methods.
6. Add repository queries for selected channel, raw message counts, and messages by scope.

**Validation:** temp SQLite tests for metadata, dedupe, and indexing-run updates.

**Cheap agent:** no for core implementation; yes for adding extra repository tests after interface stabilizes.

### G006 Message fetching and progress

**Files:** `src/ragram/app.py`, `src/ragram/progress.py`, `src/ragram/telegram_client.py`, `tests/test_ingestion.py`, `tests/test_progress.py`.

**Steps:**

1. Represent indexing scopes as typed values: `last_n`, `all`, `from_year`, `from_date`.
2. Convert scope to Telethon iteration parameters.
3. Persist each raw message before downstream work.
4. Show Rich progress for fetching/saving with rate and ETA where total is known.
5. Catch FloodWait and persist progress before waiting or exiting safely.

**Validation:** fake iterator tests for scope, resume, and FloodWait behavior.

**Cheap agent:** mixed; progress helper only is cheap.

### G007 Chunking pipeline

**Files:** `src/ragram/chunking.py`, `tests/test_chunking.py`.

**Steps:**

1. Implement approximate token counter suitable for Russian/English MVP text.
2. Keep long messages as standalone chunks when suitable.
3. Group short nearby messages until target range or boundary condition.
4. Preserve message ID/date ranges and metadata.
5. Generate deterministic chunk IDs.

**Validation:** pure unit tests with short chats, long posts, Russian text, and deterministic IDs.

**Cheap agent:** yes.

### G008 Embeddings and Chroma index

**Files:** `src/ragram/embeddings.py`, `src/ragram/vector_store.py`, `tests/test_embeddings.py`, `tests/test_vector_store.py`.

**Steps:**

1. Define embedding provider protocol.
2. Implement sentence-transformers provider with model choice defaults.
3. Add slug function for embedding model names.
4. Implement Chroma persistent collection creation/upsert/query.
5. Support fake embedding provider for tests.

**Validation:** unit tests for slug/provider interface; temp Chroma test where feasible.

**Cheap agent:** mixed; slug/interface tests cheap, Chroma integration stronger.

### G009 Ollama LLM and grounded RAG

**Files:** `src/ragram/llm.py`, `src/ragram/rag.py`, `tests/test_llm.py`, `tests/test_rag.py`.

**Steps:**

1. Implement Ollama health and model-list checks.
2. Print exact pull commands for `qwen3:4b` and `qwen3:8b` when missing.
3. Build grounded prompt from retrieved chunks.
4. Return answer plus structured sources.
5. Preserve insufficient-context behavior and language-of-question instruction.

**Validation:** fake Ollama and fake retriever tests, including Russian question.

**Cheap agent:** prompt tests only.

### G010 Local UI

**Files:** `src/ragram/ui.py` and `src/ragram/streamlit_app.py`, `src/ragram/app.py`, `tests/test_ui_launch.py`.

**Steps:**

1. Implement Streamlit page reading local config/storage/index.
2. Add channel info panel.
3. Add question input, model selector, top K selector.
4. Render answer and sources.
5. Add raw context expander.
6. Implement UI launch command that prints URL.

**Validation:** launch command unit test; manual local UI smoke after integration.

**Cheap agent:** no for full UI; yes for static docs/screensmoke checklist.

### G011 CLI orchestration, restart/status, final QA

**Files:** `src/ragram/cli.py`, `src/ragram/app.py`, `README.md`, `tests/e2e/test_smoke.py`.

**Steps:**

1. Wire `ragram start` end-to-end with fakes available for tests.
2. Wire `--no-ui` and `--ui-port`.
3. Implement `ragram restart` guided reconfiguration and destructive clear confirmations.
4. Implement `ragram status` using real config/storage state.
5. Update README quickstart and local model prerequisites.
6. Run install, unit tests, smoke tests, cleanup pass, and final code review gate.

**Validation:** `pip install -e .`, `pytest`, CLI smoke, and final review evidence.

**Cheap agent:** smoke-test script/docs portions only.

## Implementation status

As of 2026-06-17, G001-G010 are implemented and checkpointed. G011 wires the interactive CLI orchestration, restart/reset behavior, README updates, final verification, cleanup, and independent review gate. Automated tests use fakes for Telegram/Ollama/Chroma; real Telegram login remains a manual local smoke because it depends on user credentials and Telegram-delivered codes.

## Stop condition

The MVP is complete when all G001-G011 stories are checkpointed, `ragram start` can run through faked automated smoke paths, real-user setup instructions are documented, no paid API path exists, and the final Ultragoal quality gate passes.
