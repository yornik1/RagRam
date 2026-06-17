# RagRam

RagRam is a local-first MVP for asking grounded questions over **one Telegram channel/group** using local storage, local embeddings, Chroma, and Ollama. It is a small Python product scaffold, not a raw script.

## Principles

- Main command: `ragram start`.
- No `.env` required.
- No paid APIs and no OpenAI/Anthropic hosted APIs in the MVP.
- macOS/Python-first.
- Telegram access uses your Telegram user account via Telethon/MTProto.
- Data stays under `~/.ragram/` unless `RAGRAM_HOME` is set.

## Install

Use Python 3.11-3.13 for the full local ML/UI stack.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development tests:

```bash
pip install -e '.[dev]'
pytest
```

The base install includes the local MVP runtime (`sentence-transformers`, `chromadb`, `streamlit`). It still does not download embedding or Ollama model weights until the first real indexing/model use.

## Local model prerequisites

Install and run Ollama, then pull one or both MVP answer models:

```bash
ollama pull qwen3:4b
ollama pull qwen3:8b
```

Default models:

- Embeddings: `ai-forever/ru-en-RoSBERTa`
- Embedding fallbacks: `BAAI/bge-m3`, `intfloat/multilingual-e5-small`
- Answer model: Ollama `qwen3:4b`
- Better answer model: Ollama `qwen3:8b`
- Summarization model: defaults to answer model

## Telegram API credentials

RagRam uses Telegram MTProto through your Telegram user account. Telegram requires an app `api_id` and `api_hash`.

Get them from Telegram's app management page:

```text
https://my.telegram.org/apps
```

Steps:

1. Open `https://my.telegram.org/apps`.
2. Log in with your Telegram phone number.
3. Telegram sends the confirmation code inside Telegram, not by SMS.
4. Create an app if you do not already have one.
5. Copy `App api_id` into the `Telegram api_id` prompt.
6. Copy `App api_hash` into the hidden `Telegram api_hash` prompt.

RagRam stores these credentials locally in `~/.ragram/config.toml`; it does not use `.env` or paid APIs.

## Run

```bash
ragram start
```

Interactive `start` will:

1. Create local folders and config under `~/.ragram/`.
2. Ask for Telegram `api_id`, hidden `api_hash`, and phone number if missing.
3. Login through Telethon, including login code and 2FA password when Telegram requires it.
4. List accessible channels/groups by recent activity and allow custom username/URL/title/entity id input.
5. Ask indexing scope: last N (default 1000), all, from year, or from exact date.
6. Ask embedding, answer, and summarization model choices.
7. Store raw messages in SQLite before embedding.
8. Chunk, embed, and write vectors to Chroma.
9. Launch Streamlit and print a local URL, for example:

```text
Open RagRam: http://localhost:8501
```

Useful variants:

```bash
ragram start --no-ui
ragram start --ui-port 8601
ragram status
ragram restart
ragram restart --clear-data
```

`restart --clear-data` asks confirmation before deleting local raw SQLite data and Chroma indexes.

## Troubleshooting

- On macOS, use `python3`, not `python`, if `python` is not installed.
- Run `ragram start` from a real interactive terminal. The first run needs secure prompts for Telegram credentials and login code.
- The Telegram login code is normally sent inside Telegram to an already logged-in app/session, often as a service chat or login notification. It is not shown in RagRam and may not arrive as SMS.
- At the `Telegram login code` prompt, type `r` to request a new code or `q` to quit and re-check the saved phone/API credentials.
- Telegram may refuse immediate resends after the first accepted request because all delivery options for that phone number were already used. In that case, wait a few minutes, check all logged-in Telegram apps/sessions, then run `ragram start` again.
- If setup was interrupted, retry with:

```bash
ragram restart --reconfigure
ragram start
```

## Local storage layout

```text
~/.ragram/
  config.toml
  sessions/
  data/
    ragram.sqlite
    chroma/
  logs/
```

## Current MVP limitations

- Text-only Telegram messages.
- One selected channel/group at a time.
- Local-only models/providers.
- Automated tests use fakes for Telegram/Ollama/Chroma; real Telegram login is a manual local smoke path because it requires user credentials and Telegram delivery of a login code.

## Planning docs

See `docs/README.md` for the product spec, architecture, acceptance matrix, ADR, and implementation plan.
