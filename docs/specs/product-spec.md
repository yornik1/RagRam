# RagRam MVP Product Spec

## Goal

RagRam is a local-first macOS-oriented app that lets a user connect a Telegram user account, select one accessible Telegram channel or group, index text messages locally, and ask grounded questions over that channel.

The MVP should feel like a small product, not a raw script: one main command (`ragram start`), guided interactive setup, visible progress, local UI, local storage, and no paid or hosted LLM APIs.

## Non-negotiable constraints

- No `.env` file requirement.
- No paid APIs.
- No OpenAI, Anthropic, or other hosted answer APIs in MVP.
- Free/local model providers only.
- Python-first implementation.
- macOS-first UX.
- One Telegram channel/group per MVP run/config.
- Text-only Telegram messages in MVP.
- CLI should ask questions step by step; flags stay minimal.

## MVP commands

Required:

```bash
ragram start
ragram restart
ragram status
```

Optional MVP flags:

```bash
ragram start --no-ui
ragram start --ui-port 8501
```

## First-run UX

`ragram start` performs this sequence:

1. Create the local app layout if missing.
2. Load `~/.ragram/config.toml` if present.
3. If Telegram config/session is missing:
   - explain Telegram MTProto requirements: `api_id`, `api_hash`, and phone number;
   - ask for `api_id`;
   - ask for `api_hash` without echoing it;
   - ask for phone number;
   - run Telethon login;
   - ask for login code when needed;
   - ask for 2FA password without echoing it when needed;
   - store config and reuse the Telethon session on later runs.
4. List accessible Telegram channels/groups sorted by recent activity.
5. Allow filtering/searching by title text.
6. Allow custom entity input: username, URL, exact title, or entity ID.
7. Ask indexing scope:
   - last N messages, default 1000;
   - all messages;
   - from year;
   - from exact date.
8. Ask model choices:
   - embedding model;
   - answer model;
   - summarization model, default same as answer model.
9. Check Ollama health/model availability and print exact pull commands if missing.
10. Fetch messages, store raw messages, chunk, embed, write Chroma index, optionally summarize.
11. Start local UI unless `--no-ui` is set.
12. Print the URL clearly, for example `Open RagRam: http://localhost:8501`.

## Restart/reset UX

`ragram restart` opens a guided reconfiguration flow. It can reconfigure:

- Telegram account/session;
- selected channel;
- indexing scope;
- embedding model;
- answer model;
- summarization model;
- raw data/index clearing.

Destructive actions require explicit confirmation. The default path should keep existing local data.

## Status UX

`ragram status` prints:

- app directory paths;
- whether config exists;
- whether Telegram session exists;
- selected channel if configured;
- raw message count;
- chunk/index count if available;
- configured embedding and answer models;
- Ollama availability and configured model availability;
- UI hint command.

## Channel listing requirements

For each accessible channel/group, show when available:

- title;
- username;
- type: channel, megagroup, group;
- last message date;
- unread count;
- entity ID.

Sort by last message date descending. If last activity is missing, place lower in the list and label date as unknown.

## Message metadata

Persist raw text messages before embedding with this metadata:

- channel title;
- channel username;
- entity ID;
- message ID;
- date;
- sender ID/name if available;
- reply-to message ID if available;
- forward info if available;
- message link if possible;
- text.

Deduplicate by `(entity_id, message_id)`.

## Chunking requirements

Do not embed every tiny Telegram message alone. Group short nearby messages into chunks while preserving source traceability.

Rules:

- target chunk size: 400-900 approximate tokens;
- long channel posts may remain individual chunks if they fit the target range or are semantically self-contained;
- chunks preserve message ID range and date range;
- chunk IDs are deterministic from entity ID, embedding model slug, and message range/content hash;
- Russian and English content are both supported.

## Ask behavior

The UI retrieves top K chunks, default 8, then builds a grounded prompt:

- answer only from retrieved context;
- if context is insufficient, say so;
- answer in the language of the user question;
- preserve uncertainty;
- show sources under the answer.

## UI requirements

Use a local web UI with:

- selected channel info;
- question input;
- answer model selector if feasible;
- top K selector;
- answer area;
- retrieved sources under the answer;
- source dates and message IDs;
- option to show raw context.

## MVP skipped features

Explicitly out of scope for the first implementation:

- multiple channels/groups in one index;
- media OCR/transcription;
- attachments and files;
- hosted LLM providers;
- cloud sync;
- multi-user server deployment;
- advanced Telegram admin features;
- scheduled background sync daemon;
- authentication for the local UI;
- sophisticated reranking;
- message deletion synchronization.
