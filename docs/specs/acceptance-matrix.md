# RagRam Acceptance Matrix

| Requirement | Planned artifact/task | Verification |
| --- | --- | --- |
| `pip install -e .` works | G002 | Run install in repo and import package. |
| `ragram start` works | G002, G011 | CLI smoke test with faked integrations. |
| First run asks Telegram credentials interactively | G004 | CLI test with mocked prompts and Telethon fake. |
| `api_hash` is not echoed | G003, G004 | Prompt uses password/secret input path; tests assert prompt type. |
| Telegram session stored/reused | G004 | Fake session exists path skips login. |
| Accessible channels/groups listed by recency | G004 | Dialog fake sorted by last date. |
| Search/filter/custom entity input | G004 | Prompt selection tests. |
| Scope: last N/all/from year/from date | G006 | Scope parser tests. |
| Model choices for embedding/answer/summarization | G003, G009 | Config and prompt tests. |
| Ollama missing model commands shown | G009 | Fake Ollama unavailable test. |
| Progress bars and ETA shown | G006 | Progress helper unit test/smoke output. |
| Raw messages stored locally | G005, G006 | SQLite tests verify rows/metadata. |
| Dedupe by entity/message ID | G005 | Duplicate insert test. |
| Resumable ingestion | G005, G006 | Interrupted run simulation. |
| FloodWait handled safely | G004, G006 | Fake FloodWait test verifies wait/stop path. |
| Chunk target 400-900 tokens | G007 | Chunking tests over short/long messages. |
| Chroma persistent local index | G008 | Temp Chroma write/read integration test. |
| Collection name deterministic | G008 | Unit tests for slug/name. |
| Local UI starts and URL printed | G010, G011 | UI launch command test. |
| Russian question grounded in sources | G009, G010, G011 | Fake retrieval + fake Ollama Russian prompt smoke test. |
| Sources shown | G009, G010 | RAG result and UI rendering tests. |
| No paid APIs | All | Dependency/config review; no hosted LLM provider implementation. |
