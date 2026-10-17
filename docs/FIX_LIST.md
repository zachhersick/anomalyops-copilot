# anomalyops-copilot: fix list

Audited at commit `c91e7ad` (main) on 2026-10-17.

Baseline: 688 unit tests pass, `ruff check .` is clean. The code is well tested and defensive, so most of this list is about what the tests do not catch: behavior a visitor sees, failure modes under real load, and cleanup.

Items marked **Verified** were reproduced locally. The rest come from reading the code.

---

## Priority 1: things a visitor or reviewer will notice

### 1. Offline / Docker mode returns random chunks with a fake confidence score
**Verified.** The Docker image and the browser UI default to `manifest` retrieval with the `deterministic` provider. That retriever (`copilot/retrieval/embeddings.py`) turns text into a vector by SHA-256 hashing it, so similarity between a question and a chunk is noise. It never refuses.

I ran it against the real manifest:
- "Which SQLite tables store pipeline runs, predictions, and alerts?" returned README status notes, `tests/test_api.py`, and `tests/test_alert_events.py` at 0.60 "confidence".
- "What is the capital of France?" returned a JSON test fixture at **0.62 confidence**, higher than the real question, with no refusal.

**Why it matters:** `docker compose up` is the easiest way for anyone to try the project, and it is the mode that looks broken. A recruiter who asks the France question sees the copilot confidently answer with sensor readings.

**Fix:** replace the hash retriever for manifest mode with a real lexical one (BM25 or TF-IDF in pure Python, no API key needed), and refuse when no query terms match. If you keep the hash embeddings for unit tests, keep them out of the default runtime path. At minimum, label the mode in the UI as "offline demo, not semantic search".

### 2. Valid answers about code get rejected as a 502
**Verified.** `copilot/providers/openai_answers.py:111` finds citation markers with `re.findall(r"\[(\d+)\]", draft.answer)`. Any answer that mentions indexing, such as `features[0]` or `row[1]`, picks up a fake citation ID and fails the "markers do not match citation IDs" check. I confirmed `"The model reads `features[0]` first [1]."` raises `InvalidGroundedAnswerResponseError`.

**Why it matters:** the corpus is Python source, so answers that quote code are normal. Each one becomes "Grounded answer provider returned an invalid response."

**Fix:** only match markers that are not attached to an identifier or inside backticks (for example a negative lookbehind for `\w`, `]` and `` ` ``), or stop asking the model to write inline markers and render them yourself from `citation_ids`. Add a test with `x[0]` in the answer.

### 3. Request traces are never printed
**Verified.** `copilot/observability.py:136` emits traces with `logger.info` on the `anomalyops.trace` logger, and nothing in `copilot/` or `scripts/` configures a handler or level. Under `uvicorn copilot.api.app:app` I sent two requests and got zero trace lines; only uvicorn's access log appeared. The tests pass because pytest's `caplog` captures INFO on its own.

**Why it matters:** "correlated JSON traces" is a headline feature in the README, and in a real run it produces nothing.

**Fix:** configure the `anomalyops.trace` logger (StreamHandler, INFO, message-only formatter) inside `create_app`, or ship a uvicorn `--log-config` file and document it. Then check the output by running uvicorn, not only through tests.

---

## Priority 2: robustness under real use

### 4. Embedding and database failures on `/query` return a generic 500
`copilot/api/app.py` maps grounded-answer errors to 502 but does not catch `EmbeddingProviderError` (OpenAI down, rate limited, bad key) or SQLAlchemy `OperationalError` (Postgres not running). Both fall through to the middleware's `"Internal server error."` at `app.py:125`.

**Why it matters:** the most common failure in pgvector mode (forgot `docker compose up -d postgres`, or a rate limit) gives the least useful message, and it is inconsistent with the 502s you already return for answer failures.

**Fix:** catch `EmbeddingProviderError` and return 502, and catch `OperationalError` and return 503 with "Vector database unavailable." Add one test for each.

### 5. No timeout on OpenAI calls
`copilot/providers/factory.py:65, 108, 150` build `OpenAI(api_key=...)` with SDK defaults, which is a 10-minute timeout with 2 retries. The endpoints are sync, so each stalled call holds a worker thread, and `/triage` can make up to 8 model calls in one request.

**Why it matters:** one slow upstream response can hang a request for many minutes, and the UI just spins.

**Fix:** pass `timeout=30` (or similar) and an explicit `max_retries`, and build one shared client instead of three.

### 6. No upper bounds on request size
**Verified.** `copilot/schemas/query.py:15-16` has `top_k` with only `gt=0` and `query` with only `min_length=1`. `top_k=1000000000` and a 2 MB query both returned 200 in manifest mode. In OpenAI mode, `top_k` sets how many chunks go into the prompt and the query is embedded and sent as is.

**Why it matters:** cost and latency scale with input size, and a typo in a client can send the whole index to the model.

**Fix:** `top_k: int = Field(default=3, gt=0, le=20)` and `query: str = Field(min_length=1, max_length=2000)`. `TriageRequest.max_events` already does this (`le=20`), so copy that pattern.

### 7. Manifest mode re-reads and re-embeds the whole corpus on every request
`retrieve_chunks_for_query` (`copilot/api/query_service.py:112`) loads the JSON manifest from disk per request, and `copilot/retrieval/search.py` re-embeds all 133 chunks every time.

**Why it matters:** wasted work that grows linearly with the corpus. Do it together with item 1.

**Fix:** load and index the manifest once in `create_app` and keep it on `app.state`. This also lets a missing manifest fail at startup instead of on the first query.

### 8. Retrieval quality is the real ceiling on the RAG numbers
Your own eval (`evals/results/openai-pgvector-2026-08-06.json`) shows 93% Hit@5 but only 0.67 citation hit rate, 0.67 source recall@5, and 0.71 refusal precision, meaning 2 of 15 answerable questions were refused. Causes I can see in the code:
- `copilot/ingestion/chunking.py` cuts fixed 80-line windows, so functions and README sections are split mid-body.
- Only `chunk.content` is embedded; the file path is not, so "which file handles X" questions lose their best signal.
- Tests make up 58 of 133 chunks. The README's own sample retrieval for a schema question returns three test files and no source file.

**Why it matters:** this is the number a reviewer will quote, and these changes are cheap compared with model changes.

**Fix:** prepend `source_path` to the embedded text, chunk Python on `def`/`class` boundaries and Markdown on headings, and either tag tests as a separate `source_type` or down-rank them. Re-run the semantic eval after each change. Also grow the eval past 20 cases (about 50) before publishing percentages; with 20 cases, one case is 5 points.

### 9. The triage agent fails the whole request on one bad tool call
In `copilot/providers/openai_triage.py`, `_parse_tool_input` (line 185) and the `_execute_*` validators raise `InvalidTriageAgentResponseError` on any malformed tool argument, which ends the request with a 502.

**Why it matters:** models routinely send one bad argument and fix it when told. Right now a single slip costs the user the whole triage.

**Fix:** for tool-argument errors, return the validation message to the model as the tool result and let it retry within `max_tool_rounds`. Keep the strict validation on the final report as it is.

### 10. Reindexing sends every chunk in one embeddings request
`copilot/providers/openai_embeddings.py:90` sends `input=list(texts)` for the whole corpus. The OpenAI embeddings endpoint has per-request limits on input count and total tokens.

**Why it matters:** this works at 133 chunks and breaks once the corpus grows.

**Fix:** batch in groups of about 100 and concatenate the results, keeping your index-order checks.

---

## Priority 3: cleanup

### 11. Settings: hand-rolled env parsing, duplicated checks, and a fake knob
- `copilot/api/settings.py` parses every env var by hand, and lines 115-119 repeat the integer and positive checks that already ran above them (dead code).
- `pydantic-settings` is a declared dependency but `ApiSettings` is a plain `BaseModel`. Switching to `BaseSettings` with `env_prefix="ANOMALYOPS_"` removes about 90 lines.
- `ANOMALYOPS_EMBEDDING_DIMENSIONS` looks configurable, but the column is hardcoded to `VECTOR(256)` in `copilot/storage/models.py`. Any other value fails at query time. Either remove the setting or check it against the column at startup.

### 12. Dependencies: unused packages and no pins
- `numpy`, `pandas`, `tiktoken`, and `pydantic-settings` are runtime dependencies in `pyproject.toml` but are not imported anywhere in `copilot/` or `scripts/`. They make the Docker image bigger for no benefit.
- Nothing is pinned and there is no lockfile, so CI installs whatever is newest. A FastAPI or `openai` release can turn CI red with no change on your side (it already deprecated `on_event`, see item 13).

**Fix:** remove the unused packages, then add version bounds or a lockfile (`uv lock` or `pip-tools`).

### 13. Replace the deprecated shutdown hook
`copilot/api/app.py:284` uses `@app.on_event("shutdown")`, which produces most of the 122 deprecation warnings in the test run. Move it to a `lifespan` handler.

### 14. Small items
- CI tests only Python 3.11 while `pyproject.toml` allows `>=3.11`. I ran the suite on 3.13 and it passes, so adding a 3.12/3.13 matrix costs nothing.
- Typo in a test filename: `tests/test_determinisitc_triage_agent.py`.
- The Dockerfile runs the app as root. Add a non-root `USER`.
- `docker-compose.yml` publishes Postgres on all interfaces with password `anomalyops`. This is fine on a laptop; bind it to `127.0.0.1:5432:5432` anyway so it stays that way.

---

## Suggested order
1. Items 2, 3, 6: each is a few lines, and two of them are real bugs.
2. Item 1 together with 7: the biggest visible improvement.
3. Items 4, 5, 9: error handling, about one session together.
4. Item 8, then re-run the eval and update the README numbers.
5. Items 10-14 whenever.
