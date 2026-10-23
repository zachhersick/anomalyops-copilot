# 6. Development and operations

[Handbook index](README.md) · Previous: [Upstream platform](05-upstream-platform.md) · Next: [Findings](07-review-findings.md)

## Configuration and lifecycle

[`ApiSettings`](../../copilot/api/settings.py) is a Pydantic `BaseModel`, not `BaseSettings`. `load_api_settings()` manually reads the process environment. Creating `ApiSettings()` directly uses model defaults; it does not read `.env`. The module-level FastAPI app calls `load_api_settings()` at import time. Pass explicit settings to `create_app` in tests and restart the application after changing environment configuration.

| Variable | Default / use |
| --- | --- |
| `ANOMALYOPS_RETRIEVAL_BACKEND` | `manifest`; alternative `pgvector` |
| `ANOMALYOPS_MANIFEST_PATH` | Unset; required for manifest queries |
| `ANOMALYOPS_DATABASE_URL` | Unset; required for pgvector retrieval/indexing |
| `ANOMALYOPS_AI_PROVIDER` | `deterministic`; alternative `openai`; selects embedding, answer and triage providers |
| `ANOMALYOPS_EMBEDDING_MODEL` | Unset; required for OpenAI embedding provider |
| `ANOMALYOPS_EMBEDDING_DIMENSIONS` | 256; must also match the storage column for pgvector |
| `ANOMALYOPS_GROUNDED_ANSWER_MODEL` | Unset; required for OpenAI answer generation |
| `ANOMALYOPS_TRIAGE_MODEL` | Unset; required for OpenAI triage |
| `ANOMALYOPS_ANOMALY_API_BASE_URL` | Unset; required for operational triage |
| `OPENAI_API_KEY` | Unset; required when OpenAI providers are constructed |
| `ANOMALYOPS_TEST_DATABASE_URL` | Test fixture only; must point to a disposable database |

Most configuration-dependent scripts call `load_dotenv()` without overriding existing environment values. Thus exported shell variables win over `.env`, then application defaults fill missing values. Uvicorn reads a file only when launched with `--env-file`. Compose's `.env` interpolation does not automatically pass all values into a container; the current API service explicitly supplies manifest/deterministic settings.

At application creation, pgvector mode with a database URL creates an engine, session factory, and embedding provider. A configured anomaly URL creates an HTTP client, tools and triage agent. The answer generator is always constructed. Engines connect lazily, so `/health` can succeed while the database is unavailable. The shutdown hook closes the anomaly client and disposes the database engine; migration to FastAPI lifespan handling is a recorded maintenance item.

## Test strategy

```bash
python -m pytest -m 'not integration' -q
ruff check .
git diff --check
```

The ordinary suite uses temporary files, mocks, and injected transports. It checks OpenAI response handling without paid model calls. `testpaths = ["tests"]` excludes the bundled upstream tests, and Ruff explicitly excludes `data_sources`. A green Copilot suite therefore does not validate the copied ML system.

| Change area | Useful starting tests |
| --- | --- |
| Source ingestion / manifests | `test_source_discovery.py`, `test_chunking.py`, `test_ingestion_pipeline.py`, `test_manifest.py` |
| Retrieval / storage | `test_search.py`, `test_pgvector_retrieval.py`, `test_chunk_storage.py`, `test_reindex_embeddings.py` |
| Grounded answers | `test_grounded_answer_builder.py`, `test_openai_answer_provider.py`, `test_query_service.py`, `test_query_api.py` |
| Triage / upstream client | `test_anomaly_api_client.py`, `test_anomaly_tools.py`, `test_triage_service.py`, `test_openai_triage_agent.py`, `test_triage_api.py` |
| Browser / packaging / traces | `test_frontend.py`, `test_docker_packaging.py`, `test_observability.py`, `test_observability_tracing.py` |
| Evaluation changes | `test_rag_eval_runner.py`, `test_triage_eval_runner.py`, `test_run_rag_evals.py`, `test_run_triage_evals.py` |

All names above live in [`tests/`](../../tests/); the [file inventory](02-architecture.md#tracked-file-inventory) links individual files. The frontend tests exercise served markup and source-level behavior; they are not a full browser interaction test suite.

### Database integration

[`tests/integration/test_pgvector_query_flow.py`](../../tests/integration/test_pgvector_query_flow.py) contains three integration tests. They rebuild `source_chunks`, so **never point them at a database containing an index you want to keep**. They use deterministic embeddings to test real SQL storage/retrieval and API/evaluation contracts without OpenAI calls.

When Docker is running, this creates a separate disposable instance on a separate loopback port:

```bash
docker run --rm -d --name anomalyops-handbook-test-db \
  -e POSTGRES_USER=anomalyops -e POSTGRES_PASSWORD=anomalyops \
  -e POSTGRES_DB=anomalyops_test \
  -p 127.0.0.1:55432:5432 pgvector/pgvector:pg16
docker exec anomalyops-handbook-test-db pg_isready -U anomalyops -d anomalyops_test
# Wait until pg_isready reports accepting connections before running pytest.
ANOMALYOPS_TEST_DATABASE_URL=postgresql+psycopg://anomalyops:anomalyops@localhost:55432/anomalyops_test \
  python -m pytest -m integration -q
docker stop anomalyops-handbook-test-db
```

No volume is attached; stopping removes this test container and its data. These Docker commands were not executed during this review because the daemon was stopped. The suite's skip reasons are recorded in chapter 7.

## Evaluation is different from unit testing

```bash
python scripts/run_retrieval_evals.py outputs/chunks.json evals/retrieval_cases.json
python scripts/run_rag_evals.py outputs/chunks.json evals/rag_cases.json --mode offline
# A failed evaluation case makes this command return exit code 1.
python scripts/run_rag_evals.py outputs/chunks.json evals/rag_cases.json --mode offline --strict
```

Offline evaluation forces manifest retrieval and deterministic generation. The reviewed baseline passes 3 of 10 cases; strict mode should fail. Do not “fix” the fixtures just to obtain a green score. Unit tests verify the evaluator and contracts, while evaluation cases measure behavior on selected questions.

For the paid semantic path, first configure and reindex a development database as in chapter 1, then run:

```bash
python scripts/run_rag_evals.py outputs/chunks.json evals/semantic_rag_cases.json \
  --mode semantic --output outputs/semantic-review.json
```

Semantic mode requires OpenAI plus pgvector and never reindexes automatically. Its preflight checks exact chunk-ID membership, row count and embedding configuration. It does not compare stored text or vector content against the manifest. The result snapshot records a manifest hash, model/provider metadata, counts and duration, which supports provenance but does not repair that preflight limitation.

| Metric | Interpretation |
| --- | --- |
| Retrieval hit / Hit@k | Whether at least one expected source appears among retrieved chunks, at the specified cutoff |
| MRR@5 | Reciprocal rank of the first relevant source within five results, averaged over supported cases |
| Source recall@5 | Fraction of expected sources covered within five results, averaged over supported cases |
| Citation validity | Structural/source-range checks against supplied context |
| Citation hit | Whether an answer cites an expected source |
| Refusal accuracy / precision / recall | Agreement with answerability labels / proportion of refusals justified / proportion of expected refusals made |
| Answer-term accuracy | Whether required normalized answer terms appear; not a general semantic correctness judge |
| Pass rate | Fraction satisfying the evaluator's applicable case checks |

The [2026-08-06 committed snapshot](../../evals/results/openai-pgvector-2026-08-06.json) records 20 cases, 93.3% Hit@5, 0.733 MRR@5, and 75% pass rate. Those are historical results for that corpus/configuration; no live semantic evaluation was run for this handbook.

`python scripts/run_triage_evals.py evals/triage_cases.json --strict` requires a running anomaly API and uses the configured provider. With OpenAI selected it incurs model calls. Triage checks schema, evidence, run consistency, count limits and status semantics. The two committed cases have no expected status/findings/count labels, so those accuracy fields are `N/A`, not proof of correct incident prioritization.

## CI and Docker

The [CI workflow](../../.github/workflows/ci.yml) runs on pushes and pull requests with Python 3.11 and a pgvector/Postgres 16 service. It installs editable dev dependencies, runs Ruff, then runs ordinary and integration tests separately. Its test OpenAI key is a dummy; provider tests mock network behavior. No paid semantic evaluation or upstream ML training is included.

The [Dockerfile](../../Dockerfile) installs Copilot and builds the manifest from the bundled corpus at image build time. The API image defaults to offline mode. [Compose](../../docker-compose.yml) starts API and Postgres services, but the default API does not query that Postgres service. To use `.env` OpenAI/pgvector configuration, the documented path is Postgres in Docker and Uvicorn on the host.

Useful commands are `docker compose config --quiet`, `docker compose build api`, `docker compose up -d`, `docker compose ps`, and `docker compose down`. Ordinary `down` leaves the named database volume intact. Source changes require an image rebuild because there is no development source bind mount. The container health check probes only `/health`.

## Tracing and troubleshooting

[`observability.py`](../../copilot/observability.py) defines `http.request`, `query.retrieval`, `query.answer`, `provider.request`, `triage.agent`, and `triage.tool` trace events. Metadata includes request ID, elapsed milliseconds, status/error type, provider/model, and counts. Sensitive-key values are redacted, long strings truncated, and non-scalar values reduced to type labels. Call sites intentionally omit prompts, source context, tool payloads, credentials and exception messages.

**The default launcher does not configure the `anomalyops.trace` logger at INFO.** Merely emitting a trace does not make it visible. For local diagnosis, the following launcher configures only that logger and uses the already exported offline environment from chapter 1:

```bash
python - <<'PY'
import logging
import uvicorn

trace = logging.getLogger("anomalyops.trace")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
trace.addHandler(handler)
trace.setLevel(logging.INFO)
trace.propagate = False
uvicorn.run("copilot.api.app:app", host="127.0.0.1", port=8000)
PY
```

This is an operator workaround, not a committed application fix. It intentionally does not enable DEBUG logs for HTTP clients or print `.env` contents.

| Symptom | First checks |
| --- | --- |
| CLI cannot import `copilot`, but pytest passes | Activate the intended environment and reinstall with `python -m pip install -e '.[dev]'`; pytest adds the repo root independently |
| `/health` works but `/query` fails | Inspect manifest path or database/provider configuration; health is not a readiness check |
| Answer is irrelevant with a positive confidence | Check backend/provider; deterministic hashing is not semantic retrieval |
| Unexpected OpenAI call from a CLI | Check exported provider variables and `.env`; `query_manifest.py --answer` uses the configured answer provider |
| Generic query 500 in pgvector mode | Check database reachability, embedding configuration and provider failure traces |
| Valid-looking code answer returns 502 | Check citation-marker parsing, especially bracketed indexing such as `features[0]` |
| Triage says incomplete data despite known critical events | Check upstream severity casing, selected severities, filters and run contents |
| No JSON traces in server output | Configure `anomalyops.trace` at INFO with a handler |
| Vector dimension mismatch | Verify provider dimensions and actual table schema; changing `.env` does not migrate the column |
| Reindexing leaves old rows | Verify you used replacement, not the upsert-only storage script |

## Make a typical change

**Add a provider:** implement the appropriate protocol and typed failures; add factory selection and settings validation; test success, malformed output, upstream errors, and configuration failures using fakes. For embeddings, test count/order/dimensions and compatibility with the fixed storage schema. For answers/triage, preserve the existing local grounding checks rather than trusting a new SDK's parser alone. Update chapters 1, 3, and 4.

**Add an operational tool:** first define the upstream read contract and corresponding client method, then add input/output schemas and the tools adapter. Add the OpenAI tool definition, dispatch branch and state/evidence rules explicitly; instructions alone are insufficient. Test wrong run IDs, unavailable resources, malformed responses, and evidence reuse. Extend deterministic behavior only if that workflow should use the tool. Update API contracts and the sequence diagram.

**Add a source type:** update discovery/type classification and tests; establish UTF-8 or a deliberate decoding strategy and chunk boundaries. Test skipped directories, empty files, ordering, line ranges and stable source paths. Rebuild the manifest/index in development, then compare retrieval/evaluation outcomes. Update file inventory and corpus counts only after measuring them.

**Change retrieval behavior:** start with labeled failing questions, preserve citation metadata through ranking, and compare semantic and offline evaluations separately. If embedding inputs or models change, rebuild the complete index and record new provenance. A passing unit test or improved single example is not a replacement for the labeled evaluation.
