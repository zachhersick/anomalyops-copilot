# 1. Getting started

[Handbook index](README.md) · Next: [Architecture](02-architecture.md)

## What you are running

Copilot is a Python FastAPI application with a small browser interface. A question goes through retrieval and answer generation. A triage request goes through operational tools that read an anomaly API. Those paths share provider configuration but use different data sources.

| Term | Meaning here |
| --- | --- |
| RAG | Retrieval-augmented generation: retrieve project text, then ask a model to answer using that text |
| Corpus | The files selected for ingestion |
| Chunk / manifest | A slice of source text with line metadata / a JSON list of those slices |
| Embedding | A numeric representation of text used for similarity search |
| pgvector | PostgreSQL's vector extension, used to store and compare embeddings |
| Grounding / citation | Constraining an answer to evidence / identifying the retrieved source path and line range |
| Run / row alert / event | One upstream pipeline execution / one flagged sensor observation / a group of related alerts |
| Triage | Inspecting a run and its alerts to produce findings tied to evidence |
| Provider | An implementation behind a Python protocol, such as OpenAI or the deterministic test implementation |

## First run: offline, no API key

Use Python 3.11 or newer. CI uses 3.11; this review used 3.12.2. Commands below are for a POSIX shell, run from the repository root. On Windows, activate the environment with `.venv\Scripts\Activate.ps1` and set environment variables using PowerShell syntax.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

export ANOMALYOPS_RETRIEVAL_BACKEND=manifest
export ANOMALYOPS_AI_PROVIDER=deterministic
export ANOMALYOPS_MANIFEST_PATH=outputs/chunks.json
unset ANOMALYOPS_ANOMALY_API_BASE_URL

python scripts/ingest_sources.py data_sources/anomaly_detection_platform \
  --output outputs/chunks.json
python scripts/query_manifest.py outputs/chunks.json \
  'Which SQLite tables store pipeline runs, predictions, and alerts?' \
  --answer --show-context
python -m uvicorn copilot.api.app:app --reload --host 127.0.0.1 --port 8000
```

The reviewed corpus produces **32 documents and 133 chunks**. The CLI should print an answer, confidence, citation, and source excerpt. Its answer may be irrelevant: offline retrieval hashes text, and the deterministic generator echoes the highest-scoring chunk. This verifies the plumbing, not question-answering accuracy.

Open [the query UI](http://127.0.0.1:8000/) or [interactive API documentation](http://127.0.0.1:8000/docs). In another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Which SQLite tables store pipeline runs, predictions, and alerts?","show_context":true}'
```

Health returns `{"status":"ok"}`. Query returns the [documented response](04-api-and-data-contracts.md#query), including context because it was requested. Stop Uvicorn with Ctrl-C. Generated `outputs/` files are ignored by Git.

Most CLIs load `.env`, but exported variables take precedence. The explicit deterministic export above prevents `query_manifest.py --answer` from selecting OpenAI from an existing `.env`. Uvicorn does not load `.env` in this command. A normal `/triage` call is unavailable until an anomaly API base URL is configured.

## Real semantic retrieval: OpenAI and pgvector

This path needs Docker Desktop running, network access, and an OpenAI project key with API billing. **Indexing, live queries, the demo, and semantic evaluations make paid API calls.** Model names below are the repository's example configuration, not a claim about current availability or price.

If you used the offline exports above, open a fresh shell and reactivate `.venv`, or unset those exports before proceeding. Otherwise they override `.env`.

1. If no `.env` exists, copy `.env.example` to `.env`. Preserve an existing file. Set `OPENAI_API_KEY` privately.
2. Confirm `ANOMALYOPS_RETRIEVAL_BACKEND=pgvector`, `ANOMALYOPS_AI_PROVIDER=openai`, database URL, and model names match [`.env.example`](../../.env.example). The stored vector dimension is **256**.
3. Build the manifest if necessary, then run:

```bash
docker compose up -d postgres
python scripts/ingest_sources.py data_sources/anomaly_detection_platform \
  --output outputs/chunks.json
# Replaces the entire source_chunks index in the configured database.
python scripts/reindex_embeddings.py outputs/chunks.json
python -m uvicorn copilot.api.app:app --reload --env-file .env \
  --host 127.0.0.1 --port 8000
```

Use a development database for the initial setup. Reindexing creates the extension/table if absent, generates and checks embeddings, then transactionally replaces the index. It is not a schema migration and cannot convert an existing table with a different vector dimension. See [storage](04-api-and-data-contracts.md#storage-and-identifiers).

For the full paid demonstration, `python scripts/run_demo.py` uses OpenAI and the configured pgvector database. Its operational dataset is an in-process mock transport with run 42; it does not demonstrate connectivity to a live anomaly API. Inspect [the demo implementation](../../scripts/run_demo.py) before treating its results as operational evidence.

## Connect operational triage

Run the separate anomaly platform API on port 8001 and configure:

```text
ANOMALYOPS_ANOMALY_API_BASE_URL=http://localhost:8001
```

Restart Copilot after configuration changes. Then:

```bash
curl --fail http://127.0.0.1:8000/triage \
  -H 'Content-Type: application/json' \
  -d '{"max_events":2}'
```

Omitting `run_id` selects the latest upstream run. Setting it requires that run to exist. Deterministic triage requires the upstream API too; it avoids model calls but is not a synthetic-data mode. The [upstream chapter](05-upstream-platform.md) explains service setup and the verified severity-filter incompatibility with the bundled snapshot.

For a key-free triage smoke check, use the existing mocked tests:

```bash
python -m pytest tests/test_triage_api.py tests/test_triage_service.py -q
```

## First-day checklist

- Get `/health` and `/query` working, then inspect `context_snippets` alongside the cited source.
- Follow `create_app` → `query_service` → retrieval → answer provider in chapter 3.
- Run `python -m pytest -m 'not integration' -q` and `ruff check .`.
- Read the provider and tool boundaries before adding a feature.
- Read the findings before interpreting confidence as correctness or using the Docker setup outside development.
