# 4. API and data contracts

[Handbook index](README.md) · Previous: [Pipelines](03-data-pipelines.md) · Next: [Upstream platform](05-upstream-platform.md)

## HTTP surface

The application factory is [`create_app`](../../copilot/api/app.py). Its injected settings, OpenAI client, and anomaly HTTP transport make routes testable without external services. The module-level `app` loads process environment settings at import. The browser and API share an origin; the frontend is a static file with no build step.

| Route | Behavior |
| --- | --- |
| `GET /` | Serves the query UI; omitted from OpenAPI |
| `GET /health` | Returns `{"status":"ok"}`; does not probe Postgres, OpenAI, the manifest, or the anomaly API |
| `POST /query` | Returns an answer or refusal, citations, confidence, and optional context |
| `POST /triage` | Returns an operational report through the configured triage agent |
| `GET /docs`, `/redoc`, `/openapi.json` | FastAPI's generated contract documentation |

Each request receives an `X-Request-ID`. Incoming values are preserved only when they match `[A-Za-z0-9._:-]{1,128}`; otherwise a UUID hex string is generated. No authentication/authorization layer is implemented in this application.

## Query

Source: [`schemas/query.py`](../../copilot/schemas/query.py).

```json
{
  "query": "Which tables store predictions and alerts?",
  "top_k": 3,
  "min_score": 0.0,
  "show_context": true
}
```

| Field | Current validation/default |
| --- | --- |
| `query` | Required string, minimum length 1; whitespace is not rejected here; no maximum length |
| `top_k` | Integer greater than zero, default 3; no upper bound |
| `min_score` | Float at least zero, default 0; no upper bound, so values above 1 force threshold refusal for normal cosine scores |
| `show_context` | Boolean, default false |

An illustrative **supported response**, not a captured model output:

```json
{
  "answer": "Predictions are stored in model_predictions [1].",
  "confidence": 0.78,
  "citations": [
    {"citation_id": 1, "source_path": "source_code/db.py", "start_line": 1, "end_line": 80}
  ],
  "refusal_reason": null,
  "context": null,
  "context_snippets": []
}
```

This represents `show_context=false`. When true, `context` is a formatted string and `context_snippets` includes each retrieved chunk's `citation_id`, `source_path`, `start_line`, `end_line`, `content`, and `score`. Context includes all retrieved chunks, even ones the answer did not cite. Citation IDs are one-based positions in that request's ranked context, not persistent database IDs. Paths are relative to the ingested source root: `source_code/db.py` means the bundled platform file, not a Copilot module.

A no-context refusal:

```json
{
  "answer": "",
  "confidence": 0.0,
  "citations": [],
  "refusal_reason": "No relevant context was retrieved.",
  "context": null,
  "context_snippets": []
}
```

[`GroundedAnswer`](../../copilot/schemas/answer.py) constrains confidence to `[0, 1]`, but answer/citation/refusal consistency is enforced in application/provider code rather than by that model alone. OpenAI drafts contain `answer`, `citation_ids`, and `refusal_reason`; local code resolves IDs into citation objects. Returned line ranges cover the entire cited chunk, not a model-selected sentence or independently verified minimal evidence span.

## Triage

Source: [`schemas/triage.py`](../../copilot/schemas/triage.py).

```json
{"run_id": 42, "max_events": 2}
```

`run_id` is optional/null or a positive integer. `max_events` defaults to 5 and must be 1–20. A report always contains `run_id`, `status`, `run_summary`, `findings`, `evidence`, and `refusal_reason`.

| Object | Contents |
| --- | --- |
| `run_summary` | Run ID; prediction, anomaly, row-alert and event counts; severity counts; machine count; maximum/mean anomaly score |
| Finding | Locally generated `finding_id`, severity, machine, sensor, nullable anomaly type, summary text, and evidence IDs |
| Evidence | `evidence_id`, full event record, and its fetched row-alert records |
| Event | Run/event IDs, machine, sensor, type, start/end steps, duration, alert count, severity, score/value aggregates, reasons and status |
| Row alert | Run/alert IDs, step, machine, sensor, sensor value, prediction/score, severity/type/reason/status and ground-truth metadata |

An illustrative no-alerts report:

```json
{
  "run_id": 42,
  "status": "no_alerts",
  "run_summary": {
    "run_id": 42,
    "total_predictions": 100,
    "total_anomalies_predicted": 0,
    "total_row_alerts": 0,
    "total_alert_events": 0,
    "critical_alert_events": 0,
    "warning_alert_events": 0,
    "info_alert_events": 0,
    "machines_with_alerts": 0,
    "max_anomaly_score": 0.1,
    "mean_anomaly_score": 0.02
  },
  "findings": [],
  "evidence": [],
  "refusal_reason": null
}
```

| Status | Interpretation |
| --- | --- |
| `completed` | A report with findings; OpenAI validation requires a resolved run, summary with positive event count, evidence-backed findings, and no refusal reason |
| `no_alerts` | Trusted summary reports zero events; findings/evidence are empty |
| `incomplete_data` | Insufficient evidence; OpenAI requires a reason and may include validated partial findings; deterministic mode uses this when no selected events are available |
| `refused` | OpenAI cannot perform triage; requires a reason and no findings; deterministic service has no ordinary branch producing this status |

The report schema alone does not enforce every cross-field rule above. The two provider implementations perform different levels of validation. In particular, OpenAI draft severity is one of `critical`, `warning`, `unknown`; final public finding severity is a string. The model's free-text finding summary is not mechanically checked for every factual claim.

## Operational tool contracts

Source: [`schemas/anomaly_tools.py`](../../copilot/schemas/anomaly_tools.py), [`tools/anomaly.py`](../../copilot/tools/anomaly.py).

| Tool | Input | Output wrapper | Upstream route |
| --- | --- | --- | --- |
| `get_latest_run` | Empty object | `run` | `/runs/latest` |
| `get_run_summary` | Positive `run_id` | `summary` | `/runs/{run_id}/summary` |
| `list_alert_events` | Run ID; optional severity/sensor/anomaly type; `limit=100` in 1–500; `offset=0` or greater | `events` | `/runs/{run_id}/events` |
| `get_event_alerts` | Positive run and event IDs | `alerts` | `/runs/{run_id}/events/{event_id}/alerts` |

Tool input/output wrappers forbid extra fields. They do not globally enable Pydantic strict type coercion, and their nested operational models use ordinary `BaseModel` defaults. Do not interpret “strict tools” as every field rejecting coercion or every upstream enum being constrained. OpenAI tool schemas are separately transformed to require all properties, use null for unused optional filters, and forbid additional properties. Its event-list limit may not exceed the user's `max_events`.

## Errors

Route errors generally use `{"detail":"..."}`. FastAPI request validation returns a structured `detail` array with HTTP 422.

| Condition | HTTP result |
| --- | --- |
| Invalid request fields | 422 |
| Manifest unconfigured/missing/invalid, database URL missing | 500 with a specific configuration message |
| Triage dependency not configured | 500, `Anomaly API base URL is not configured.` |
| Triage resource not found | 404, `The requested run was not found.`; underlying resource errors share this mapping |
| Invalid answer or triage provider output | 502 with a sanitized invalid-response message |
| Answer/triage model failure or operational tool failure | 502 with a sanitized provider/tool message |
| Unhandled exception, including query embedding/database failures | 500, `Internal server error.` |
| Insufficient answer context or a valid triage refusal | 200 with a refusal/status payload |

The HTTP middleware catches otherwise unhandled exceptions, emits safe metadata, and returns a generic error with a request ID. It does not retry the operation. Provider construction can fail at application import/startup before any request exists.

## Storage and identifiers

The manifest is a JSON array of [`SourceChunk`](../../copilot/schemas/chunk.py) objects:

```json
[
  {
    "chunk_id": "demo:example.py#chunk-0000",
    "source_id": "demo:example.py",
    "project_name": "demo",
    "source_type": "python",
    "source_path": "example.py",
    "chunk_index": 0,
    "content": "print('example')",
    "start_line": 1,
    "end_line": 1
  }
]
```

[`SourceChunkRecord`](../../copilot/storage/models.py) stores those fields in `source_chunks`, plus an auto-increment primary key, a unique `chunk_id`, embedding provider/model/dimension metadata, and a non-null `VECTOR(256)`. Retrieval filters metadata and uses cosine distance. No approximate-nearest-neighbor index or multi-tenant partition is declared. The dimension setting must match the fixed column; changing the environment variable does not migrate it.

| Identifier | Scope |
| --- | --- |
| `source_id` / `chunk_id` | Corpus-root name, source path, and positional chunk index |
| Citation ID | One response's ranked context |
| Run ID | Upstream SQLite pipeline run |
| Event ID / alert ID | Scoped by run in upstream composite keys |
| Evidence ID | `event-<event_id>` within a report for one run |
| Finding ID | Local report identifier; format differs by provider |
| Request ID | Trace correlation for an HTTP request |

Copilot's Postgres table has no foreign keys into upstream SQLite. Triage crosses HTTP; it never joins these databases. Source citations and operational evidence IDs are different mechanisms.

## Provider interfaces

[`providers/interfaces.py`](../../copilot/providers/interfaces.py) defines runtime-checkable Python protocols:

```python
class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str
    dimensions: int
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...

class GroundedAnswerGenerator(Protocol):
    provider_name: str
    model_name: str
    def generate(self, query: str, context: Sequence[ScoredChunk]) -> GroundedAnswer: ...

class ToolCallingTriageAgent(Protocol):
    provider_name: str
    model_name: str
    def triage(self, request: TriageRequest) -> TriageReport: ...
```

[`factory.py`](../../copilot/providers/factory.py) selects all three from the single `ai_provider` setting. Retrieval backend is a separate choice: selecting OpenAI while keeping `manifest` still uses hash retrieval, but pays for OpenAI answer generation. Dependency injection in `create_app` supports mocked OpenAI responses and mocked operational HTTP transports without modifying these interfaces.
