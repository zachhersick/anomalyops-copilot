# 7. Review findings

[Handbook index](README.md) · Previous: [Development and operations](06-development-and-operations.md)

Reviewed at commit `73bf6e49b949c878e5556cb25e4b9ea5bf0156f1` on 2026-10-19. No application fixes were made. This is a codebase review with focused reproductions, not an exhaustive security or performance audit.

**Evidence labels:** “reproduced” means a local check demonstrated the behavior during this review; “code-confirmed” means the implementation establishes the behavior but its downstream impact was not load-tested or exercised against a live provider. Confidence applies to the stated observation, not to a forecast of production impact. High severity means a core workflow can miss data or produce materially misleading results; medium means incorrect handling or a meaningful operational risk; low means maintenance friction.

## F1. Upstream severity filters do not match deterministic triage

**High · reproduced · high confidence.** [`TriageService.triage`](../../copilot/services/triage.py) requests lowercase `critical` and `warning`. The bundled upstream [`alerts.py`](../../data_sources/anomaly_detection_platform/source_code/alerts.py) emits uppercase severities, while [`get_filtered_alert_events_for_run`](../../data_sources/anomaly_detection_platform/source_code/db_queries.py) uses exact `max_severity = ?` comparisons. The upstream API passes the filter through unchanged.

**Reproduction:** create the bundled SQLite schema in memory, insert run 42 and one event with `max_severity='CRITICAL'`, then query through the bundled helper. Filtering with `critical` returns zero rows; `CRITICAL` returns one. Wiring those actual query helpers into deterministic triage returns `incomplete_data` despite a trusted summary containing one event.

**Impact:** deterministic triage cannot find critical/warning events from this snapshot's native data. The demo and current client/service fixtures use lowercase values, so they conceal the incompatibility. This does not establish the behavior of a separately deployed upstream version.

**Recommendation:** agree a canonical severity representation at the service boundary, normalize filters and values consistently, and add a contract test using the upstream SQLite/API behavior. Test both casing variants and info-only runs. Do not fix this by silently rewriting fixture expectations without testing the real boundary.

## F2. Python indexing is mistaken for citation markers

**Medium · reproduced · high confidence.** [`OpenAIGroundedAnswerGenerator._validate_draft`](../../copilot/providers/openai_answers.py) extracts every bracketed integer using `\[(\d+)\]`.

**Reproduction:** a draft with answer `The model reads the first feature [1].`, citation IDs `[1]`, and one context chunk passes validation. Changing the answer to ``The model reads `features[0]` first [1].`` raises `InvalidGroundedAnswerResponseError`. The API maps that error to HTTP 502.

**Impact:** supported explanations of ordinary Python source can be rejected. **Recommendation:** separate citation rendering from code syntax, or parse citation markers outside inline/fenced code with explicit boundary rules. Add cases for `x[0]`, `row[1]`, repeated valid references, and fenced code. Avoid relying on a single negative-lookbehind without testing Markdown/code cases.

## F3. Offline confidence can accompany irrelevant output

**Medium · reproduced; intentional provider limitation · high confidence.** [`retrieval/search.py`](../../copilot/retrieval/search.py) compares SHA-256-derived vectors, and [`deterministic_answers.py`](../../copilot/providers/deterministic_answers.py) echoes the highest-ranked chunk.

**Reproduction:** on the 133-chunk manifest, the SQLite-table question selected `README.md:491-519` at about 0.6015. “What is the capital of France?” returned `tests/test_api.py:491-570` at about 0.6179 with no refusal. This is deterministic output, not random execution, but its similarity has no semantic meaning.

**Impact:** the default browser/Docker experience can appear confidently wrong. **Recommendation:** label offline mode in the UI and describe confidence as retrieval similarity, or introduce a separately evaluated lexical retriever for usable key-free search. Retain hash providers for repeatable contract tests. The earlier fix list's “never refuses” wording is too broad: empty context and `min_score` gating do produce refusals.

## F4. Trace emission is disabled by default logging configuration

**Medium · reproduced at logger configuration level · high confidence.** [`observability.py`](../../copilot/observability.py) calls `logger.info` on `anomalyops.trace`; application and CLI source do not configure an INFO handler/level for it.

**Reproduction:** after importing and exercising the app in a fresh offline process, `logger.isEnabledFor(logging.INFO)` was false and `logger.handlers` was empty. Test capture can explicitly enable INFO, so passing trace tests does not establish default console delivery. This review did not rerun the older fix list's external Uvicorn trace capture.

**Impact:** the documented trace feature is not visible under default logger configuration. **Recommendation:** ship explicit logger configuration and test a subprocess launcher, not only `caplog`. The operations chapter supplies a scoped local workaround.

## F5. Query failures have inconsistent error mapping

**Medium · reproduced for embedding error; database path code-confirmed · high confidence.** [`app.py`](../../copilot/api/app.py) maps answer-generation failures to 502, but has no dedicated handler for embedding-provider or SQLAlchemy operational failures.

**Reproduction:** inject `EmbeddingProviderError` from `query_service` through a real TestClient request. The response is HTTP 500 with `{"detail":"Internal server error."}`. **Impact:** upstream-provider failure and unavailable database cases are hard for callers to distinguish from application bugs. **Recommendation:** introduce sanitized, typed mappings for dependency failures and test embedding failures, unavailable Postgres, and invalid dimensions without exposing credentials or raw exception messages.

## F6. Query inputs lack upper bounds

**Medium · reproduced at schema boundary · high confidence.** [`QueryRequest`](../../copilot/schemas/query.py) constrains only minimum query length and positive `top_k`.

**Reproduction:** constructing the model with `top_k=1_000_000_000` or a 2,000,000-character query succeeds. No paid request was made with those values. **Impact:** large inputs can increase embedding cost, context size, memory and latency; the practical impact depends on deployment/provider limits. **Recommendation:** define query-length, retrieval-count and prompt/context budgets, validate them before provider calls, and test boundary values. Also reject whitespace-only queries at the public boundary for consistent 422 behavior.

## F7. Work and upstream latency budgets are incomplete

**Medium · code-confirmed · high confidence in implementation, unmeasured load impact.** [`factory.py`](../../copilot/providers/factory.py) constructs OpenAI clients without application-specific timeout/retry settings. [`openai_triage.py`](../../copilot/providers/openai_triage.py) bounds tool-bearing rounds but has no overall deadline or explicit per-response function-call count cap. [`openai_embeddings.py`](../../copilot/providers/openai_embeddings.py) submits all documents in one request.

**Impact:** SDK defaults control timeouts/retries; large corpora and long tool loops can exceed practical request budgets. Exact current provider limits and SDK defaults were not externally researched for this review. **Recommendation:** set explicit configurable deadlines/retries, bound tool count and input tokens, batch embeddings, and test timeout/partial-batch failure behavior. Preserve the existing embedding-before-delete and transactional replacement guarantees. Eight tool-bearing rounds can require a ninth model request for completion; the old fix list's “eight model calls” is not exact.

## F8. Semantic index preflight cannot detect changed chunk contents

**Medium · code-confirmed · high confidence.** [`validate_semantic_index`](../../scripts/run_rag_evals.py) compares row count, chunk-ID set and provider/model/dimensions. [`build_chunk_id`](../../copilot/ingestion/chunking.py) uses source identity plus positional index, without hashing contents.

**Trigger:** edit source text without changing paths or chunk count, regenerate the manifest, and leave the database untouched. The same IDs/configuration can pass preflight even though stored content/embeddings are stale. The output snapshot's manifest hash does not prove the database was built from that manifest.

**Recommendation:** persist and validate an index-generation/corpus digest or per-chunk content hash. Add a case where IDs remain identical but contents change. Until then, reindex deliberately before semantic evaluation and avoid claiming content equality from the preflight alone.

## F9. Triage quality coverage does not establish real incident accuracy

**Medium · code-confirmed coverage gap · high confidence.** [`evals/triage_cases.json`](../../evals/triage_cases.json) has two unlabeled latest-run cases. Expected status/findings/count metrics are unavailable. Existing [service tests](../../tests/test_triage_service.py) and [demo transport](../../scripts/run_demo.py) use simplified lowercase-severity data.

**Impact:** strong schema/evidence tests can pass while cross-service compatibility or prioritization remains wrong. Deterministic triage filters to critical/warning events; an info-only run produces `incomplete_data`. It also ranks only the limited events returned by the upstream API, not necessarily all events in the run. OpenAI draft validation checks structured fields but cannot establish all free-text summary claims.

**Recommendation:** add labeled operational scenarios, uppercase-severity integration cases, info-only runs, pagination/ordering cases and changed-evidence cases. Define expected behavior before treating these scenarios as defects. Malformed tool arguments currently fail closed with 502; a bounded correction loop is a possible usability improvement, not an existing guarantee.

## F10. Bundled feature builder discards its sort result

**Medium for arbitrary upstream input · code-confirmed · high confidence.** [`build_features`](../../data_sources/anomaly_detection_platform/source_code/features.py) calls `df.sort_values(...).reset_index(...)` without assigning the result or sorting in place, then computes per-machine deltas and rolling features in incoming order.

**Impact:** shuffled observations can yield features that do not represent chronological history. Generator-produced ordering can hide the issue. This review did not retrain the model or measure the accuracy impact. **Recommendation:** explicitly sort/assign before feature computation and add an order-invariance test comparing chronological and shuffled copies of the same observations. Keep any upstream code change separate from this handbook.

## Other constraints and maintenance work

| Item | Evidence, impact, recommendation, confidence |
| --- | --- |
| Medium: development deployment defaults | [Compose](../../docker-compose.yml) publishes API and Postgres ports on all host interfaces with fixed development DB credentials; [Dockerfile](../../Dockerfile) has no non-root user, and [app](../../copilot/api/app.py) has no auth. Code-confirmed, high confidence. Use deployment-specific network/auth/secret controls and a non-root runtime before broader exposure; this is not a claim of exploitation. |
| Medium: retrieval scaling | Manifest mode reloads and re-embeds every chunk per query; Postgres schema declares no ANN vector index. Code-confirmed, high confidence. Cache an explicitly versioned manifest index and measure database query plans/latency before selecting vector indexing changes. No load benchmark was run. |
| Medium: snapshot and line-based chunking | [Discovery/chunking](../../copilot/ingestion/chunking.py) uses a fixed snapshot and 80-line windows; [storage](../../copilot/storage/chunks.py) embeds content without paths. Code-confirmed, high confidence. Evaluate structure-aware chunks and path metadata on labeled cases; any claimed quality gain remains a hypothesis. |
| Medium: fixed vector schema | [Settings](../../copilot/api/settings.py) accept positive dimensions, but [storage](../../copilot/storage/models.py) is fixed at 256. Code-confirmed, high confidence. Fail early on incompatible configuration and design an explicit migration/rebuild workflow; there is no automatic schema migration today. |
| Low: dependency reproducibility | [pyproject.toml](../../pyproject.toml) has unpinned runtime/dev dependencies and no tracked lockfile. Code-confirmed, high confidence. Adopt a reproducible dependency policy. Declared `numpy`, `pandas`, `tiktoken`, and `pydantic-settings` are not directly imported by Copilot/scripts; validate packaging needs before removal. |
| Low: lifecycle and version coverage | The shutdown hook emits deprecation warnings; CI tests only Python 3.11 despite `>=3.11`. Current review ran on 3.12.2. High confidence. Migrate to lifespan and choose a supported-version test matrix. A new Starlette/httpx warning is environment-specific evidence, not a reason to add a dependency blindly. |
| Medium upstream limitation: partial loads | [Loader](../../data_sources/anomaly_detection_platform/source_code/load_to_db.py) commits the run and table loads separately and skips missing files. Code-confirmed, high confidence. Define run completion state/atomic loading if consumers need complete snapshots. |

Suggested remediation order: cross-service severity compatibility; citation parsing; visible trace configuration and input bounds; typed dependency failures and budgets; content-aware index provenance and labeled triage evaluation. Keep quality improvements measurable and separate from correctness fixes.

## Validation record

| Check | Result on 2026-10-19 |
| --- | --- |
| Repository freshness | Fetched origin and fast-forwarded clean main to `73bf6e4` before review |
| Environment | Existing Python 3.12.2 venv; FastAPI 0.141.1, OpenAI SDK 2.53.0, Pydantic 2.13.4, pytest 9.1.1, Ruff 0.16.1 |
| Package setup | Existing editable install failed CLI imports; refreshed with `python -m pip install --no-deps -e .`; CLI checks then passed. No tracked dependency changes |
| Offline suite | `python -m pytest -m 'not integration' -q`: **688 passed, 3 deselected**, 123 pre-existing dependency/lifecycle warnings |
| Lint | `ruff check .`: passed |
| Integration selection | `python -m pytest -m integration -q -rs`: **3 skipped**, `ANOMALYOPS_TEST_DATABASE_URL` unset; Docker daemon unavailable |
| Ingestion CLI | 32 documents, 133 chunks written to a temporary manifest |
| Query CLI | Offline `--answer --json` completed successfully against that manifest |
| Offline RAG evaluation | 10 cases: 3 passed, 7 failed; retrieval hit 25%, citation hit 12.5%, schema/citation validity 100%, refusal accuracy 100%. These are evaluation outcomes, not unit-test failures |
| In-process HTTP smoke | `/`, `/health`, `/docs`, `/openapi.json` returned 200; query with context and threshold refusal worked; invalid `top_k=0` returned 422 |
| Mock operational smoke | Run 42 with two events returned 200/completed and two findings using deterministic triage plus the demo transport |
| Focused defect checks | Severity mismatch, citation indexing collision, unsupported offline answer, disabled INFO traces, query bounds and embedding-error mapping reproduced as described above |
| Paid or external checks | No OpenAI calls, semantic rerun, live anomaly API validation, upstream training, image build, or database integration execution |

The committed [August semantic snapshot](../../evals/results/openai-pgvector-2026-08-06.json) remains historical evidence: 15/20 cases passed. Its five failures were not fixed or rerun. Temporary smoke outputs are deliberately not committed; the documented source functions, commands and inputs identify how to reproduce the observations.

Documentation validation completed on **2026-10-21**: all eight handbook files were checked; all 280 local links and referenced headings resolved; six fenced JSON examples validated against their Pydantic contracts; all five diagrams passed the Mermaid parser; the inventory covered all 169 baseline tracked files; whitespace checks passed. `docker compose config --quiet` also passed, and the offline evaluator's `--strict` exit code was verified as 1 for its seven failing cases. Diagram syntax validation does not imply a visual browser-rendering test. Application code was unchanged after the recorded test run.
