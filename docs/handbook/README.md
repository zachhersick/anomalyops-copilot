# Developer handbook

AnomalyOps Copilot adds two capabilities to an industrial anomaly platform: answering questions about its source code and explaining operational alert events. It does not train the anomaly model. This handbook explains the implementation, how to work on it, and where its current guarantees stop.

**Reviewed baseline:** [`73bf6e49b949c878e5556cb25e4b9ea5bf0156f1`](https://github.com/zachhersick/anomalyops-copilot/tree/73bf6e49b949c878e5556cb25e4b9ea5bf0156f1), reviewed **2026-10-19**. The clean local `main` was fetched and fast-forwarded from `07e1a39` before review. These documents describe that source revision; later code changes may require updates.

Handbook finalized and documentation checks completed on **2026-10-21**. Application test results retain their original execution date below.

## Read in this order

| Chapter | What you will learn |
| --- | --- |
| [1. Getting started](01-getting-started.md) | Vocabulary, a working offline setup, and the separate live-service setup |
| [2. Architecture and repository map](02-architecture.md) | Component responsibilities, dependency boundaries, and a linked file inventory |
| [3. Data pipelines](03-data-pipelines.md) | How source files become answers and operational events become triage reports |
| [4. API and data contracts](04-api-and-data-contracts.md) | Request/response shapes, validation, errors, storage, and provider interfaces |
| [5. Upstream anomaly platform](05-upstream-platform.md) | Synthetic sensors, features, model training, alerts, SQLite, API, and dashboard |
| [6. Development and operations](06-development-and-operations.md) | Testing, evaluations, tracing, Docker, troubleshooting, and extension recipes |
| [7. Review findings](07-review-findings.md) | Reproduced defects, engineering risks, design limitations, and validation evidence |

For a first contribution, complete chapter 1, follow one request in chapter 3, and use chapter 6 to choose the relevant tests. Read chapter 7 before interpreting demo quality or connecting a real upstream service.

## The three boundaries to remember

- **Reference corpus:** `data_sources/anomaly_detection_platform/` contains a curated source snapshot. Ingestion reads its text; it does not execute the ML pipeline or synchronize the upstream repository.
- **Knowledge service:** Copilot stores source chunks and optional embeddings, retrieves context, and generates answers with source references.
- **Operational service:** triage makes read-only HTTP calls to a separately configured anomaly API. Operational records are not obtained from the vector database.

The [main README](../../README.md) remains the product overview. The [earlier fix list](../FIX_LIST.md) is a historical review; this handbook's findings distinguish what was rechecked from what remains a hypothesis. The ignored local Graphify graph was used for navigation, with claims checked against source and tests.

## Evidence and validation

This review read application, CLI, test, configuration, and bundled upstream source. It ran the offline suite, lint, CLI ingestion/query/evaluation, in-process API smoke checks, and focused defect reproductions without paid model calls. See the [validation record](07-review-findings.md#validation-record) for exact results and untested boundaries. In particular, a green unit suite does not establish retrieval quality, production readiness, or compatibility with a deployed upstream version.

## Keep the handbook current

| Change | Update and verify |
| --- | --- |
| Module, script, or dependency boundary | Architecture map and file inventory |
| Environment setting, default, install, or Docker behavior | Getting started, configuration table, operations commands |
| Endpoint, schema, validation, or error mapping | API contracts, walkthroughs, corresponding API/schema tests |
| Chunking, source discovery, embedding model, or vector dimension | Pipeline chapter, storage contracts, ingestion examples, evaluation provenance |
| Provider or operational tool | Architecture, provider contracts, triage sequence, extension recipe, mocked provider/tool tests |
| Upstream snapshot or operational API | Upstream chapter, severity/ID compatibility checks, corpus counts and evaluations |
| Fixed or newly reproduced issue | Findings status and evidence; preserve historical result dates |

Before merging handbook updates, follow the examples, check links and Mermaid diagrams, and record the new reviewed revision/date. Keep source links relative for portability; use the baseline commit above when investigating historical behavior.
