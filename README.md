# AnomalyOps Copilot

AnomalyOps Copilot is an AI engineering layer built on top of my Industrial Anomaly Detection and Alerting Platform.

The original platform generates synthetic industrial sensor data, engineers time-series features, trains an anomaly detection model, creates alert events, stores results in SQLite, exposes FastAPI endpoints, and displays monitoring results in Streamlit.

This project adds:

- repository and metrics ingestion
- document chunking with metadata
- vector search with Postgres and pgvector
- grounded question answering with citations
- structured tool calls for alert triage
- evaluation tests for retrieval, citations, refusals, and schema validity

## Relationship to the source platform

This is a separate project. It does not retrain the anomaly model. It ingests selected source files, docs, metrics, and sample alert outputs from the completed anomaly detection platform, then uses them as context for a copilot API.

## Target architecture

```text
source files/docs/metrics/alerts
        ↓
ingestion + chunking
        ↓
Postgres + pgvector
        ↓
retrieval
        ↓
grounded answer generation
        ↓
FastAPI copilot endpoints
        ↓
evals and tests