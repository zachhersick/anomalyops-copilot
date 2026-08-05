import argparse
import json
import warnings
from typing import Any

import httpx
from dotenv import load_dotenv

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=(
            "Using `httpx` with `starlette.testclient` "
            "is deprecated.*"
        ),
    )
    from fastapi.testclient import TestClient

from copilot.api.app import create_app
from copilot.api.settings import (
    ApiSettings,
    load_api_settings,
)
from copilot.storage.models import EMBEDDING_DIMENSIONS


DEMO_ANOMALY_API_URL = "http://anomaly-api.demo"

DEMO_QUERY = (
    "Which SQLite tables store pipeline runs, "
    "predictions, and alerts?"
)

UNSUPPORTED_QUERY = (
    "What will the weather in Tokyo be tomorrow?"
)


def _run_summary() -> dict[str, object]:
    return {
        "run_id": 42,
        "total_predictions": 500,
        "total_anomalies_predicted": 14,
        "total_row_alerts": 8,
        "total_alert_events": 2,
        "critical_alert_events": 1,
        "warning_alert_events": 1,
        "info_alert_events": 0,
        "machines_with_alerts": 2,
        "max_anomaly_score": 0.98,
        "mean_anomaly_score": 0.9,
    }


def _critical_event() -> dict[str, object]:
    return {
        "run_id": 42,
        "event_id": 7,
        "machine_id": 3,
        "sensor": "temperature",
        "anomaly_type": "spike",
        "start_step": 120,
        "end_step": 122,
        "duration": 3,
        "alert_count": 2,
        "max_severity": "critical",
        "max_severity_reason": "High anomaly score",
        "max_anomaly_score": 0.98,
        "mean_anomaly_score": 0.96,
        "min_sensor_value": 72.0,
        "max_sensor_value": 118.0,
        "first_reason": "Temperature spike detected",
        "status": "open",
        "real_value": 1,
    }


def _warning_event() -> dict[str, object]:
    return {
        "run_id": 42,
        "event_id": 8,
        "machine_id": 5,
        "sensor": "pressure",
        "anomaly_type": "drift",
        "start_step": 200,
        "end_step": 204,
        "duration": 5,
        "alert_count": 1,
        "max_severity": "warning",
        "max_severity_reason": "Elevated anomaly score",
        "max_anomaly_score": 0.82,
        "mean_anomaly_score": 0.82,
        "min_sensor_value": 31.0,
        "max_sensor_value": 48.0,
        "first_reason": "Pressure drift detected",
        "status": "open",
        "real_value": 1,
    }


def _critical_alerts() -> list[dict[str, object]]:
    return [
        {
            "run_id": 42,
            "alert_id": 101,
            "step": 120,
            "machine_id": 3,
            "sensor": "temperature",
            "sensor_value": 113.0,
            "prediction": 1,
            "anomaly_score": 0.94,
            "severity": "critical",
            "alert_type": "anomaly",
            "reason": "Temperature spike detected",
            "status": "open",
            "anomaly_type": "spike",
            "real_value": 1,
        },
        {
            "run_id": 42,
            "alert_id": 102,
            "step": 122,
            "machine_id": 3,
            "sensor": "temperature",
            "sensor_value": 118.0,
            "prediction": 1,
            "anomaly_score": 0.98,
            "severity": "critical",
            "alert_type": "anomaly",
            "reason": "Temperature spike detected",
            "status": "open",
            "anomaly_type": "spike",
            "real_value": 1,
        },
    ]


def _warning_alerts() -> list[dict[str, object]]:
    return [
        {
            "run_id": 42,
            "alert_id": 103,
            "step": 200,
            "machine_id": 5,
            "sensor": "pressure",
            "sensor_value": 48.0,
            "prediction": 1,
            "anomaly_score": 0.82,
            "severity": "warning",
            "alert_type": "anomaly",
            "reason": "Pressure drift detected",
            "status": "open",
            "anomaly_type": "drift",
            "real_value": 1,
        }
    ]


def build_demo_anomaly_transport() -> httpx.MockTransport:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        path = request.url.path

        if path == "/runs/latest":
            return httpx.Response(
                200,
                json={"run_id": 42},
            )

        if path == "/runs/42/summary":
            return httpx.Response(
                200,
                json=_run_summary(),
            )

        if path == "/runs/42/events":
            severity = request.url.params.get(
                "severity"
            )

            if severity == "critical":
                events = [_critical_event()]
            elif severity == "warning":
                events = [_warning_event()]
            else:
                events = [
                    _critical_event(),
                    _warning_event(),
                ]

            return httpx.Response(
                200,
                json=events,
            )

        if path == "/runs/42/events/7/alerts":
            return httpx.Response(
                200,
                json=_critical_alerts(),
            )

        if path == "/runs/42/events/8/alerts":
            return httpx.Response(
                200,
                json=_warning_alerts(),
            )

        return httpx.Response(
            404,
            json={"detail": "Not found"},
        )

    return httpx.MockTransport(
        handler
    )


def validate_demo_settings(
    settings: ApiSettings,
) -> None:
    if settings.retrieval_backend != "pgvector":
        raise RuntimeError(
            "Demo requires "
            "ANOMALYOPS_RETRIEVAL_BACKEND=pgvector."
        )

    if settings.ai_provider != "openai":
        raise RuntimeError(
            "Demo requires "
            "ANOMALYOPS_AI_PROVIDER=openai."
        )

    if (
        settings.database_url is None
        or not settings.database_url.strip()
    ):
        raise RuntimeError(
            "ANOMALYOPS_DATABASE_URL is required."
        )

    if (
        settings.openai_api_key is None
        or not settings.openai_api_key.strip()
    ):
        raise RuntimeError(
            "OPENAI_API_KEY is required."
        )

    if (
        settings.embedding_model is None
        or not settings.embedding_model.strip()
    ):
        raise RuntimeError(
            "ANOMALYOPS_EMBEDDING_MODEL is required."
        )

    if (
        settings.grounded_answer_model is None
        or not settings.grounded_answer_model.strip()
    ):
        raise RuntimeError(
            "ANOMALYOPS_GROUNDED_ANSWER_MODEL is required."
        )

    if (
        settings.triage_model is None
        or not settings.triage_model.strip()
    ):
        raise RuntimeError(
            "ANOMALYOPS_TRIAGE_MODEL is required."
        )

    if (
        settings.embedding_dimensions
        != EMBEDDING_DIMENSIONS
    ):
        raise RuntimeError(
            "ANOMALYOPS_EMBEDDING_DIMENSIONS "
            f"must be {EMBEDDING_DIMENSIONS}."
        )


def run_demo(
    settings: ApiSettings,
) -> dict[str, Any]:
    validate_demo_settings(
        settings
    )

    demo_settings = settings.model_copy(
        update={
            "anomaly_api_base_url": (
                DEMO_ANOMALY_API_URL
            ),
        }
    )

    app = create_app(
        settings=demo_settings,
        anomaly_transport=(
            build_demo_anomaly_transport()
        ),
    )

    with TestClient(app) as client:
        health_response = client.get(
            "/health"
        )
        health_response.raise_for_status()

        query_response = client.post(
            "/query",
            json={
                "query": DEMO_QUERY,
                "top_k": 3,
                "min_score": 0.0,
                "show_context": True,
            },
        )
        query_response.raise_for_status()

        unsupported_response = client.post(
            "/query",
            json={
                "query": UNSUPPORTED_QUERY,
                "top_k": 3,
                "min_score": 0.0,
                "show_context": True,
            },
        )
        unsupported_response.raise_for_status()

        triage_response = client.post(
            "/triage",
            json={
                "run_id": None,
                "max_events": 2,
            },
        )
        triage_response.raise_for_status()

    return {
        "mode": "openai+pgvector",
        "embedding_model": (
            settings.embedding_model
        ),
        "answer_model": (
            settings.grounded_answer_model
        ),
        "triage_model": (
            settings.triage_model
        ),
        "health": health_response.json(),
        "query": query_response.json(),
        "unsupported_query": (
            unsupported_response.json()
        ),
        "triage": triage_response.json(),
    }


def print_demo(
    result: dict[str, Any],
) -> None:
    query = result["query"]
    unsupported = result[
        "unsupported_query"
    ]
    triage = result["triage"]

    print("AnomalyOps Copilot Demo")
    print("=======================")
    print(
        f"Mode: {result['mode']}"
    )
    print(
        "Embedding model: "
        f"{result['embedding_model']}"
    )
    print(
        "Answer model: "
        f"{result['answer_model']}"
    )
    print(
        "Triage model: "
        f"{result['triage_model']}"
    )
    print(
        f"Health: "
        f"{result['health']['status']}"
    )

    print()
    print("RAG Retrieval")
    print("=============")
    print(
        f"Query: {DEMO_QUERY}"
    )

    for index, snippet in enumerate(
        query["context_snippets"],
        start=1,
    ):
        print(
            f"{index}. "
            f"{snippet['source_path']}:"
            f"{snippet['start_line']}-"
            f"{snippet['end_line']} "
            f"score={snippet['score']:.4f}"
        )

    print()
    print("Grounded Answer")
    print("===============")
    print(query["answer"])
    print(
        "Confidence: "
        f"{query['confidence']:.4f}"
    )

    print()
    print("Citations")
    print("=========")

    for citation in query["citations"]:
        print(
            f"[{citation['citation_id']}] "
            f"{citation['source_path']}:"
            f"{citation['start_line']}-"
            f"{citation['end_line']}"
        )

    print()
    print("Unsupported Question")
    print("====================")
    print(
        f"Query: {UNSUPPORTED_QUERY}"
    )

    if unsupported["refusal_reason"]:
        print(
            "Refused: "
            f"{unsupported['refusal_reason']}"
        )
    else:
        print(
            "Answer: "
            f"{unsupported['answer']}"
        )

    print()
    print("Tool-Calling Triage")
    print("===================")
    print(
        "Real OpenAI tool calling over controlled synthetic operational data."
    )
    print(
        f"Run: {triage['run_id']}"
    )
    print(
        f"Status: {triage['status']}"
    )

    for finding in triage["findings"]:
        print(
            "- "
            f"{finding['severity'].upper()} | "
            f"machine "
            f"{finding['machine_id']} | "
            f"{finding['sensor']} | "
            f"{finding['summary']} | "
            "evidence: "
            f"{', '.join(finding['evidence_ids'])}"
        )


def main(
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the OpenAI + pgvector "
            "AnomalyOps Copilot demo."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
    )

    args = parser.parse_args(argv)

    load_dotenv()

    settings = load_api_settings()

    result = run_demo(
        settings
    )

    if args.json:
        print(
            json.dumps(
                result,
                indent=2,
            )
        )
    else:
        print_demo(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
