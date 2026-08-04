import json

import pytest

import scripts.run_demo as demo_module
from copilot.api.settings import ApiSettings
from scripts.run_demo import (
    DEMO_QUERY,
    UNSUPPORTED_QUERY,
    main,
    print_demo,
    run_demo,
    validate_demo_settings,
)


def make_settings(
    **updates,
) -> ApiSettings:
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url=(
            "postgresql+psycopg://"
            "anomalyops:anomalyops@"
            "localhost:5432/anomalyops"
        ),
        ai_provider="openai",
        embedding_model=(
            "text-embedding-3-small"
        ),
        grounded_answer_model=(
            "gpt-5-mini"
        ),
        triage_model="gpt-5-mini",
        openai_api_key="test-key",
        embedding_dimensions=16,
    )

    return settings.model_copy(
        update=updates
    )


def make_demo_result():
    return {
        "mode": "openai+pgvector",
        "embedding_model": (
            "text-embedding-3-small"
        ),
        "answer_model": "gpt-5-mini",
        "triage_model": "gpt-5-mini",
        "health": {
            "status": "ok",
        },
        "query": {
            "answer": (
                "The platform stores "
                "operational data in SQLite [1]."
            ),
            "confidence": 0.91,
            "citations": [
                {
                    "citation_id": 1,
                    "source_path": (
                        "source_code/db.py"
                    ),
                    "start_line": 10,
                    "end_line": 50,
                }
            ],
            "refusal_reason": None,
            "context": "context",
            "context_snippets": [
                {
                    "citation_id": 1,
                    "source_path": (
                        "source_code/db.py"
                    ),
                    "start_line": 10,
                    "end_line": 50,
                    "content": "database code",
                    "score": 0.91,
                }
            ],
        },
        "unsupported_query": {
            "answer": "",
            "confidence": 0.0,
            "citations": [],
            "refusal_reason": (
                "Retrieved context does not "
                "support this answer."
            ),
            "context": "context",
            "context_snippets": [],
        },
        "triage": {
            "run_id": 42,
            "status": "completed",
            "run_summary": {},
            "findings": [
                {
                    "finding_id": "finding-7",
                    "severity": "critical",
                    "machine_id": 3,
                    "sensor": "temperature",
                    "anomaly_type": "spike",
                    "summary": (
                        "Critical temperature event."
                    ),
                    "evidence_ids": [
                        "event-7"
                    ],
                }
            ],
            "evidence": [],
            "refusal_reason": None,
        },
    }


@pytest.mark.parametrize(
    (
        "updates",
        "message",
    ),
    [
        (
            {
                "retrieval_backend": (
                    "manifest"
                )
            },
            "ANOMALYOPS_RETRIEVAL_BACKEND",
        ),
        (
            {
                "ai_provider": (
                    "deterministic"
                )
            },
            "ANOMALYOPS_AI_PROVIDER",
        ),
        (
            {
                "database_url": None,
            },
            "ANOMALYOPS_DATABASE_URL",
        ),
        (
            {
                "openai_api_key": None,
            },
            "OPENAI_API_KEY",
        ),
        (
            {
                "embedding_model": None,
            },
            "ANOMALYOPS_EMBEDDING_MODEL",
        ),
        (
            {
                "grounded_answer_model": None,
            },
            (
                "ANOMALYOPS_"
                "GROUNDED_ANSWER_MODEL"
            ),
        ),
        (
            {
                "triage_model": None,
            },
            "ANOMALYOPS_TRIAGE_MODEL",
        ),
        (
            {
                "embedding_dimensions": 8,
            },
            (
                "ANOMALYOPS_"
                "EMBEDDING_DIMENSIONS"
            ),
        ),
    ],
)
def test_validate_demo_settings(
    updates,
    message,
):
    with pytest.raises(
        RuntimeError,
        match=message,
    ):
        validate_demo_settings(
            make_settings(
                **updates
            )
        )


def test_validate_demo_settings_accepts_openai_pgvector():
    validate_demo_settings(
        make_settings()
    )


def test_print_demo_shows_rag_evidence(
    capsys,
):
    print_demo(
        make_demo_result()
    )

    output = capsys.readouterr().out

    assert "openai+pgvector" in output
    assert "RAG Retrieval" in output
    assert DEMO_QUERY in output
    assert (
        "source_code/db.py:10-50"
        in output
    )
    assert "score=0.9100" in output
    assert "Grounded Answer" in output
    assert "Citations" in output
    assert "Unsupported Question" in output
    assert UNSUPPORTED_QUERY in output
    assert "Refused:" in output
    assert "Tool-Calling Triage" in output
    assert "CRITICAL" in output


class FakeResponse:
    def __init__(
        self,
        payload,
    ):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeTestClient:
    requests = []

    def __init__(
        self,
        app,
    ):
        self.app = app

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        return None

    def get(
        self,
        path,
    ):
        self.requests.append(
            ("GET", path, None)
        )

        return FakeResponse(
            {
                "status": "ok",
            }
        )

    def post(
        self,
        path,
        json,
    ):
        self.requests.append(
            ("POST", path, json)
        )

        result = make_demo_result()

        if path == "/triage":
            return FakeResponse(
                result["triage"]
            )

        if json["query"] == DEMO_QUERY:
            return FakeResponse(
                result["query"]
            )

        return FakeResponse(
            result["unsupported_query"]
        )


def test_run_demo_uses_real_demo_configuration(
    monkeypatch,
):
    captured = {}

    def fake_create_app(
        *,
        settings,
        anomaly_transport,
    ):
        captured["settings"] = settings
        captured[
            "anomaly_transport"
        ] = anomaly_transport
        return object()

    FakeTestClient.requests = []

    monkeypatch.setattr(
        demo_module,
        "create_app",
        fake_create_app,
    )
    monkeypatch.setattr(
        demo_module,
        "TestClient",
        FakeTestClient,
    )

    result = run_demo(
        make_settings()
    )

    assert (
        result["mode"]
        == "openai+pgvector"
    )

    settings = captured["settings"]

    assert (
        settings.retrieval_backend
        == "pgvector"
    )
    assert settings.ai_provider == "openai"
    assert (
        settings.anomaly_api_base_url
        == "http://anomaly-api.demo"
    )

    query_requests = [
        request
        for request in FakeTestClient.requests
        if (
            request[0] == "POST"
            and request[1] == "/query"
        )
    ]

    assert len(query_requests) == 2
    assert (
        query_requests[0][2]["query"]
        == DEMO_QUERY
    )
    assert (
        query_requests[0][2][
            "show_context"
        ]
        is True
    )


def test_main_json_mode(
    monkeypatch,
    capsys,
):
    settings = make_settings()
    result = make_demo_result()

    monkeypatch.setattr(
        demo_module,
        "load_dotenv",
        lambda: None,
    )
    monkeypatch.setattr(
        demo_module,
        "load_api_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        demo_module,
        "run_demo",
        lambda received_settings: result,
    )

    exit_code = main(
        ["--json"]
    )

    payload = json.loads(
        capsys.readouterr().out
    )

    assert exit_code == 0
    assert (
        payload["mode"]
        == "openai+pgvector"
    )
    assert (
        payload["query"]["citations"]
    )
    assert (
        payload["unsupported_query"][
            "refusal_reason"
        ]
    )