import json
import logging
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

import copilot.api.app as app_module
import copilot.api.query_service as query_service_module
import copilot.providers.openai_answers as answers_module
import copilot.providers.openai_embeddings as embeddings_module
import copilot.providers.openai_triage as triage_module
from copilot.api.settings import ApiSettings
from copilot.observability import (
    TRACE_LOGGER_NAME,
    get_request_id,
)
from copilot.providers.openai_answers import (
    OpenAIGroundedAnswerGenerator,
)
from copilot.providers.openai_embeddings import (
    OpenAIEmbeddingProvider,
)
from copilot.providers.openai_triage import (
    OpenAITriageAgent,
)
from copilot.schemas.answer import (
    GroundedAnswer,
    GroundedAnswerDraft,
)
from copilot.schemas.anomaly import RunSummary
from copilot.schemas.anomaly_tools import (
    GetRunSummaryOutput,
)
from copilot.schemas.chunk import SourceChunk
from copilot.schemas.query import QueryRequest
from copilot.schemas.retrieval import ScoredChunk
from copilot.schemas.triage import (
    TriageReport,
    TriageReportDraft,
    TriageRequest,
)
from copilot.tools.anomaly import (
    AnomalyOperationalTools,
)


def install_span_recorder(
    monkeypatch,
    module,
) -> list[tuple[str, dict[str, object]]]:
    spans: list[
        tuple[str, dict[str, object]]
    ] = []

    @contextmanager
    def fake_trace_span(
        event: str,
        **attributes: object,
    ):
        spans.append(
            (
                event,
                attributes,
            )
        )
        yield

    monkeypatch.setattr(
        module,
        "trace_span",
        fake_trace_span,
    )

    return spans


def make_chunk() -> ScoredChunk:
    return ScoredChunk(
        chunk=SourceChunk(
            chunk_id="chunk-1",
            source_id="source-1",
            project_name="anomalyops",
            source_type="python",
            source_path="api.py",
            chunk_index=0,
            content="Private retrieved content.",
            start_line=10,
            end_line=20,
        ),
        score=0.9,
    )


def make_zero_alert_summary() -> RunSummary:
    return RunSummary(
        run_id=42,
        total_predictions=100,
        total_anomalies_predicted=0,
        total_row_alerts=0,
        total_alert_events=0,
        critical_alert_events=0,
        warning_alert_events=0,
        info_alert_events=0,
        machines_with_alerts=0,
        max_anomaly_score=None,
        mean_anomaly_score=None,
    )


def make_no_alerts_report() -> TriageReport:
    return TriageReport(
        run_id=42,
        status="no_alerts",
        run_summary=make_zero_alert_summary(),
        findings=[],
        evidence=[],
    )


def trace_payloads(caplog) -> list[dict[str, object]]:
    return [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == TRACE_LOGGER_NAME
    ]


def test_http_middleware_preserves_valid_request_id(
    caplog,
):
    app = app_module.create_app(
        settings=ApiSettings(
            anomaly_api_base_url=None,
        )
    )

    with caplog.at_level(
        logging.INFO,
        logger=TRACE_LOGGER_NAME,
    ):
        with TestClient(app) as client:
            response = client.get(
                "/health",
                headers={
                    "X-Request-ID": "request-123",
                },
            )

    assert response.status_code == 200
    assert response.headers[
        "X-Request-ID"
    ] == "request-123"

    payload = trace_payloads(caplog)[-1]

    assert payload["event"] == "http.request"
    assert payload["request_id"] == "request-123"
    assert payload["method"] == "GET"
    assert payload["path"] == "/health"
    assert payload["status"] == "ok"
    assert payload["status_code"] == 200


def test_http_middleware_replaces_invalid_request_id():
    app = app_module.create_app(
        settings=ApiSettings(
            anomaly_api_base_url=None,
        )
    )

    with TestClient(app) as client:
        response = client.get(
            "/health",
            headers={
                "X-Request-ID": (
                    "invalid request id"
                ),
            },
        )

    generated_id = response.headers[
        "X-Request-ID"
    ]

    assert generated_id != "invalid request id"
    assert len(generated_id) == 32


def test_http_middleware_generates_request_id_when_missing():
    app = app_module.create_app(
        settings=ApiSettings(
            anomaly_api_base_url=None,
        )
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert len(
        response.headers["X-Request-ID"]
    ) == 32


def test_http_trace_does_not_log_query_string(
    caplog,
):
    app = app_module.create_app(
        settings=ApiSettings(
            anomaly_api_base_url=None,
        )
    )

    with caplog.at_level(
        logging.INFO,
        logger=TRACE_LOGGER_NAME,
    ):
        with TestClient(app) as client:
            response = client.get(
                "/health?token=super-secret-token"
            )

    assert response.status_code == 200

    messages = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == TRACE_LOGGER_NAME
    )

    assert "super-secret-token" not in messages
    assert '"path": "/health"' in messages


def test_http_error_response_includes_request_id(
    caplog,
):
    app = app_module.create_app(
        settings=ApiSettings(
            anomaly_api_base_url=None,
        )
    )

    with caplog.at_level(
        logging.INFO,
        logger=TRACE_LOGGER_NAME,
    ):
        with TestClient(app) as client:
            response = client.post(
                "/triage",
                json={
                    "run_id": 42,
                },
            )

    assert response.status_code == 500
    assert "X-Request-ID" in response.headers

    payload = trace_payloads(caplog)[-1]

    assert payload["status"] == "error"
    assert payload["status_code"] == 500


def test_unexpected_error_is_sanitized_and_traced(
    caplog,
):
    app = app_module.create_app(
        settings=ApiSettings(
            anomaly_api_base_url=None,
        )
    )

    @app.get("/unexpected-error")
    def unexpected_error():
        raise RuntimeError("super-secret-error")

    with caplog.at_level(
        logging.INFO,
        logger=TRACE_LOGGER_NAME,
    ):
        with TestClient(
            app,
            raise_server_exceptions=False,
        ) as client:
            response = client.get(
                "/unexpected-error",
                headers={
                    "X-Request-ID": "request-123",
                },
            )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Internal server error.",
    }
    assert response.headers["X-Request-ID"] == "request-123"

    payload = trace_payloads(caplog)[-1]
    assert payload["status"] == "error"
    assert payload["error_type"] == "RuntimeError"
    assert "super-secret-error" not in caplog.text


def test_request_id_context_is_reset_after_request():
    app = app_module.create_app(
        settings=ApiSettings(
            anomaly_api_base_url=None,
        )
    )

    with TestClient(app) as client:
        response = client.get(
            "/health",
            headers={
                "X-Request-ID": "request-123",
            },
        )

    assert response.status_code == 200
    assert get_request_id() is None


def test_triage_route_emits_agent_span(
    monkeypatch,
):
    spans = install_span_recorder(
        monkeypatch,
        app_module,
    )

    app = app_module.create_app(
        settings=ApiSettings(
            anomaly_api_base_url=None,
        )
    )

    agent = Mock()
    agent.provider_name = "fake"
    agent.model_name = "fake-triage"
    agent.triage.return_value = (
        make_no_alerts_report()
    )

    app.state.triage_agent = agent

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={
                "run_id": 42,
                "max_events": 3,
            },
        )

    assert response.status_code == 200

    assert spans == [
        (
            "triage.agent",
            {
                "provider": "fake",
                "model": "fake-triage",
                "max_events": 3,
                "explicit_run": True,
            },
        )
    ]


def test_query_service_emits_retrieval_and_answer_spans(
    monkeypatch,
):
    spans = install_span_recorder(
        monkeypatch,
        query_service_module,
    )

    selected_chunks = [make_chunk()]

    monkeypatch.setattr(
        query_service_module,
        "retrieve_chunks_for_query",
        Mock(
            return_value=selected_chunks
        ),
    )

    monkeypatch.setattr(
        query_service_module,
        "build_grounded_answer",
        Mock(
            return_value=GroundedAnswer(
                answer="Supported answer [1]",
                citations=[],
                confidence=0.9,
                refusal_reason=None,
            )
        ),
    )

    generator = SimpleNamespace(
        provider_name="fake",
        model_name="fake-answer-model",
    )

    response = query_service_module.query_service(
        settings=ApiSettings(
            retrieval_backend="manifest",
        ),
        query_request=QueryRequest(
            query="Private user query",
            top_k=4,
            min_score=0.2,
        ),
        grounded_answer_generator=generator,
    )

    assert response.answer == "Supported answer [1]"

    assert spans == [
        (
            "query.retrieval",
            {
                "backend": "manifest",
                "top_k": 4,
            },
        ),
        (
            "query.answer",
            {
                "provider": "fake",
                "model": "fake-answer-model",
                "context_count": 1,
            },
        ),
    ]

    assert "Private user query" not in repr(
        spans
    )
    assert "Private retrieved content" not in repr(
        spans
    )


def test_openai_embedding_request_emits_safe_span(
    monkeypatch,
):
    spans = install_span_recorder(
        monkeypatch,
        embeddings_module,
    )

    client = Mock()
    client.embeddings.create.return_value = (
        SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=0,
                    embedding=[0.1, 0.2],
                )
            ]
        )
    )

    provider = OpenAIEmbeddingProvider(
        model_name="embedding-test",
        dimensions=2,
        client=client,
    )

    result = provider.embed_query(
        "Private embedding input"
    )

    assert result == [0.1, 0.2]

    assert spans == [
        (
            "provider.request",
            {
                "provider": "openai",
                "model": "embedding-test",
                "operation": "embed_query",
                "input_count": 1,
                "dimensions": 2,
            },
        )
    ]

    assert "Private embedding input" not in repr(
        spans
    )


def test_openai_answer_request_emits_safe_span(
    monkeypatch,
):
    spans = install_span_recorder(
        monkeypatch,
        answers_module,
    )

    client = Mock()
    client.responses.parse.return_value = (
        SimpleNamespace(
            output_parsed=GroundedAnswerDraft(
                answer="Supported answer [1]",
                citation_ids=[1],
                refusal_reason=None,
            )
        )
    )

    generator = OpenAIGroundedAnswerGenerator(
        model_name="answer-test",
        client=client,
    )

    result = generator.generate(
        query="Private question",
        context=[make_chunk()],
    )

    assert result.answer == "Supported answer [1]"

    assert spans == [
        (
            "provider.request",
            {
                "provider": "openai",
                "model": "answer-test",
                "operation": "grounded_answer",
                "context_count": 1,
            },
        )
    ]

    span_text = repr(spans)

    assert "Private question" not in span_text
    assert "Private retrieved content" not in span_text
    assert "Supported answer" not in span_text


def test_openai_triage_emits_provider_and_tool_spans(
    monkeypatch,
):
    spans = install_span_recorder(
        monkeypatch,
        triage_module,
    )

    function_call = SimpleNamespace(
        type="function_call",
        name="get_run_summary",
        arguments='{"run_id": 42}',
        call_id="summary-call",
    )

    client = Mock()
    client.responses.parse.side_effect = [
        SimpleNamespace(
            output=[function_call],
            output_parsed=None,
        ),
        SimpleNamespace(
            output=[],
            output_parsed=TriageReportDraft(
                status="no_alerts",
                findings=[],
                refusal_reason=None,
            ),
        ),
    ]

    tools = Mock(
        spec=AnomalyOperationalTools
    )
    tools.get_run_summary.return_value = (
        GetRunSummaryOutput(
            summary=make_zero_alert_summary()
        )
    )

    agent = OpenAITriageAgent(
        model_name="triage-test",
        client=client,
        tools=tools,
    )

    report = agent.triage(
        TriageRequest(
            run_id=42,
            max_events=5,
        )
    )

    assert report.status == "no_alerts"

    assert spans == [
        (
            "provider.request",
            {
                "provider": "openai",
                "model": "triage-test",
                "operation": "triage",
                "tool_round": 1,
            },
        ),
        (
            "triage.tool",
            {
                "provider": "openai",
                "model": "triage-test",
                "tool_name": "get_run_summary",
            },
        ),
        (
            "provider.request",
            {
                "provider": "openai",
                "model": "triage-test",
                "operation": "triage",
                "tool_round": 2,
            },
        ),
    ]

    span_text = repr(spans)

    assert '{"run_id": 42}' not in span_text
    assert "total_predictions" not in span_text
