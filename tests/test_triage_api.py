from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from copilot.api.app import create_app
from copilot.api.settings import ApiSettings
from copilot.providers.errors import (
    InvalidTriageAgentResponseError,
    TriageAgentProviderError,
    TriageAgentResourceNotFoundError,
    TriageAgentToolError,
)
from copilot.schemas.anomaly import RunSummary
from copilot.schemas.triage import TriageReport


ANOMALY_API_BASE_URL = "http://anomaly-api.test"


def make_settings(
    anomaly_api_base_url: str | None = ANOMALY_API_BASE_URL,
) -> ApiSettings:
    return ApiSettings(
        anomaly_api_base_url=anomaly_api_base_url,
    )


def make_run_summary(
    *,
    run_id: int = 42,
    total_alert_events: int = 2,
) -> dict[str, object]:
    has_alerts = total_alert_events > 0

    return {
        "run_id": run_id,
        "total_predictions": 100,
        "total_anomalies_predicted": 10 if has_alerts else 0,
        "total_row_alerts": 8 if has_alerts else 0,
        "total_alert_events": total_alert_events,
        "critical_alert_events": 1 if has_alerts else 0,
        "warning_alert_events": 1 if total_alert_events > 1 else 0,
        "info_alert_events": 0,
        "machines_with_alerts": 2 if has_alerts else 0,
        "max_anomaly_score": 0.97 if has_alerts else None,
        "mean_anomaly_score": 0.74 if has_alerts else None,
    }


def make_alert_event(
    *,
    event_id: int,
    severity: str,
    anomaly_score: float | None,
    run_id: int = 42,
    machine_id: int = 3,
    sensor: str = "temperature",
    anomaly_type: str | None = "spike",
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "event_id": event_id,
        "machine_id": machine_id,
        "sensor": sensor,
        "anomaly_type": anomaly_type,
        "start_step": 100,
        "end_step": 110,
        "duration": 11,
        "alert_count": 4,
        "max_severity": severity,
        "max_severity_reason": "High anomaly score",
        "max_anomaly_score": anomaly_score,
        "mean_anomaly_score": anomaly_score,
        "min_sensor_value": 70.0,
        "max_sensor_value": 105.0,
        "first_reason": "Sensor reading outside expected range",
        "status": "open",
        "real_value": 1,
    }


def make_row_alert(
    *,
    alert_id: int,
    step: int,
    run_id: int = 42,
    machine_id: int = 3,
    sensor: str = "temperature",
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "alert_id": alert_id,
        "step": step,
        "machine_id": machine_id,
        "sensor": sensor,
        "sensor_value": 103.5,
        "prediction": 1,
        "anomaly_score": 0.96,
        "severity": "critical",
        "alert_type": "anomaly",
        "reason": "Sensor reading outside expected range",
        "status": "open",
        "anomaly_type": "spike",
        "real_value": 1,
    }


def create_test_app(
    handler,
):
    return create_app(
        settings=make_settings(),
        anomaly_transport=httpx.MockTransport(handler),
    )


def test_triage_with_explicit_run_id_returns_completed_report():
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)

        if request.url.path == "/runs/42/summary":
            return httpx.Response(
                200,
                json=make_run_summary(total_alert_events=2),
            )

        if request.url.path == "/runs/42/events":
            severity = request.url.params.get("severity")

            if severity == "critical":
                return httpx.Response(
                    200,
                    json=[
                        make_alert_event(
                            event_id=7,
                            severity="critical",
                            anomaly_score=0.97,
                        )
                    ],
                )

            if severity == "warning":
                return httpx.Response(
                    200,
                    json=[
                        make_alert_event(
                            event_id=8,
                            severity="warning",
                            anomaly_score=0.82,
                            machine_id=4,
                            sensor="pressure",
                            anomaly_type="drift",
                        )
                    ],
                )

        if request.url.path == "/runs/42/events/7/alerts":
            return httpx.Response(
                200,
                json=[
                    make_row_alert(
                        alert_id=15,
                        step=105,
                    )
                ],
            )

        if request.url.path == "/runs/42/events/8/alerts":
            return httpx.Response(
                200,
                json=[
                    make_row_alert(
                        alert_id=16,
                        step=120,
                        machine_id=4,
                        sensor="pressure",
                    )
                ],
            )

        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    app = create_test_app(handler)

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={
                "run_id": 42,
                "max_events": 5,
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["run_id"] == 42
    assert payload["status"] == "completed"
    assert payload["run_summary"]["run_id"] == 42
    assert len(payload["findings"]) == 2
    assert len(payload["evidence"]) == 2

    assert "/runs/latest" not in requested_paths


def test_triage_without_run_id_resolves_latest_run():
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)

        if request.url.path == "/runs/latest":
            return httpx.Response(
                200,
                json={"run_id": 42},
            )

        if request.url.path == "/runs/42/summary":
            return httpx.Response(
                200,
                json=make_run_summary(total_alert_events=0),
            )

        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    app = create_test_app(handler)

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={
                "max_events": 5,
            },
        )

    assert response.status_code == 200
    assert response.json()["run_id"] == 42
    assert requested_paths == [
        "/runs/latest",
        "/runs/42/summary",
    ]


def test_triage_with_zero_alert_events_returns_no_alerts():
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)

        if request.url.path == "/runs/42/summary":
            return httpx.Response(
                200,
                json=make_run_summary(total_alert_events=0),
            )

        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    app = create_test_app(handler)

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={
                "run_id": 42,
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "no_alerts"
    assert payload["findings"] == []
    assert payload["evidence"] == []
    assert requested_paths == ["/runs/42/summary"]


def test_triage_requests_critical_and_warning_events_with_filters():
    event_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/runs/42/summary":
            return httpx.Response(
                200,
                json=make_run_summary(total_alert_events=2),
            )

        if request.url.path == "/runs/42/events":
            event_requests.append(request)

            return httpx.Response(
                200,
                json=[],
            )

        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    app = create_test_app(handler)

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={
                "run_id": 42,
                "max_events": 4,
            },
        )

    assert response.status_code == 200
    assert len(event_requests) == 2

    critical_params = dict(event_requests[0].url.params)
    warning_params = dict(event_requests[1].url.params)

    assert critical_params == {
        "limit": "4",
        "offset": "0",
        "severity": "critical",
    }
    assert warning_params == {
        "limit": "4",
        "offset": "0",
        "severity": "warning",
    }


def test_triage_requests_alerts_for_each_selected_event():
    requested_alert_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/runs/42/summary":
            return httpx.Response(
                200,
                json=make_run_summary(total_alert_events=2),
            )

        if request.url.path == "/runs/42/events":
            severity = request.url.params.get("severity")

            if severity == "critical":
                return httpx.Response(
                    200,
                    json=[
                        make_alert_event(
                            event_id=10,
                            severity="critical",
                            anomaly_score=0.95,
                        )
                    ],
                )

            return httpx.Response(
                200,
                json=[
                    make_alert_event(
                        event_id=11,
                        severity="warning",
                        anomaly_score=0.85,
                    )
                ],
            )

        if request.url.path in {
            "/runs/42/events/10/alerts",
            "/runs/42/events/11/alerts",
        }:
            requested_alert_paths.append(request.url.path)

            return httpx.Response(
                200,
                json=[],
            )

        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    app = create_test_app(handler)

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={
                "run_id": 42,
                "max_events": 5,
            },
        )

    assert response.status_code == 200
    assert requested_alert_paths == [
        "/runs/42/events/10/alerts",
        "/runs/42/events/11/alerts",
    ]


def test_triage_findings_reference_existing_evidence():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/runs/42/summary":
            return httpx.Response(
                200,
                json=make_run_summary(total_alert_events=1),
            )

        if request.url.path == "/runs/42/events":
            if request.url.params.get("severity") == "critical":
                return httpx.Response(
                    200,
                    json=[
                        make_alert_event(
                            event_id=7,
                            severity="critical",
                            anomaly_score=0.97,
                        )
                    ],
                )

            return httpx.Response(
                200,
                json=[],
            )

        if request.url.path == "/runs/42/events/7/alerts":
            return httpx.Response(
                200,
                json=[
                    make_row_alert(
                        alert_id=15,
                        step=105,
                    )
                ],
            )

        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    app = create_test_app(handler)

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={
                "run_id": 42,
            },
        )

    assert response.status_code == 200

    payload = response.json()

    evidence_ids = {
        evidence["evidence_id"]
        for evidence in payload["evidence"]
    }

    assert evidence_ids == {"event-7"}

    finding = payload["findings"][0]

    assert finding["finding_id"] == "finding-7"
    assert finding["evidence_ids"] == ["event-7"]
    assert set(finding["evidence_ids"]).issubset(evidence_ids)


def test_triage_limits_findings_and_evidence_to_max_events():
    requested_alert_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/runs/42/summary":
            return httpx.Response(
                200,
                json=make_run_summary(total_alert_events=4),
            )

        if request.url.path == "/runs/42/events":
            severity = request.url.params.get("severity")

            if severity == "critical":
                return httpx.Response(
                    200,
                    json=[
                        make_alert_event(
                            event_id=1,
                            severity="critical",
                            anomaly_score=0.99,
                        ),
                        make_alert_event(
                            event_id=2,
                            severity="critical",
                            anomaly_score=0.90,
                        ),
                        make_alert_event(
                            event_id=3,
                            severity="critical",
                            anomaly_score=0.80,
                        ),
                    ],
                )

            return httpx.Response(
                200,
                json=[
                    make_alert_event(
                        event_id=4,
                        severity="warning",
                        anomaly_score=0.95,
                    )
                ],
            )

        if request.url.path.endswith("/alerts"):
            requested_alert_paths.append(request.url.path)

            return httpx.Response(
                200,
                json=[],
            )

        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    app = create_test_app(handler)

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={
                "run_id": 42,
                "max_events": 2,
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert len(payload["findings"]) == 2
    assert len(payload["evidence"]) == 2

    assert [
        evidence["event"]["event_id"]
        for evidence in payload["evidence"]
    ] == [1, 2]

    assert requested_alert_paths == [
        "/runs/42/events/1/alerts",
        "/runs/42/events/2/alerts",
    ]


@pytest.mark.parametrize("run_id", [0, -1])
def test_triage_rejects_invalid_run_id(
    run_id: int,
):
    app = create_app(
        settings=make_settings(anomaly_api_base_url=None),
    )

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={
                "run_id": run_id,
            },
        )

    assert response.status_code == 422


@pytest.mark.parametrize("max_events", [0, -1, 21])
def test_triage_rejects_invalid_max_events(
    max_events: int,
):
    app = create_app(
        settings=make_settings(anomaly_api_base_url=None),
    )

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={
                "run_id": 42,
                "max_events": max_events,
            },
        )

    assert response.status_code == 422


def test_triage_without_anomaly_api_base_url_returns_500():
    app = create_app(
        settings=make_settings(anomaly_api_base_url=None),
    )

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={
                "run_id": 42,
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Anomaly API base URL is not configured.",
    }


def test_triage_maps_upstream_not_found_to_404():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/runs/999/summary":
            return httpx.Response(
                404,
                json={"detail": "Run not found."},
            )

        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    app = create_test_app(handler)

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={
                "run_id": 999,
            },
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "The requested run was not found.",
    }


def test_triage_maps_upstream_server_error_to_502():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/runs/42/summary":
            return httpx.Response(
                500,
                json={"detail": "Internal server error."},
            )

        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    app = create_test_app(handler)

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={
                "run_id": 42,
            },
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Anomaly API request failed.",
    }


def test_triage_maps_connection_failure_to_502():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "Connection failed.",
            request=request,
        )

    app = create_test_app(handler)

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={
                "run_id": 42,
            },
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Anomaly API request failed.",
    }


def test_triage_maps_invalid_upstream_payload_to_502():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/runs/42/summary":
            return httpx.Response(
                200,
                json={
                    "run_id": 42,
                },
            )

        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    app = create_test_app(handler)

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={
                "run_id": 42,
            },
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Anomaly API request failed.",
    }


def test_anomaly_api_client_closes_when_test_client_exits():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/runs/42/summary":
            return httpx.Response(
                200,
                json=make_run_summary(total_alert_events=0),
            )

        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    app = create_test_app(handler)

    close_spy = Mock(
        wraps=app.state.anomaly_client.close,
    )
    app.state.anomaly_client.close = close_spy

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={
                "run_id": 42,
            },
        )

        assert response.status_code == 200
        close_spy.assert_not_called()

    close_spy.assert_called_once_with()
    
    
def make_no_alerts_report() -> TriageReport:
    return TriageReport(
        run_id=42,
        status="no_alerts",
        run_summary=RunSummary(
            **make_run_summary(
                run_id=42,
                total_alert_events=0,
            )
        ),
        findings=[],
        evidence=[],
    )


def test_app_configures_triage_agent():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    app = create_test_app(handler)

    assert app.state.triage_agent is not None
    assert not hasattr(app.state, "triage_service")


def test_triage_route_delegates_to_configured_agent():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    app = create_test_app(handler)

    agent = Mock()
    agent.triage.return_value = make_no_alerts_report()
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

    triage_request = agent.triage.call_args.args[0]

    assert triage_request.run_id == 42
    assert triage_request.max_events == 3


def test_triage_maps_agent_resource_not_found_to_404():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    app = create_test_app(handler)

    agent = Mock()
    agent.triage.side_effect = (
        TriageAgentResourceNotFoundError(
            "missing"
        )
    )
    app.state.triage_agent = agent

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={"run_id": 42},
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "The requested run was not found.",
    }


def test_triage_maps_invalid_agent_response_to_502():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    app = create_test_app(handler)

    agent = Mock()
    agent.triage.side_effect = (
        InvalidTriageAgentResponseError(
            "invalid"
        )
    )
    app.state.triage_agent = agent

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={"run_id": 42},
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": (
            "Triage agent returned an invalid response."
        ),
    }


def test_triage_maps_agent_tool_error_to_502():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    app = create_test_app(handler)

    agent = Mock()
    agent.triage.side_effect = (
        TriageAgentToolError(
            "failed"
        )
    )
    app.state.triage_agent = agent

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={"run_id": 42},
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Anomaly API request failed.",
    }


def test_triage_maps_agent_provider_error_to_502():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    app = create_test_app(handler)

    agent = Mock()
    agent.triage.side_effect = (
        TriageAgentProviderError(
            "failed"
        )
    )
    app.state.triage_agent = agent

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={"run_id": 42},
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Triage agent request failed.",
    }