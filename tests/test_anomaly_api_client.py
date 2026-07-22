import httpx
import pytest

from pydantic import ValidationError

from copilot.clients.anomaly_api import (
    AnomalyApiClient,
    AnomalyResourceNotFoundError,
    AnomalyApiError,
    InvalidAnomalyApiResponseError,
)
from copilot.schemas.anomaly import (
    LatestRun,
    RunSummary,
    AlertEvent,
    RowAlert,
)


def handle_get_latest_run(request):
    assert request.method == "GET"
    assert request.url.path == "/runs/latest"
    
    return httpx.Response(
        200,
        json={
            "run_id": 5,
        }
    )


def handle_get_run_summary(request):
    assert request.method == "GET"
    assert request.url.path == "/runs/5/summary"
    
    return httpx.Response(
        200,
        json={
            "run_id": 5,
            "total_predictions": 23409,
            "total_anomalies_predicted": 12987,
            "total_row_alerts": 2341,
            "total_alert_events": 1245,
            "critical_alert_events": 356,
            "warning_alert_events": 581,
            "info_alert_events": 790,
            "machines_with_alerts": 10,
            "max_anomaly_score": 0.8,
            "mean_anomaly_score": 0.2,
        }
    )
    
    
def handle_get_run_summary_score_none(request):
    assert request.method == "GET"
    assert request.url.path == "/runs/5/summary"
    
    return httpx.Response(
        200,
        json={
            "run_id": 5,
            "total_predictions": 23409,
            "total_anomalies_predicted": 12987,
            "total_row_alerts": 2341,
            "total_alert_events": 1245,
            "critical_alert_events": 356,
            "warning_alert_events": 581,
            "info_alert_events": 790,
            "machines_with_alerts": 10,
            "max_anomaly_score": None,
            "mean_anomaly_score": None,
        }
    )
    
    
def handle_list_alert_events_filtered(request):
    assert request.method == "GET"
    assert request.url.path == "/runs/5/events"
    assert request.url.params["severity"] == "critical"
    assert request.url.params["sensor"] == "temperature"
    assert request.url.params["anomaly_type"] == "spike"
    assert request.url.params["limit"] == "50"
    assert request.url.params["offset"] == "10"
    
    return httpx.Response(
        200,
        json=[{
            "run_id": 5,
            "event_id": 1,
            "machine_id": 5,
            "sensor": "temperature",
            "anomaly_type": "spike",
            "start_step": 3289,
            "end_step": 3290,
            "duration": 1,
            "alert_count": 1,
            "max_severity": "critical",
            "max_severity_reason": None,
            "max_anomaly_score": 0.9,
            "mean_anomaly_score": 0.3,
            "min_sensor_value": 10.0,
            "max_sensor_value": 100.0,
            "first_reason": None,
            "status": None,
            "real_value": None,
        }],
    )
    
    
def handle_list_alert_events_unfiltered(request):
    assert request.method == "GET"
    assert request.url.path == "/runs/5/events"
    assert "severity" not in request.url.params
    assert "sensor" not in request.url.params
    assert "anomaly_type" not in request.url.params
    assert request.url.params["limit"] == "100"
    assert request.url.params["offset"] == "0"
    
    return httpx.Response(
        200,
        json=[{
            "run_id": 5,
            "event_id": 1,
            "machine_id": 5,
            "sensor": "temperature",
            "anomaly_type": "spike",
            "start_step": 3289,
            "end_step": 3290,
            "duration": 1,
            "alert_count": 1,
            "max_severity": "critical",
            "max_severity_reason": None,
            "max_anomaly_score": 0.9,
            "mean_anomaly_score": 0.3,
            "min_sensor_value": 10.0,
            "max_sensor_value": 100.0,
            "first_reason": None,
            "status": None,
            "real_value": None,
        }],
    )
    
    
def handle_get_event_alerts(request):
    assert request.method == "GET"
    assert request.url.path == "/runs/5/events/1/alerts"
    
    return httpx.Response(
        200,
        json=[{
            "run_id": 5,
            "alert_id": 1,
            "step": 50,
            "machine_id": 1,
            "sensor": "temperature",
            "sensor_value": 10,
            "prediction": 1,
            "anomaly_score": 0.7,
            "severity": "critical",
            "alert_type": "critical",
            "reason": None,
            "status": None,
            "anomaly_type": "spike",
            "real_value": None,
        }]
    )


def handle_get_event_alerts_empty(request):
    assert request.method == "GET"
    assert request.url.path == "/runs/5/events/1/alerts"
    
    return httpx.Response(
        200,
        json=[],
    )
    
def handler_404(request):
    assert request.method == "GET"
    assert request.url.path == "/runs/999/summary"
    
    return httpx.Response(404)
    

def handler_500(request):
    assert request.method == "GET"
    assert request.url.path == "/runs/5/summary"
    
    return httpx.Response(500)


def handler_connect_error(request):
    raise httpx.ConnectError(
        "Connection failed.",
        request=request,
    )
    
    
def handle_timeout(request):
    raise httpx.ReadTimeout(
        "Request timed out.",
        request=request,
    )
    
    
def handle_invalid_json(request):
    assert request.method == "GET"
    assert request.url.path == "/runs/latest"
    
    return httpx.Response(
        200,
        text="invalid json"
    )
    
    
def handle_invalid_schema(request):
    assert request.method == "GET"
    assert request.url.path == "/runs/latest"
    
    return httpx.Response(
        200,
        json={
            "run_id": "not an integer",
        }
    )
    
    
def handle_list_alert_events_json_not_list(request):
    return httpx.Response(
        200,
        json={
            "unexpected": "object"
        }
    )
    
    
def handle_get_event_alerts_json_null(request):
    return httpx.Response(
        200,
        json=None,
    )


def test_get_latest_run_is_valid():
    transport = httpx.MockTransport(handle_get_latest_run)
    with AnomalyApiClient(
        "http://anomaly-api.test",
        transport=transport,
    ) as client:
        response = client.get_latest_run()
    
    assert isinstance(response, LatestRun)
    assert response.run_id == 5
    
    
def test_get_run_summary_returns_typed_summary():
    transport = httpx.MockTransport(handle_get_run_summary)
    with AnomalyApiClient(
        "http://anomaly-api.test",
        transport=transport,
    ) as client:
        response = client.get_run_summary(5)
        
    assert isinstance(response, RunSummary)
    assert response.total_alert_events == 1245
    assert response.max_anomaly_score == 0.8
    
    
def test_get_run_summary_anomaly_score_none_returns_none():
    transport = httpx.MockTransport(handle_get_run_summary_score_none)
    with AnomalyApiClient(
        "http://anomaly-api.test",
        transport=transport,
    ) as client:
        response = client.get_run_summary(5)
        
    assert response.mean_anomaly_score is None
    assert response.max_anomaly_score is None
    
    
def test_list_alert_events_filtered():
    transport = httpx.MockTransport(handle_list_alert_events_filtered)
    with AnomalyApiClient(
        "http://anomaly-api.test",
        transport=transport,
    ) as client:
        response = client.list_alert_events(
            run_id=5,
            severity="critical",
            sensor="temperature",
            anomaly_type="spike",
            limit=50,
            offset=10,
        )
        
    assert isinstance(response, list)
    assert isinstance(response[0], AlertEvent)
    assert response[0].event_id == 1
    assert response[0].max_severity == "critical"
    
    
def test_list_alert_events_unfiltered():
    transport = httpx.MockTransport(handle_list_alert_events_unfiltered)
    with AnomalyApiClient(
        "http://anomaly-api.test",
        transport=transport,
    ) as client:
        response = client.list_alert_events(
            run_id=5,
        )
        
    assert isinstance(response, list)
    assert isinstance(response[0], AlertEvent)
    assert response[0].event_id == 1
    assert response[0].max_severity == "critical"
    
    
def test_get_event_alerts():
    transport = httpx.MockTransport(handle_get_event_alerts)
    with AnomalyApiClient(
        "http://anomaly-api.test",
        transport=transport,
    ) as client:
        response = client.get_event_alerts(
            run_id=5,
            event_id=1,
        )
        
    assert isinstance(response, list)
    assert isinstance(response[0], RowAlert)
    assert response[0].alert_id == 1
    assert response[0].severity == "critical"
    
    
def test_get_event_alerts_returns_empty_list():
    transport = httpx.MockTransport(handle_get_event_alerts_empty)
    with AnomalyApiClient(
        "http://anomaly-api.test",
        transport=transport,
    ) as client:
        response = client.get_event_alerts(
            run_id=5,
            event_id=1,
        )
    
    assert response == []
    
    
def test_context_manager_cleanup():
    transport = httpx.MockTransport(handle_get_event_alerts)
    client = AnomalyApiClient(
        "http://anomaly-api.test",
        transport=transport,
    )
    
    assert client._client.is_closed is False
    
    with client as entered_client:
        assert entered_client is client
    
    assert client._client.is_closed
    
    
def test_exceptions_are_not_suppressed():
    transport = httpx.MockTransport(handle_get_event_alerts)
    client = AnomalyApiClient(
        "http://anomaly-api.test",
        transport=transport,
    )
    
    with pytest.raises(RuntimeError):
        with client:
            raise RuntimeError()
    
    assert client._client.is_closed
    
    
def test_get_run_summary_404():
    transport = httpx.MockTransport(handler_404)
    with pytest.raises(AnomalyResourceNotFoundError):
        with AnomalyApiClient(
            "http://anomaly-api.test",
            transport=transport,
        ) as client:
            client.get_run_summary(999)
            
            
def test_get_run_summary_500():
    transport = httpx.MockTransport(handler_500)
    with pytest.raises(AnomalyApiError) as exc_info:
        with AnomalyApiClient(
            "http://anomaly-api.test",
            transport=transport,
        ) as client:
            client.get_run_summary(5)
            
    assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)
            
            
def test_get_latest_run_anomaly_api_error():
    transport = httpx.MockTransport(handler_connect_error)
    with pytest.raises(AnomalyApiError) as exc_info:
        with AnomalyApiClient(
            "http://anomaly-api.test",
            transport=transport,
        ) as client:
            client.get_latest_run()
            
    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)
    
    
def test_get_latest_run_timeout():
    transport = httpx.MockTransport(handle_timeout)
    with pytest.raises(AnomalyApiError) as exc_info:
        with AnomalyApiClient(
            "http://anomaly-api.test",
            transport=transport,
        ) as client:
            client.get_latest_run()
        
    assert isinstance(exc_info.value.__cause__, httpx.ReadTimeout)
    
    
def test_get_latest_run_invalid_json():
    transport = httpx.MockTransport(handle_invalid_json)
    with pytest.raises(InvalidAnomalyApiResponseError) as exc_info:
        with AnomalyApiClient(
            "http://anomaly-api.test",
            transport=transport,
        ) as client:
            client.get_latest_run()
            
        assert isinstance(exc_info.value.__cause__, httpx.JSONDecodeError)
        
        
def test_get_latest_run_invalid_schema():
    transport = httpx.MockTransport(handle_invalid_schema)
    with pytest.raises(InvalidAnomalyApiResponseError) as exc_info:
        with AnomalyApiClient(
            "http://anomaly-api.test",
            transport=transport,
        ) as client:
            client.get_latest_run()
    
    assert isinstance(exc_info.value.__cause__, ValidationError)
    
    
def test_list_alert_events_json_not_list():
    transport = httpx.MockTransport(handle_list_alert_events_json_not_list)
    with pytest.raises(InvalidAnomalyApiResponseError):
        with AnomalyApiClient(
            "http://anomaly-api.test",
            transport=transport,
        ) as client:
            client.list_alert_events(
                run_id=5,
            )
            
            
def test_get_event_alerts_receives_null_json():
    transport = httpx.MockTransport(handle_get_event_alerts_json_null)
    with pytest.raises(InvalidAnomalyApiResponseError):
        with AnomalyApiClient(
            "http://anomaly-api.test",
            transport=transport,
        ) as client:
            client.get_event_alerts(
                run_id=5,
                event_id=1,
            )
            
            
def fail_if_transport_called(request: httpx.Request) -> httpx.Response:
    pytest.fail(f"Transport should not be called: {request.method} {request.url}")


@pytest.mark.parametrize("base_url", ["", "   "])
def test_client_rejects_empty_base_url(base_url: str):
    with pytest.raises(
        ValueError,
        match="base_url cannot be empty or whitespace-only",
    ):
        AnomalyApiClient(base_url)


@pytest.mark.parametrize("timeout_seconds", [0, -1])
def test_client_rejects_non_positive_timeout(timeout_seconds: float):
    with pytest.raises(
        ValueError,
        match="timeout_seconds must be positive",
    ):
        AnomalyApiClient(
            "http://anomaly-api.test",
            timeout_seconds=timeout_seconds,
        )


@pytest.mark.parametrize(
    ("method_name", "kwargs", "expected_message"),
    [
        (
            "get_run_summary",
            {"run_id": 0},
            "run_id must be positive",
        ),
        (
            "get_run_summary",
            {"run_id": -1},
            "run_id must be positive",
        ),
        (
            "list_alert_events",
            {"run_id": 0},
            "run_id must be positive",
        ),
        (
            "list_alert_events",
            {"run_id": -1},
            "run_id must be positive",
        ),
        (
            "get_event_alerts",
            {"run_id": 0, "event_id": 1},
            "run_id must be positive",
        ),
        (
            "get_event_alerts",
            {"run_id": -1, "event_id": 1},
            "run_id must be positive",
        ),
        (
            "get_event_alerts",
            {"run_id": 1, "event_id": 0},
            "event_id must be positive",
        ),
        (
            "get_event_alerts",
            {"run_id": 1, "event_id": -1},
            "event_id must be positive",
        ),
        (
            "list_alert_events",
            {"run_id": 1, "limit": 0},
            "limit must be positive",
        ),
        (
            "list_alert_events",
            {"run_id": 1, "limit": -1},
            "limit must be positive",
        ),
        (
            "list_alert_events",
            {"run_id": 1, "limit": 501},
            "limit must not be greater than 500",
        ),
        (
            "list_alert_events",
            {"run_id": 1, "offset": -1},
            "offset must not be negative",
        ),
    ],
)
def test_client_rejects_invalid_method_arguments(
    method_name: str,
    kwargs: dict[str, int],
    expected_message: str,
):
    transport = httpx.MockTransport(fail_if_transport_called)

    with AnomalyApiClient(
        "http://anomaly-api.test",
        transport=transport,
    ) as client:
        method = getattr(client, method_name)

        with pytest.raises(ValueError, match=expected_message):
            method(**kwargs)