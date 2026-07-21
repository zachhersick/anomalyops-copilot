import httpx
import pytest

from copilot.clients.anomaly_api import (
    AnomalyApiClient,
    AnomalyResourceNotFoundError,
    AnomalyApiError,
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