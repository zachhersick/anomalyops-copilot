from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from copilot.clients.anomaly_api import (
    AnomalyApiClient,
    AnomalyApiError,
    AnomalyResourceNotFoundError,
)
from copilot.schemas.anomaly import (
    AlertEvent,
    LatestRun,
    RowAlert,
    RunSummary,
)
from copilot.schemas.anomaly_tools import (
    GetEventAlertsInput,
    GetEventAlertsOutput,
    GetLatestRunInput,
    GetLatestRunOutput,
    GetRunSummaryInput,
    GetRunSummaryOutput,
    ListAlertEventsInput,
    ListAlertEventsOutput,
)
from copilot.tools.anomaly import (
    AnomalyOperationalTools,
    OperationalResourceNotFoundError,
    OperationalToolError,
)


def make_run_summary(run_id: int = 42) -> RunSummary:
    return RunSummary(
        run_id=run_id,
        total_predictions=100,
        total_anomalies_predicted=10,
        total_row_alerts=8,
        total_alert_events=4,
        critical_alert_events=1,
        warning_alert_events=2,
        info_alert_events=1,
        machines_with_alerts=3,
        max_anomaly_score=0.95,
        mean_anomaly_score=0.62,
    )


def make_alert_event(
    run_id: int = 42,
    event_id: int = 7,
) -> AlertEvent:
    return AlertEvent(
        run_id=run_id,
        event_id=event_id,
        machine_id=3,
        sensor="temperature",
        anomaly_type="spike",
        start_step=100,
        end_step=110,
        duration=11,
        alert_count=4,
        max_severity="critical",
        max_severity_reason="High anomaly score",
        max_anomaly_score=0.97,
        mean_anomaly_score=0.84,
        min_sensor_value=72.0,
        max_sensor_value=105.0,
        first_reason="Temperature spike",
        status="open",
        real_value=1,
    )


def make_row_alert(
    run_id: int = 42,
    alert_id: int = 15,
) -> RowAlert:
    return RowAlert(
        run_id=run_id,
        alert_id=alert_id,
        step=250,
        machine_id=3,
        sensor="temperature",
        sensor_value=103.5,
        prediction=1,
        anomaly_score=0.96,
        severity="critical",
        alert_type="anomaly",
        reason="Temperature exceeded expected range",
        status="open",
        anomaly_type="spike",
        real_value=1,
    )


def test_get_latest_run_returns_typed_output():
    client = Mock(spec=AnomalyApiClient)
    client.get_latest_run.return_value = LatestRun(run_id=42)

    tools = AnomalyOperationalTools(client)

    result = tools.get_latest_run(GetLatestRunInput())

    assert isinstance(result, GetLatestRunOutput)
    assert isinstance(result.run, LatestRun)
    assert result.run.run_id == 42
    client.get_latest_run.assert_called_once_with()


def test_get_run_summary_forwards_run_id_and_returns_typed_output():
    client = Mock(spec=AnomalyApiClient)
    client.get_run_summary.return_value = make_run_summary()

    tools = AnomalyOperationalTools(client)

    result = tools.get_run_summary(
        GetRunSummaryInput(run_id=42)
    )

    assert isinstance(result, GetRunSummaryOutput)
    assert isinstance(result.summary, RunSummary)
    assert result.summary.run_id == 42

    client.get_run_summary.assert_called_once_with(
        run_id=42,
    )


def test_list_alert_events_forwards_all_arguments():
    client = Mock(spec=AnomalyApiClient)
    client.list_alert_events.return_value = [
        make_alert_event(),
    ]

    tools = AnomalyOperationalTools(client)

    tool_input = ListAlertEventsInput(
        run_id=42,
        severity="critical",
        sensor="temperature",
        anomaly_type="spike",
        limit=25,
        offset=10,
    )

    result = tools.list_alert_events(tool_input)

    assert isinstance(result, ListAlertEventsOutput)
    assert isinstance(result.events[0], AlertEvent)
    assert result.events[0].event_id == 7

    client.list_alert_events.assert_called_once_with(
        run_id=42,
        severity="critical",
        sensor="temperature",
        anomaly_type="spike",
        limit=25,
        offset=10,
    )


def test_list_alert_events_forwards_default_arguments():
    client = Mock(spec=AnomalyApiClient)
    client.list_alert_events.return_value = []

    tools = AnomalyOperationalTools(client)

    result = tools.list_alert_events(
        ListAlertEventsInput(run_id=42)
    )

    assert isinstance(result, ListAlertEventsOutput)
    assert result.events == []

    client.list_alert_events.assert_called_once_with(
        run_id=42,
        severity=None,
        sensor=None,
        anomaly_type=None,
        limit=100,
        offset=0,
    )


def test_get_event_alerts_forwards_ids_and_returns_typed_output():
    client = Mock(spec=AnomalyApiClient)
    client.get_event_alerts.return_value = [
        make_row_alert(),
    ]

    tools = AnomalyOperationalTools(client)

    result = tools.get_event_alerts(
        GetEventAlertsInput(
            run_id=42,
            event_id=7,
        )
    )

    assert isinstance(result, GetEventAlertsOutput)
    assert isinstance(result.alerts[0], RowAlert)
    assert result.alerts[0].alert_id == 15

    client.get_event_alerts.assert_called_once_with(
        run_id=42,
        event_id=7,
    )


def test_list_alert_events_allows_empty_result():
    client = Mock(spec=AnomalyApiClient)
    client.list_alert_events.return_value = []

    tools = AnomalyOperationalTools(client)

    result = tools.list_alert_events(
        ListAlertEventsInput(run_id=42)
    )

    assert isinstance(result, ListAlertEventsOutput)
    assert result.events == []


def test_get_event_alerts_allows_empty_result():
    client = Mock(spec=AnomalyApiClient)
    client.get_event_alerts.return_value = []

    tools = AnomalyOperationalTools(client)

    result = tools.get_event_alerts(
        GetEventAlertsInput(
            run_id=42,
            event_id=7,
        )
    )

    assert isinstance(result, GetEventAlertsOutput)
    assert result.alerts == []


@pytest.mark.parametrize("run_id", [0, -1])
def test_get_run_summary_input_rejects_invalid_run_id(
    run_id: int,
):
    client = Mock(spec=AnomalyApiClient)
    tools = AnomalyOperationalTools(client)

    with pytest.raises(ValidationError):
        tool_input = GetRunSummaryInput(run_id=run_id)
        tools.get_run_summary(tool_input)

    client.get_run_summary.assert_not_called()


@pytest.mark.parametrize(
    "input_values",
    [
        {"run_id": 0},
        {"run_id": -1},
        {"run_id": 42, "limit": 0},
        {"run_id": 42, "limit": -1},
        {"run_id": 42, "limit": 501},
        {"run_id": 42, "offset": -1},
    ],
)
def test_list_alert_events_input_rejects_invalid_values(
    input_values: dict[str, int],
):
    client = Mock(spec=AnomalyApiClient)
    tools = AnomalyOperationalTools(client)

    with pytest.raises(ValidationError):
        tool_input = ListAlertEventsInput(**input_values)
        tools.list_alert_events(tool_input)

    client.list_alert_events.assert_not_called()


@pytest.mark.parametrize(
    ("run_id", "event_id"),
    [
        (0, 7),
        (-1, 7),
        (42, 0),
        (42, -1),
    ],
)
def test_get_event_alerts_input_rejects_invalid_ids(
    run_id: int,
    event_id: int,
):
    client = Mock(spec=AnomalyApiClient)
    tools = AnomalyOperationalTools(client)

    with pytest.raises(ValidationError):
        tool_input = GetEventAlertsInput(
            run_id=run_id,
            event_id=event_id,
        )
        tools.get_event_alerts(tool_input)

    client.get_event_alerts.assert_not_called()


def test_tool_maps_not_found_error():
    client = Mock(spec=AnomalyApiClient)
    original_error = AnomalyResourceNotFoundError(
        "Run was not found."
    )
    client.get_run_summary.side_effect = original_error

    tools = AnomalyOperationalTools(client)

    with pytest.raises(
        OperationalResourceNotFoundError
    ) as exc_info:
        tools.get_run_summary(
            GetRunSummaryInput(run_id=42)
        )

    assert exc_info.value.__cause__ is original_error


def test_tool_maps_general_api_error():
    client = Mock(spec=AnomalyApiClient)
    original_error = AnomalyApiError(
        "Anomaly API request failed."
    )
    client.get_latest_run.side_effect = original_error

    tools = AnomalyOperationalTools(client)

    with pytest.raises(OperationalToolError) as exc_info:
        tools.get_latest_run(GetLatestRunInput())

    assert exc_info.value.__cause__ is original_error