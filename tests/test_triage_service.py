from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from copilot.schemas.anomaly import (
    AlertEvent,
    LatestRun,
    RowAlert,
    RunSummary,
)
from copilot.schemas.anomaly_tools import (
    GetLatestRunInput,
    GetEventAlertsOutput,
    GetLatestRunOutput,
    GetRunSummaryOutput,
    ListAlertEventsOutput,
)
from copilot.schemas.triage import TriageRequest
from copilot.services.triage import (
    TriageRunNotFoundError,
    TriageService,
    TriageServiceError,
)
from copilot.tools.anomaly import (
    AnomalyOperationalTools,
    OperationalResourceNotFoundError,
    OperationalToolError,
)


def make_run_summary(
    run_id: int = 42,
    total_alert_events: int = 3,
) -> RunSummary:
    return RunSummary(
        run_id=run_id,
        total_predictions=100,
        total_anomalies_predicted=10,
        total_row_alerts=8,
        total_alert_events=total_alert_events,
        critical_alert_events=2 if total_alert_events else 0,
        warning_alert_events=1 if total_alert_events else 0,
        info_alert_events=0,
        machines_with_alerts=2 if total_alert_events else 0,
        max_anomaly_score=0.95 if total_alert_events else None,
        mean_anomaly_score=0.72 if total_alert_events else None,
    )


def make_alert_event(
    event_id: int,
    severity: str,
    score: float | None,
    run_id: int = 42,
    machine_id: int = 3,
    sensor: str = "temperature",
    anomaly_type: str | None = "spike",
) -> AlertEvent:
    return AlertEvent(
        run_id=run_id,
        event_id=event_id,
        machine_id=machine_id,
        sensor=sensor,
        anomaly_type=anomaly_type,
        start_step=100,
        end_step=110,
        duration=11,
        alert_count=4,
        max_severity=severity,
        max_severity_reason="High anomaly score",
        max_anomaly_score=score,
        mean_anomaly_score=score,
        min_sensor_value=70.0,
        max_sensor_value=105.0,
        first_reason="Sensor reading outside expected range",
        status="open",
        real_value=1,
    )


def make_row_alert(
    alert_id: int,
    step: int,
    run_id: int = 42,
    machine_id: int = 3,
    sensor: str = "temperature",
) -> RowAlert:
    return RowAlert(
        run_id=run_id,
        alert_id=alert_id,
        step=step,
        machine_id=machine_id,
        sensor=sensor,
        sensor_value=103.5,
        prediction=1,
        anomaly_score=0.96,
        severity="critical",
        alert_type="anomaly",
        reason="Sensor reading outside expected range",
        status="open",
        anomaly_type="spike",
        real_value=1,
    )


def configure_summary(
    tools: Mock,
    *,
    run_id: int = 42,
    total_alert_events: int = 3,
) -> None:
    tools.get_run_summary.return_value = GetRunSummaryOutput(
        summary=make_run_summary(
            run_id=run_id,
            total_alert_events=total_alert_events,
        )
    )


def configure_empty_event_results(tools: Mock) -> None:
    tools.list_alert_events.return_value = ListAlertEventsOutput(events=[])
    tools.get_event_alerts.return_value = GetEventAlertsOutput(alerts=[])


def test_explicit_run_id_skips_latest_run_lookup():
    tools = Mock(spec=AnomalyOperationalTools)
    configure_summary(tools, total_alert_events=0)

    service = TriageService(tools)

    result = service.triage(TriageRequest(run_id=42))

    assert result.run_id == 42
    tools.get_latest_run.assert_not_called()
    tools.get_run_summary.assert_called_once()

    summary_input = tools.get_run_summary.call_args.args[0]
    assert summary_input.run_id == 42


def test_missing_run_id_resolves_latest_run():
    tools = Mock(spec=AnomalyOperationalTools)

    tools.get_latest_run.return_value = GetLatestRunOutput(run=LatestRun(run_id=42))
    configure_summary(tools, total_alert_events=0)

    service = TriageService(tools)

    result = service.triage(TriageRequest(run_id=None))

    assert result.run_id == 42
    tools.get_latest_run.assert_called_once()

    latest_input = tools.get_latest_run.call_args.args[0]
    assert isinstance(latest_input, GetLatestRunInput)
    tools.get_run_summary.assert_called_once()

    summary_input = tools.get_run_summary.call_args.args[0]
    assert summary_input.run_id == 42


def test_empty_run_returns_no_alerts():
    tools = Mock(spec=AnomalyOperationalTools)
    configure_summary(tools, total_alert_events=0)

    service = TriageService(tools)

    result = service.triage(TriageRequest(run_id=42))

    assert result.run_id == 42
    assert result.status == "no_alerts"
    assert result.findings == []
    assert result.evidence == []

    tools.get_latest_run.assert_not_called()
    tools.get_run_summary.assert_called_once()
    tools.list_alert_events.assert_not_called()
    tools.get_event_alerts.assert_not_called()


def test_fetches_critical_and_warning_events_with_correct_arguments():
    tools = Mock(spec=AnomalyOperationalTools)
    configure_summary(tools)
    configure_empty_event_results(tools)

    service = TriageService(tools)

    service.triage(
        TriageRequest(
            run_id=42,
            max_events=5,
        )
    )

    assert tools.list_alert_events.call_count == 2

    critical_input = tools.list_alert_events.call_args_list[0].args[0]
    warning_input = tools.list_alert_events.call_args_list[1].args[0]

    assert critical_input.run_id == 42
    assert critical_input.severity == "critical"
    assert critical_input.limit == 5
    assert critical_input.offset == 0

    assert warning_input.run_id == 42
    assert warning_input.severity == "warning"
    assert warning_input.limit == 5
    assert warning_input.offset == 0


def test_events_are_sorted_by_priority():
    tools = Mock(spec=AnomalyOperationalTools)
    configure_summary(tools, total_alert_events=5)

    critical_events = [
        make_alert_event(4, "critical", 0.80),
        make_alert_event(1, "critical", None),
        make_alert_event(3, "critical", 0.90),
        make_alert_event(2, "critical", 0.90),
    ]
    warning_events = [
        make_alert_event(5, "warning", 0.99),
    ]

    tools.list_alert_events.side_effect = [
        ListAlertEventsOutput(events=critical_events),
        ListAlertEventsOutput(events=warning_events),
    ]
    tools.get_event_alerts.return_value = GetEventAlertsOutput(alerts=[])

    service = TriageService(tools)

    result = service.triage(
        TriageRequest(
            run_id=42,
            max_events=5,
        )
    )

    ordered_event_ids = [item.event.event_id for item in result.evidence]

    assert ordered_event_ids == [2, 3, 4, 1, 5]


def test_selected_events_are_limited_to_max_events():
    tools = Mock(spec=AnomalyOperationalTools)
    configure_summary(tools, total_alert_events=4)

    tools.list_alert_events.side_effect = [
        ListAlertEventsOutput(
            events=[
                make_alert_event(1, "critical", 0.99),
                make_alert_event(2, "critical", 0.90),
                make_alert_event(3, "critical", 0.80),
            ]
        ),
        ListAlertEventsOutput(
            events=[
                make_alert_event(4, "warning", 0.95),
            ]
        ),
    ]
    tools.get_event_alerts.return_value = GetEventAlertsOutput(alerts=[])

    service = TriageService(tools)

    result = service.triage(
        TriageRequest(
            run_id=42,
            max_events=2,
        )
    )

    assert len(result.findings) == 2
    assert len(result.evidence) == 2
    assert tools.get_event_alerts.call_count == 2


def test_fetches_alerts_for_every_selected_event():
    tools = Mock(spec=AnomalyOperationalTools)
    configure_summary(tools, total_alert_events=3)

    tools.list_alert_events.side_effect = [
        ListAlertEventsOutput(
            events=[
                make_alert_event(10, "critical", 0.95),
                make_alert_event(11, "critical", 0.90),
            ]
        ),
        ListAlertEventsOutput(
            events=[
                make_alert_event(12, "warning", 0.85),
            ]
        ),
    ]
    tools.get_event_alerts.return_value = GetEventAlertsOutput(alerts=[])

    service = TriageService(tools)

    service.triage(
        TriageRequest(
            run_id=42,
            max_events=3,
        )
    )

    assert tools.get_event_alerts.call_count == 3

    alert_inputs = [call.args[0] for call in tools.get_event_alerts.call_args_list]

    assert [tool_input.event_id for tool_input in alert_inputs] == [10, 11, 12]

    assert all(tool_input.run_id == 42 for tool_input in alert_inputs)


def test_row_alerts_are_sorted_by_step_then_alert_id():
    tools = Mock(spec=AnomalyOperationalTools)
    configure_summary(tools, total_alert_events=1)

    tools.list_alert_events.side_effect = [
        ListAlertEventsOutput(
            events=[
                make_alert_event(7, "critical", 0.95),
            ]
        ),
        ListAlertEventsOutput(events=[]),
    ]
    tools.get_event_alerts.return_value = GetEventAlertsOutput(
        alerts=[
            make_row_alert(alert_id=9, step=105),
            make_row_alert(alert_id=4, step=101),
            make_row_alert(alert_id=7, step=105),
        ]
    )

    service = TriageService(tools)

    result = service.triage(TriageRequest(run_id=42))

    ordered_alerts = [(alert.step, alert.alert_id) for alert in result.evidence[0].alerts]

    assert ordered_alerts == [
        (101, 4),
        (105, 7),
        (105, 9),
    ]


def test_findings_reference_existing_evidence():
    tools = Mock(spec=AnomalyOperationalTools)
    configure_summary(tools, total_alert_events=2)

    tools.list_alert_events.side_effect = [
        ListAlertEventsOutput(
            events=[
                make_alert_event(7, "critical", 0.95),
            ]
        ),
        ListAlertEventsOutput(
            events=[
                make_alert_event(8, "warning", 0.80),
            ]
        ),
    ]
    tools.get_event_alerts.return_value = GetEventAlertsOutput(alerts=[])

    service = TriageService(tools)

    result = service.triage(TriageRequest(run_id=42))

    evidence_ids = {item.evidence_id for item in result.evidence}

    assert evidence_ids == {
        "event-7",
        "event-8",
    }

    for finding in result.findings:
        assert finding.evidence_ids
        assert set(finding.evidence_ids).issubset(evidence_ids)

    assert [finding.finding_id for finding in result.findings] == [
        "finding-7",
        "finding-8",
    ]


def test_report_contains_deterministic_finding_summary():
    tools = Mock(spec=AnomalyOperationalTools)
    configure_summary(tools, total_alert_events=1)

    event = make_alert_event(
        event_id=7,
        severity="critical",
        score=0.95,
        machine_id=3,
        sensor="temperature",
        anomaly_type="spike",
    )

    tools.list_alert_events.side_effect = [
        ListAlertEventsOutput(events=[event]),
        ListAlertEventsOutput(events=[]),
    ]
    tools.get_event_alerts.return_value = GetEventAlertsOutput(alerts=[])

    service = TriageService(tools)

    result = service.triage(TriageRequest(run_id=42))

    finding = result.findings[0]

    assert finding.finding_id == "finding-7"
    assert finding.severity == "critical"
    assert finding.machine_id == 3
    assert finding.sensor == "temperature"
    assert finding.anomaly_type == "spike"
    assert finding.summary == ("Critical spike event on machine 3 temperature sensor.")
    assert finding.evidence_ids == ["event-7"]


def test_missing_run_maps_to_triage_run_not_found_error():
    tools = Mock(spec=AnomalyOperationalTools)
    original_error = OperationalResourceNotFoundError("Run was not found.")
    tools.get_run_summary.side_effect = original_error

    service = TriageService(tools)

    with pytest.raises(TriageRunNotFoundError) as exc_info:
        service.triage(TriageRequest(run_id=42))

    assert exc_info.value.__cause__ is original_error


def test_operational_failure_maps_to_triage_service_error():
    tools = Mock(spec=AnomalyOperationalTools)
    original_error = OperationalToolError("Operational API failed.")
    tools.get_run_summary.side_effect = original_error

    service = TriageService(tools)

    with pytest.raises(TriageServiceError) as exc_info:
        service.triage(TriageRequest(run_id=42))

    assert exc_info.value.__cause__ is original_error


@pytest.mark.parametrize("run_id", [0, -1])
def test_triage_request_rejects_invalid_run_id(
    run_id: int,
):
    with pytest.raises(ValidationError):
        TriageRequest(run_id=run_id)


@pytest.mark.parametrize("max_events", [0, -1, 21])
def test_triage_request_rejects_invalid_max_events(
    max_events: int,
):
    with pytest.raises(ValidationError):
        TriageRequest(
            run_id=42,
            max_events=max_events,
        )
