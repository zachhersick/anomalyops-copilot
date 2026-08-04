import pytest
from pydantic import ValidationError

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


def make_run_summary() -> dict[str, object]:
    return {
        "run_id": 42,
        "total_predictions": 100,
        "total_anomalies_predicted": 10,
        "total_row_alerts": 8,
        "total_alert_events": 2,
        "critical_alert_events": 1,
        "warning_alert_events": 1,
        "info_alert_events": 0,
        "machines_with_alerts": 2,
        "max_anomaly_score": 0.97,
        "mean_anomaly_score": 0.74,
    }


def make_alert_event() -> dict[str, object]:
    return {
        "run_id": 42,
        "event_id": 7,
        "machine_id": 3,
        "sensor": "temperature",
        "anomaly_type": "spike",
        "start_step": 100,
        "end_step": 110,
        "duration": 11,
        "alert_count": 4,
        "max_severity": "critical",
        "max_severity_reason": "High anomaly score",
        "max_anomaly_score": 0.97,
        "mean_anomaly_score": 0.90,
        "min_sensor_value": 70.0,
        "max_sensor_value": 105.0,
        "first_reason": "Sensor reading outside expected range",
        "status": "open",
        "real_value": 1,
    }


def make_row_alert() -> dict[str, object]:
    return {
        "run_id": 42,
        "alert_id": 15,
        "step": 105,
        "machine_id": 3,
        "sensor": "temperature",
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


def test_get_latest_run_input_accepts_no_arguments():
    result = GetLatestRunInput()

    assert result.model_dump() == {}


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (
            GetLatestRunInput,
            {
                "unexpected": "value",
            },
        ),
        (
            GetRunSummaryInput,
            {
                "run_id": 42,
                "unexpected": "value",
            },
        ),
        (
            ListAlertEventsInput,
            {
                "run_id": 42,
                "unexpected": "value",
            },
        ),
        (
            GetEventAlertsInput,
            {
                "run_id": 42,
                "event_id": 7,
                "unexpected": "value",
            },
        ),
    ],
)
def test_tool_input_models_reject_extra_fields(
    model,
    kwargs: dict[str, object],
):
    with pytest.raises(ValidationError):
        model(**kwargs)


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (
            GetLatestRunOutput,
            {
                "run": {
                    "run_id": 42,
                },
                "unexpected": "value",
            },
        ),
        (
            GetRunSummaryOutput,
            {
                "summary": make_run_summary(),
                "unexpected": "value",
            },
        ),
        (
            ListAlertEventsOutput,
            {
                "events": [make_alert_event()],
                "unexpected": "value",
            },
        ),
        (
            GetEventAlertsOutput,
            {
                "alerts": [make_row_alert()],
                "unexpected": "value",
            },
        ),
    ],
)
def test_tool_output_models_reject_extra_fields(
    model,
    kwargs: dict[str, object],
):
    with pytest.raises(ValidationError):
        model(**kwargs)


def test_get_run_summary_input_accepts_positive_run_id():
    result = GetRunSummaryInput(run_id=42)

    assert result.run_id == 42


@pytest.mark.parametrize(
    "run_id",
    [0, -1],
)
def test_get_run_summary_input_rejects_nonpositive_run_id(
    run_id: int,
):
    with pytest.raises(ValidationError):
        GetRunSummaryInput(run_id=run_id)


def test_list_alert_events_input_preserves_defaults():
    result = ListAlertEventsInput(run_id=42)

    assert result.run_id == 42
    assert result.severity is None
    assert result.sensor is None
    assert result.anomaly_type is None
    assert result.limit == 100
    assert result.offset == 0


def test_list_alert_events_input_accepts_filters():
    result = ListAlertEventsInput(
        run_id=42,
        severity="critical",
        sensor="temperature",
        anomaly_type="spike",
        limit=5,
        offset=10,
    )

    assert result.run_id == 42
    assert result.severity == "critical"
    assert result.sensor == "temperature"
    assert result.anomaly_type == "spike"
    assert result.limit == 5
    assert result.offset == 10


@pytest.mark.parametrize(
    "run_id",
    [0, -1],
)
def test_list_alert_events_input_rejects_nonpositive_run_id(
    run_id: int,
):
    with pytest.raises(ValidationError):
        ListAlertEventsInput(run_id=run_id)


@pytest.mark.parametrize(
    "limit",
    [0, -1, 501],
)
def test_list_alert_events_input_rejects_invalid_limit(
    limit: int,
):
    with pytest.raises(ValidationError):
        ListAlertEventsInput(
            run_id=42,
            limit=limit,
        )


@pytest.mark.parametrize(
    "limit",
    [1, 500],
)
def test_list_alert_events_input_accepts_limit_boundaries(
    limit: int,
):
    result = ListAlertEventsInput(
        run_id=42,
        limit=limit,
    )

    assert result.limit == limit


def test_list_alert_events_input_rejects_negative_offset():
    with pytest.raises(ValidationError):
        ListAlertEventsInput(
            run_id=42,
            offset=-1,
        )


def test_list_alert_events_input_accepts_zero_offset():
    result = ListAlertEventsInput(
        run_id=42,
        offset=0,
    )

    assert result.offset == 0


def test_get_event_alerts_input_accepts_positive_ids():
    result = GetEventAlertsInput(
        run_id=42,
        event_id=7,
    )

    assert result.run_id == 42
    assert result.event_id == 7


@pytest.mark.parametrize(
    ("run_id", "event_id"),
    [
        (0, 7),
        (-1, 7),
        (42, 0),
        (42, -1),
    ],
)
def test_get_event_alerts_input_rejects_nonpositive_ids(
    run_id: int,
    event_id: int,
):
    with pytest.raises(ValidationError):
        GetEventAlertsInput(
            run_id=run_id,
            event_id=event_id,
        )


def test_get_latest_run_output_validates():
    result = GetLatestRunOutput(
        run={
            "run_id": 42,
        }
    )

    assert result.run.run_id == 42


def test_get_run_summary_output_validates():
    result = GetRunSummaryOutput(
        summary=make_run_summary()
    )

    assert result.summary.run_id == 42
    assert result.summary.total_alert_events == 2


def test_list_alert_events_output_validates():
    result = ListAlertEventsOutput(
        events=[make_alert_event()]
    )

    assert len(result.events) == 1
    assert result.events[0].event_id == 7
    assert result.events[0].machine_id == 3


def test_get_event_alerts_output_validates():
    result = GetEventAlertsOutput(
        alerts=[make_row_alert()]
    )

    assert len(result.alerts) == 1
    assert result.alerts[0].alert_id == 15
    assert result.alerts[0].step == 105


def test_list_alert_events_output_accepts_empty_list():
    result = ListAlertEventsOutput(events=[])

    assert result.events == []


def test_get_event_alerts_output_accepts_empty_list():
    result = GetEventAlertsOutput(alerts=[])

    assert result.alerts == []