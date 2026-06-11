import pandas as pd
import pytest

from alert_events import (
    MAX_STEP_GAP,
    finalize_event,
    get_severity_priority,
    group_alert_events,
    load_alerts,
    safe_max,
    safe_min,
    same_event,
    save_alert_events,
    run_alert_event_pipeline,
    start_event,
    update_event,
)


def make_alerts_df():
    return pd.DataFrame(
        [
            {
                "alert_id": 1,
                "step": 10,
                "machine_id": 1,
                "sensor": "temperature",
                "sensor_value": 105.0,
                "prediction": 1,
                "anomaly_score": 0.80,
                "severity": "WARNING",
                "alert_type": "model_anomaly",
                "reason": "First warning alert",
                "status": "OPEN",
                "anomaly_type": "spike",
                "real_value": 1,
            },
            {
                "alert_id": 2,
                "step": 11,
                "machine_id": 1,
                "sensor": "temperature",
                "sensor_value": 106.0,
                "prediction": 1,
                "anomaly_score": 0.90,
                "severity": "CRITICAL",
                "alert_type": "model_and_threshold",
                "reason": "Critical alert",
                "status": "OPEN",
                "anomaly_type": "spike",
                "real_value": 1,
            },
            {
                "alert_id": 3,
                "step": 20,
                "machine_id": 1,
                "sensor": "temperature",
                "sensor_value": 107.0,
                "prediction": 1,
                "anomaly_score": 0.70,
                "severity": "WARNING",
                "alert_type": "model_anomaly",
                "reason": "Separate event because of large step gap",
                "status": "OPEN",
                "anomaly_type": "spike",
                "real_value": 1,
            },
            {
                "alert_id": 4,
                "step": 21,
                "machine_id": 2,
                "sensor": "pressure",
                "sensor_value": 120.0,
                "prediction": 1,
                "anomaly_score": 0.95,
                "severity": "CRITICAL",
                "alert_type": "model_and_threshold",
                "reason": "Different machine pressure alert",
                "status": "OPEN",
                "anomaly_type": "drop",
                "real_value": 1,
            },
        ]
    )


def test_load_alerts_loads_valid_file(tmp_path):
    alerts_df = make_alerts_df()
    input_path = tmp_path / "alerts.csv"

    alerts_df.to_csv(input_path, index=False)

    loaded_df = load_alerts(input_path)

    assert len(loaded_df) == len(alerts_df)
    assert "alert_id" in loaded_df.columns
    assert "severity" in loaded_df.columns
    assert "anomaly_score" in loaded_df.columns


def test_load_alerts_raises_when_required_column_missing(tmp_path):
    alerts_df = make_alerts_df().drop(columns=["reason"])
    input_path = tmp_path / "bad_alerts.csv"

    alerts_df.to_csv(input_path, index=False)

    with pytest.raises(KeyError, match="Column not found"):
        load_alerts(input_path)


def test_get_severity_priority_returns_expected_values():
    assert get_severity_priority("INFO") == 1
    assert get_severity_priority("WARNING") == 2
    assert get_severity_priority("CRITICAL") == 3
    assert get_severity_priority("UNKNOWN") == 0


def test_same_event_returns_true_for_same_machine_sensor_type_within_gap():
    alerts_df = make_alerts_df()

    first_row = alerts_df.iloc[0]
    second_row = alerts_df.iloc[1]

    current_event = start_event(first_row, event_id=1)

    assert same_event(current_event, second_row) is True


def test_same_event_returns_false_when_step_gap_is_too_large():
    alerts_df = make_alerts_df()

    first_row = alerts_df.iloc[0]
    later_row = alerts_df.iloc[2]

    current_event = start_event(first_row, event_id=1)

    assert later_row["step"] - current_event["end_step"] > MAX_STEP_GAP
    assert same_event(current_event, later_row) is False


def test_same_event_returns_false_for_different_machine_sensor_or_type():
    alerts_df = make_alerts_df()

    first_row = alerts_df.iloc[0]
    different_machine_row = alerts_df.iloc[3]

    current_event = start_event(first_row, event_id=1)

    assert same_event(current_event, different_machine_row) is False


def test_safe_min_handles_nan_values():
    assert safe_min(pd.NA, 5.0) == 5.0
    assert safe_min(5.0, pd.NA) == 5.0
    assert safe_min(5.0, 3.0) == 3.0


def test_safe_max_handles_nan_values():
    assert safe_max(pd.NA, 5.0) == 5.0
    assert safe_max(5.0, pd.NA) == 5.0
    assert safe_max(5.0, 8.0) == 8.0


def test_start_event_creates_expected_event_fields():
    alerts_df = make_alerts_df()
    row = alerts_df.iloc[0]

    event = start_event(row, event_id=1)

    assert event["event_id"] == 1
    assert event["machine_id"] == 1
    assert event["sensor"] == "temperature"
    assert event["anomaly_type"] == "spike"
    assert event["start_step"] == 10
    assert event["end_step"] == 10
    assert event["duration"] == 1
    assert event["alert_count"] == 1
    assert event["max_severity"] == "WARNING"
    assert event["max_anomaly_score"] == 0.80
    assert event["mean_anomaly_score"] == 0.80
    assert event["score_sum"] == 0.80


def test_update_event_extends_event_and_promotes_severity():
    alerts_df = make_alerts_df()

    first_row = alerts_df.iloc[0]
    second_row = alerts_df.iloc[1]

    event = start_event(first_row, event_id=1)
    updated_event = update_event(event, second_row)

    assert updated_event["start_step"] == 10
    assert updated_event["end_step"] == 11
    assert updated_event["duration"] == 2
    assert updated_event["alert_count"] == 2
    assert updated_event["max_severity"] == "CRITICAL"
    assert updated_event["max_severity_reason"] == "Critical alert"
    assert updated_event["max_anomaly_score"] == 0.90
    assert updated_event["mean_anomaly_score"] == pytest.approx(0.85)
    assert updated_event["min_sensor_value"] == 105.0
    assert updated_event["max_sensor_value"] == 106.0


def test_finalize_event_removes_score_sum():
    alerts_df = make_alerts_df()
    row = alerts_df.iloc[0]

    event = start_event(row, event_id=1)
    finalized_event = finalize_event(event)

    assert "score_sum" not in finalized_event


def test_group_alert_events_groups_consecutive_alerts():
    alerts_df = make_alerts_df()

    events_df = group_alert_events(alerts_df)

    first_event = events_df[events_df["event_id"] == 1].iloc[0]

    assert first_event["machine_id"] == 1
    assert first_event["sensor"] == "temperature"
    assert first_event["anomaly_type"] == "spike"
    assert first_event["start_step"] == 10
    assert first_event["end_step"] == 11
    assert first_event["duration"] == 2
    assert first_event["alert_count"] == 2
    assert first_event["max_severity"] == "CRITICAL"
    assert first_event["max_anomaly_score"] == 0.90
    assert first_event["mean_anomaly_score"] == pytest.approx(0.85)


def test_group_alert_events_separates_large_step_gap():
    alerts_df = make_alerts_df()

    events_df = group_alert_events(alerts_df)

    temperature_events = events_df[
        (events_df["machine_id"] == 1)
        & (events_df["sensor"] == "temperature")
    ]

    assert len(temperature_events) == 2


def test_group_alert_events_separates_different_machine_sensor_and_type():
    alerts_df = make_alerts_df()

    events_df = group_alert_events(alerts_df)

    assert len(events_df) == 3

    pressure_event = events_df[
        (events_df["machine_id"] == 2)
        & (events_df["sensor"] == "pressure")
    ].iloc[0]

    assert pressure_event["anomaly_type"] == "drop"
    assert pressure_event["start_step"] == 21
    assert pressure_event["end_step"] == 21
    assert pressure_event["alert_count"] == 1


def test_group_alert_events_returns_empty_dataframe_for_empty_input():
    empty_alerts = make_alerts_df().iloc[0:0]

    events_df = group_alert_events(empty_alerts)

    assert events_df.empty
    assert "event_id" in events_df.columns
    assert "max_severity" in events_df.columns
    assert "max_anomaly_score" in events_df.columns


def test_save_alert_events_writes_file(tmp_path):
    alerts_df = make_alerts_df()
    events_df = group_alert_events(alerts_df)

    output_path = tmp_path / "alert_events.csv"

    save_alert_events(events_df, output_path)

    assert output_path.exists()

    saved_df = pd.read_csv(output_path)

    assert len(saved_df) == len(events_df)
    assert "event_id" in saved_df.columns
    assert "alert_count" in saved_df.columns
    assert "max_severity" in saved_df.columns


def test_run_alert_event_pipeline_writes_output_and_returns_dataframe(tmp_path, capsys):
    alerts_df = make_alerts_df()

    input_path = tmp_path / "alerts.csv"
    output_path = tmp_path / "alert_events.csv"

    alerts_df.to_csv(input_path, index=False)

    events_df = run_alert_event_pipeline(
        input_file=input_path,
        output_file=output_path,
    )

    captured = capsys.readouterr()

    assert output_path.exists()
    assert len(events_df) == 3
    assert "EVENT SUMMARY" in captured.out
