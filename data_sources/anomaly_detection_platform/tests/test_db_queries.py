import sqlite3

import pandas as pd
import pytest

from db import create_tables
from load_to_db import insert_pipeline_run, load_dataframe_to_table
from db_queries import (
    get_latest_run_id,
    get_pipeline_runs,
    get_alert_events_for_run,
    get_critical_alert_events,
    get_row_alerts_for_event,
    get_sensor_readings_for_machine,
    get_predictions_for_run,
    run_exists,
    event_exists,
)


@pytest.fixture
def temp_connection(tmp_path):
    """
    Create a temporary SQLite database for each test.
    """
    db_path = tmp_path / "test_anomaly_detection.db"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    create_tables(conn)

    yield conn

    conn.close()


def test_get_latest_run_id_returns_none_when_no_runs_exist(temp_connection):
    latest_run_id = get_latest_run_id(temp_connection)

    assert latest_run_id is None


def test_get_latest_run_id_returns_highest_run_id(temp_connection):
    run_id_1 = insert_pipeline_run(temp_connection, notes="First test run")
    run_id_2 = insert_pipeline_run(temp_connection, notes="Second test run")

    latest_run_id = get_latest_run_id(temp_connection)

    assert latest_run_id == run_id_2
    assert latest_run_id > run_id_1


def test_get_pipeline_runs_returns_newest_first(temp_connection):
    run_id_1 = insert_pipeline_run(temp_connection, notes="First test run")
    run_id_2 = insert_pipeline_run(temp_connection, notes="Second test run")

    rows = get_pipeline_runs(temp_connection)

    run_ids = [row["run_id"] for row in rows]

    assert run_ids == [run_id_2, run_id_1]


def test_get_alert_events_for_run_filters_by_run_and_orders_by_start_step(temp_connection):
    run_id_1 = insert_pipeline_run(temp_connection, notes="Run with two events")
    run_id_2 = insert_pipeline_run(temp_connection, notes="Run with one event")

    run_1_events = pd.DataFrame(
        [
            {
                "event_id": 1,
                "machine_id": 1,
                "sensor": "temperature",
                "anomaly_type": "spike",
                "start_step": 20,
                "end_step": 25,
                "duration": 6,
                "alert_count": 3,
                "max_severity": "WARNING",
                "status": "open",
            },
            {
                "event_id": 2,
                "machine_id": 1,
                "sensor": "pressure",
                "anomaly_type": "drop",
                "start_step": 10,
                "end_step": 12,
                "duration": 3,
                "alert_count": 2,
                "max_severity": "CRITICAL",
                "status": "open",
            },
        ]
    )

    run_2_events = pd.DataFrame(
        [
            {
                "event_id": 1,
                "machine_id": 2,
                "sensor": "voltage",
                "anomaly_type": "drift",
                "start_step": 5,
                "end_step": 15,
                "duration": 11,
                "alert_count": 4,
                "max_severity": "WARNING",
                "status": "open",
            }
        ]
    )

    load_dataframe_to_table(temp_connection, run_1_events, "alert_events", run_id_1)
    load_dataframe_to_table(temp_connection, run_2_events, "alert_events", run_id_2)

    rows = get_alert_events_for_run(temp_connection, run_id_1)

    event_ids = [row["event_id"] for row in rows]

    assert len(rows) == 2
    assert event_ids == [2, 1]


def test_get_critical_alert_events_returns_only_critical_events(temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    events = pd.DataFrame(
        [
            {
                "event_id": 1,
                "machine_id": 1,
                "sensor": "temperature",
                "anomaly_type": "spike",
                "start_step": 10,
                "end_step": 12,
                "duration": 3,
                "alert_count": 2,
                "max_severity": "WARNING",
                "status": "open",
            },
            {
                "event_id": 2,
                "machine_id": 1,
                "sensor": "pressure",
                "anomaly_type": "drop",
                "start_step": 20,
                "end_step": 22,
                "duration": 3,
                "alert_count": 2,
                "max_severity": "CRITICAL",
                "status": "open",
            },
        ]
    )

    load_dataframe_to_table(temp_connection, events, "alert_events", run_id)

    rows = get_critical_alert_events(temp_connection, run_id)

    assert len(rows) == 1
    assert rows[0]["event_id"] == 2
    assert rows[0]["max_severity"] == "CRITICAL"


def test_get_row_alerts_for_event_returns_alerts_inside_event_window(temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    events = pd.DataFrame(
        [
            {
                "event_id": 1,
                "machine_id": 1,
                "sensor": "temperature",
                "anomaly_type": "spike",
                "start_step": 10,
                "end_step": 12,
                "duration": 3,
                "alert_count": 3,
                "max_severity": "CRITICAL",
                "status": "open",
            }
        ]
    )

    alerts = pd.DataFrame(
        [
            {
                "alert_id": 1,
                "step": 9,
                "machine_id": 1,
                "sensor": "temperature",
                "sensor_value": 99.0,
                "prediction": 1,
                "anomaly_score": 0.70,
                "severity": "WARNING",
                "alert_type": "model_anomaly",
                "reason": "Before event window",
                "status": "open",
                "anomaly_type": "spike",
                "real_value": 1,
            },
            {
                "alert_id": 2,
                "step": 10,
                "machine_id": 1,
                "sensor": "temperature",
                "sensor_value": 105.0,
                "prediction": 1,
                "anomaly_score": 0.90,
                "severity": "CRITICAL",
                "alert_type": "model_and_threshold",
                "reason": "Inside event window",
                "status": "open",
                "anomaly_type": "spike",
                "real_value": 1,
            },
            {
                "alert_id": 3,
                "step": 11,
                "machine_id": 1,
                "sensor": "temperature",
                "sensor_value": 106.0,
                "prediction": 1,
                "anomaly_score": 0.91,
                "severity": "CRITICAL",
                "alert_type": "model_and_threshold",
                "reason": "Inside event window",
                "status": "open",
                "anomaly_type": "spike",
                "real_value": 1,
            },
            {
                "alert_id": 4,
                "step": 12,
                "machine_id": 1,
                "sensor": "temperature",
                "sensor_value": 107.0,
                "prediction": 1,
                "anomaly_score": 0.92,
                "severity": "CRITICAL",
                "alert_type": "model_and_threshold",
                "reason": "Inside event window",
                "status": "open",
                "anomaly_type": "spike",
                "real_value": 1,
            },
            {
                "alert_id": 5,
                "step": 13,
                "machine_id": 1,
                "sensor": "temperature",
                "sensor_value": 101.0,
                "prediction": 1,
                "anomaly_score": 0.72,
                "severity": "WARNING",
                "alert_type": "model_anomaly",
                "reason": "After event window",
                "status": "open",
                "anomaly_type": "spike",
                "real_value": 1,
            },
        ]
    )

    load_dataframe_to_table(temp_connection, events, "alert_events", run_id)
    load_dataframe_to_table(temp_connection, alerts, "row_alerts", run_id)

    rows = get_row_alerts_for_event(temp_connection, run_id, event_id=1)

    steps = [row["step"] for row in rows]

    assert steps == [10, 11, 12]


def test_get_row_alerts_for_event_returns_empty_list_when_event_does_not_exist(temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    rows = get_row_alerts_for_event(temp_connection, run_id, event_id=999)

    assert rows == []


def test_get_sensor_readings_for_machine_filters_by_run_machine_and_limit(temp_connection):
    run_id_1 = insert_pipeline_run(temp_connection)
    run_id_2 = insert_pipeline_run(temp_connection)

    run_1_readings = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01 00:00:01",
                "step": 1,
                "machine_id": 1,
                "temperature": 70.0,
                "pressure": 50.0,
                "vibration": 2.0,
                "flow_rate": 1.0,
                "voltage": 120.0,
                "current": 10.0,
                "is_anomaly": 0,
                "anomaly_type": "normal",
                "target_sensor": "temperature",
            },
            {
                "timestamp": "2026-01-01 00:00:02",
                "step": 2,
                "machine_id": 1,
                "temperature": 71.0,
                "pressure": 51.0,
                "vibration": 2.1,
                "flow_rate": 1.1,
                "voltage": 121.0,
                "current": 11.0,
                "is_anomaly": 0,
                "anomaly_type": "normal",
                "target_sensor": "temperature",
            },
            {
                "timestamp": "2026-01-01 00:00:03",
                "step": 3,
                "machine_id": 1,
                "temperature": 72.0,
                "pressure": 52.0,
                "vibration": 2.2,
                "flow_rate": 1.2,
                "voltage": 122.0,
                "current": 12.0,
                "is_anomaly": 0,
                "anomaly_type": "normal",
                "target_sensor": "temperature",
            },
            {
                "timestamp": "2026-01-01 00:00:01",
                "step": 1,
                "machine_id": 2,
                "temperature": 80.0,
                "pressure": 60.0,
                "vibration": 3.0,
                "flow_rate": 1.5,
                "voltage": 125.0,
                "current": 15.0,
                "is_anomaly": 0,
                "anomaly_type": "normal",
                "target_sensor": "temperature",
            },
        ]
    )

    run_2_readings = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01 00:00:01",
                "step": 1,
                "machine_id": 1,
                "temperature": 90.0,
                "pressure": 70.0,
                "vibration": 4.0,
                "flow_rate": 1.8,
                "voltage": 127.0,
                "current": 17.0,
                "is_anomaly": 1,
                "anomaly_type": "spike",
                "target_sensor": "temperature",
            }
        ]
    )

    load_dataframe_to_table(temp_connection, run_1_readings, "sensor_readings", run_id_1)
    load_dataframe_to_table(temp_connection, run_2_readings, "sensor_readings", run_id_2)

    rows = get_sensor_readings_for_machine(
        temp_connection,
        run_id=run_id_1,
        machine_id=1,
        limit=2,
    )

    steps = [row["step"] for row in rows]

    assert len(rows) == 2
    assert steps == [1, 2]
    assert all(row["run_id"] == run_id_1 for row in rows)
    assert all(row["machine_id"] == 1 for row in rows)


def test_get_predictions_for_run_filters_by_run_and_limit(temp_connection):
    run_id_1 = insert_pipeline_run(temp_connection)
    run_id_2 = insert_pipeline_run(temp_connection)

    run_1_predictions = pd.DataFrame(
        [
            {
                "step": 1,
                "machine_id": 1,
                "real_value": 0,
                "prediction": 0,
                "anomaly_score": 0.10,
                "threshold": 0.35,
                "anomaly_type": "normal",
                "target_sensor": "temperature",
            },
            {
                "step": 2,
                "machine_id": 1,
                "real_value": 1,
                "prediction": 1,
                "anomaly_score": 0.90,
                "threshold": 0.35,
                "anomaly_type": "spike",
                "target_sensor": "temperature",
            },
            {
                "step": 3,
                "machine_id": 2,
                "real_value": 0,
                "prediction": 0,
                "anomaly_score": 0.20,
                "threshold": 0.35,
                "anomaly_type": "normal",
                "target_sensor": "pressure",
            },
        ]
    )

    run_2_predictions = pd.DataFrame(
        [
            {
                "step": 1,
                "machine_id": 1,
                "real_value": 1,
                "prediction": 1,
                "anomaly_score": 0.95,
                "threshold": 0.35,
                "anomaly_type": "drop",
                "target_sensor": "current",
            }
        ]
    )

    load_dataframe_to_table(temp_connection, run_1_predictions, "model_predictions", run_id_1)
    load_dataframe_to_table(temp_connection, run_2_predictions, "model_predictions", run_id_2)

    rows = get_predictions_for_run(temp_connection, run_id=run_id_1, limit=2)

    assert len(rows) == 2
    assert [row["step"] for row in rows] == [1, 2]
    assert all(row["run_id"] == run_id_1 for row in rows)
    
def test_run_exists_returns_false_when_run_missing(temp_connection):
    result = run_exists(temp_connection, run_id=999)

    assert result is False


def test_run_exists_returns_true_when_run_exists(temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    result = run_exists(temp_connection, run_id)

    assert result is True
    
def test_event_exists_returns_false_when_event_missing(temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    result = event_exists(
        conn=temp_connection,
        run_id=run_id,
        event_id=999,
    )

    assert result is False


def test_event_exists_returns_true_when_event_exists(temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    events = pd.DataFrame(
        [
            {
                "event_id": 1,
                "machine_id": 1,
                "sensor": "temperature",
                "anomaly_type": "spike",
                "start_step": 10,
                "end_step": 12,
                "duration": 3,
                "alert_count": 2,
                "max_severity": "CRITICAL",
                "status": "open",
            }
        ]
    )

    load_dataframe_to_table(
        conn=temp_connection,
        df=events,
        table_name="alert_events",
        run_id=run_id,
    )

    result = event_exists(
        conn=temp_connection,
        run_id=run_id,
        event_id=1,
    )

    assert result is True