import sqlite3

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app, get_db_connection
from db import create_tables
from load_to_db import insert_pipeline_run, load_dataframe_to_table


@pytest.fixture
def temp_connection(tmp_path):
    """
    Create a temporary SQLite database for each test.
    """
    db_path = tmp_path / "test_anomaly_detection.db"

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    create_tables(conn)

    yield conn

    conn.close()


@pytest.fixture
def client(temp_connection):
    """
    Create a FastAPI test client that uses the temporary test database.
    """

    def test_get_db_connection():
        yield temp_connection

    app.dependency_overrides[get_db_connection] = test_get_db_connection

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def seed_event_filter_test_data(temp_connection):
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
            },
            {
                "event_id": 2,
                "machine_id": 1,
                "sensor": "pressure",
                "anomaly_type": "drift",
                "start_step": 20,
                "end_step": 25,
                "duration": 6,
                "alert_count": 3,
                "max_severity": "WARNING",
                "status": "open",
            },
            {
                "event_id": 3,
                "machine_id": 2,
                "sensor": "vibration",
                "anomaly_type": "oscillation",
                "start_step": 30,
                "end_step": 35,
                "duration": 6,
                "alert_count": 4,
                "max_severity": "CRITICAL",
                "status": "open",
            },
        ]
    )

    load_dataframe_to_table(temp_connection, events, "alert_events", run_id)

    return run_id


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_read_pipeline_runs(client, temp_connection):
    older_run_id = insert_pipeline_run(
        temp_connection,
        notes="older test run",
        fixed_seed=42,
        model_threshold=0.80,
        max_step_gap=3,
    )

    newer_run_id = insert_pipeline_run(
        temp_connection,
        notes="newer test run",
        fixed_seed=99,
        model_threshold=0.90,
        max_step_gap=5,
    )

    response = client.get("/runs")

    assert response.status_code == 200

    data = response.json()
    returned_run_ids = [run["run_id"] for run in data]

    assert returned_run_ids == [newer_run_id, older_run_id]


def test_read_latest_run(client, temp_connection):
    older_run_id = insert_pipeline_run(
        temp_connection,
        notes="older test run",
        fixed_seed=42,
        model_threshold=0.80,
        max_step_gap=3,
    )

    newer_run_id = insert_pipeline_run(
        temp_connection,
        notes="newer test run",
        fixed_seed=99,
        model_threshold=0.90,
        max_step_gap=5,
    )

    response = client.get("/runs/latest")

    assert response.status_code == 200
    assert response.json() == {"run_id": newer_run_id}
    assert response.json()["run_id"] > older_run_id


def test_read_latest_run_returns_404_when_no_runs_exist(client):
    response = client.get("/runs/latest")

    assert response.status_code == 404
    assert response.json()["detail"] == "No pipeline runs found."


def test_read_alert_events_for_run(client, temp_connection):
    run_id_1 = insert_pipeline_run(temp_connection, notes="run one")
    run_id_2 = insert_pipeline_run(temp_connection, notes="run two")

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

    response = client.get(f"/runs/{run_id_1}/events")

    assert response.status_code == 200

    data = response.json()
    returned_event_ids = [event["event_id"] for event in data]

    assert len(data) == 2
    assert returned_event_ids == [2, 1]
    assert all(event["run_id"] == run_id_1 for event in data)


def test_read_alert_events_filters_by_severity(client, temp_connection):
    run_id = seed_event_filter_test_data(temp_connection)

    response = client.get(f"/runs/{run_id}/events?severity=CRITICAL")

    assert response.status_code == 200

    data = response.json()
    returned_event_ids = [event["event_id"] for event in data]

    assert returned_event_ids == [1, 3]
    assert all(event["max_severity"] == "CRITICAL" for event in data)


def test_read_alert_events_filters_by_sensor(client, temp_connection):
    run_id = seed_event_filter_test_data(temp_connection)

    response = client.get(f"/runs/{run_id}/events?sensor=temperature")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["event_id"] == 1
    assert data[0]["sensor"] == "temperature"


def test_read_alert_events_filters_by_anomaly_type(client, temp_connection):
    run_id = seed_event_filter_test_data(temp_connection)

    response = client.get(f"/runs/{run_id}/events?anomaly_type=spike")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["event_id"] == 1
    assert data[0]["anomaly_type"] == "spike"


def test_read_alert_events_applies_limit(client, temp_connection):
    run_id = seed_event_filter_test_data(temp_connection)

    response = client.get(f"/runs/{run_id}/events?limit=1")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["event_id"] == 1


def test_read_alert_events_combines_filters(client, temp_connection):
    run_id = seed_event_filter_test_data(temp_connection)

    response = client.get(
        f"/runs/{run_id}/events"
        "?severity=CRITICAL"
        "&sensor=temperature"
        "&anomaly_type=spike"
        "&limit=100"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["event_id"] == 1
    assert data[0]["max_severity"] == "CRITICAL"
    assert data[0]["sensor"] == "temperature"
    assert data[0]["anomaly_type"] == "spike"


def test_read_critical_alert_events_for_run(client, temp_connection):
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

    response = client.get(f"/runs/{run_id}/events/critical")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["event_id"] == 2
    assert data[0]["max_severity"] == "CRITICAL"


def test_read_row_alerts_for_event(client, temp_connection):
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

    response = client.get(f"/runs/{run_id}/events/1/alerts")

    assert response.status_code == 200

    data = response.json()
    returned_steps = [alert["step"] for alert in data]

    assert returned_steps == [10, 11, 12]
    assert all(alert["run_id"] == run_id for alert in data)
    assert all(alert["machine_id"] == 1 for alert in data)
    assert all(alert["sensor"] == "temperature" for alert in data)


def test_read_sensor_readings_for_machine(client, temp_connection):
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

    response = client.get(f"/runs/{run_id_1}/machines/1/readings?limit=2")

    assert response.status_code == 200

    data = response.json()
    returned_steps = [reading["step"] for reading in data]

    assert len(data) == 2
    assert returned_steps == [1, 2]
    assert all(reading["run_id"] == run_id_1 for reading in data)
    assert all(reading["machine_id"] == 1 for reading in data)
    
    
def test_read_sensor_readings_applies_offset(client, temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    readings = pd.DataFrame(
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
                "timestamp": "2026-01-01 00:00:04",
                "step": 4,
                "machine_id": 1,
                "temperature": 73.0,
                "pressure": 53.0,
                "vibration": 2.3,
                "flow_rate": 1.3,
                "voltage": 123.0,
                "current": 13.0,
                "is_anomaly": 0,
                "anomaly_type": "normal",
                "target_sensor": "temperature",
            },
        ]
    )

    load_dataframe_to_table(temp_connection, readings, "sensor_readings", run_id)

    response = client.get(f"/runs/{run_id}/machines/1/readings?limit=2&offset=2")

    assert response.status_code == 200

    data = response.json()
    returned_steps = [reading["step"] for reading in data]

    assert len(data) == 2
    assert returned_steps == [3, 4]


def test_read_sensor_readings_offset_zero_returns_first_page(client, temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    readings = pd.DataFrame(
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
        ]
    )

    load_dataframe_to_table(temp_connection, readings, "sensor_readings", run_id)

    response = client.get(f"/runs/{run_id}/machines/1/readings?limit=2&offset=0")

    assert response.status_code == 200

    data = response.json()
    returned_steps = [reading["step"] for reading in data]

    assert len(data) == 2
    assert returned_steps == [1, 2]


def test_read_predictions_for_run(client, temp_connection):
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

    response = client.get(f"/runs/{run_id_1}/predictions?limit=2")

    assert response.status_code == 200

    data = response.json()
    returned_steps = [prediction["step"] for prediction in data]

    assert len(data) == 2
    assert returned_steps == [1, 2]
    assert all(prediction["run_id"] == run_id_1 for prediction in data)
    

def seed_prediction_filter_test_data(temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    predictions = pd.DataFrame(
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
            {
                "step": 4,
                "machine_id": 2,
                "real_value": 1,
                "prediction": 1,
                "anomaly_score": 0.85,
                "threshold": 0.35,
                "anomaly_type": "drop",
                "target_sensor": "current",
            },
        ]
    )

    load_dataframe_to_table(
        temp_connection,
        predictions,
        "model_predictions",
        run_id,
    )

    return run_id


def test_read_predictions_filters_by_machine_id(client, temp_connection):
    run_id = seed_prediction_filter_test_data(temp_connection)

    response = client.get(f"/runs/{run_id}/predictions?machine_id=1")

    assert response.status_code == 200

    data = response.json()
    returned_steps = [prediction["step"] for prediction in data]

    assert returned_steps == [1, 2]
    assert all(prediction["machine_id"] == 1 for prediction in data)


def test_read_predictions_filters_by_anomaly_type(client, temp_connection):
    run_id = seed_prediction_filter_test_data(temp_connection)

    response = client.get(f"/runs/{run_id}/predictions?anomaly_type=spike")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["step"] == 2
    assert data[0]["anomaly_type"] == "spike"


def test_read_predictions_filters_by_target_sensor(client, temp_connection):
    run_id = seed_prediction_filter_test_data(temp_connection)

    response = client.get(f"/runs/{run_id}/predictions?target_sensor=temperature")

    assert response.status_code == 200

    data = response.json()
    returned_steps = [prediction["step"] for prediction in data]

    assert returned_steps == [1, 2]
    assert all(prediction["target_sensor"] == "temperature" for prediction in data)


def test_read_predictions_combines_filters(client, temp_connection):
    run_id = seed_prediction_filter_test_data(temp_connection)

    response = client.get(
        f"/runs/{run_id}/predictions"
        "?machine_id=1"
        "&anomaly_type=spike"
        "&target_sensor=temperature"
        "&limit=100"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["step"] == 2
    assert data[0]["machine_id"] == 1
    assert data[0]["anomaly_type"] == "spike"
    assert data[0]["target_sensor"] == "temperature"


def test_read_predictions_applies_limit_after_filters(client, temp_connection):
    run_id = seed_prediction_filter_test_data(temp_connection)

    response = client.get(
        f"/runs/{run_id}/predictions"
        "?target_sensor=temperature"
        "&limit=1"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["step"] == 1
    assert data[0]["target_sensor"] == "temperature"
    
    
def test_read_predictions_applies_offset(client, temp_connection):
    run_id = seed_prediction_filter_test_data(temp_connection)

    response = client.get(f"/runs/{run_id}/predictions?limit=2&offset=2")

    assert response.status_code == 200

    data = response.json()
    returned_steps = [prediction["step"] for prediction in data]

    assert len(data) == 2
    assert returned_steps == [3, 4]


def test_read_predictions_offset_zero_returns_first_page(client, temp_connection):
    run_id = seed_prediction_filter_test_data(temp_connection)

    response = client.get(f"/runs/{run_id}/predictions?limit=2&offset=0")

    assert response.status_code == 200

    data = response.json()
    returned_steps = [prediction["step"] for prediction in data]

    assert len(data) == 2
    assert returned_steps == [1, 2]


def test_read_predictions_applies_offset_after_filters(client, temp_connection):
    run_id = seed_prediction_filter_test_data(temp_connection)

    response = client.get(
        f"/runs/{run_id}/predictions"
        "?target_sensor=temperature"
        "&limit=1"
        "&offset=1"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["step"] == 2
    assert data[0]["target_sensor"] == "temperature"


def test_read_alert_events_returns_404_when_run_missing(client):
    response = client.get("/runs/999/events")

    assert response.status_code == 404
    assert response.json()["detail"] == "Pipeline run not found"
    

def test_read_alert_events_applies_offset(client, temp_connection):
    run_id = seed_event_filter_test_data(temp_connection)

    response = client.get(f"/runs/{run_id}/events?limit=2&offset=2")

    assert response.status_code == 200

    data = response.json()
    returned_event_ids = [event["event_id"] for event in data]

    assert len(data) == 1
    assert returned_event_ids == [3]


def test_read_alert_events_offset_zero_returns_first_page(client, temp_connection):
    run_id = seed_event_filter_test_data(temp_connection)

    response = client.get(f"/runs/{run_id}/events?limit=2&offset=0")

    assert response.status_code == 200

    data = response.json()
    returned_event_ids = [event["event_id"] for event in data]

    assert len(data) == 2
    assert returned_event_ids == [1, 2]


def test_read_alert_events_applies_offset_after_filters(client, temp_connection):
    run_id = seed_event_filter_test_data(temp_connection)

    response = client.get(
        f"/runs/{run_id}/events"
        "?severity=CRITICAL"
        "&limit=1"
        "&offset=1"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["event_id"] == 3
    assert data[0]["max_severity"] == "CRITICAL"


def test_read_predictions_returns_404_when_run_missing(client):
    response = client.get("/runs/999/predictions")

    assert response.status_code == 404
    assert response.json()["detail"] == "Pipeline run not found"


def test_read_sensor_readings_returns_404_when_run_missing(client):
    response = client.get("/runs/999/machines/1/readings")

    assert response.status_code == 404
    assert response.json()["detail"] == "Pipeline run not found"


def test_read_row_alerts_returns_404_when_run_missing(client):
    response = client.get("/runs/999/events/1/alerts")

    assert response.status_code == 404
    assert response.json()["detail"] == "Pipeline run not found"


def test_read_row_alerts_returns_404_when_event_missing(client, temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    response = client.get(f"/runs/{run_id}/events/999/alerts")

    assert response.status_code == 404
    assert response.json()["detail"] == "Alert event not found"


def test_read_row_alerts_returns_404_when_run_missing_before_event_check(client):
    response = client.get("/runs/999/events/999/alerts")

    assert response.status_code == 404
    assert response.json()["detail"] == "Pipeline run not found"
    
    
def test_read_alert_events_rejects_limit_zero(client, temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    response = client.get(f"/runs/{run_id}/events?limit=0")

    assert response.status_code == 422


def test_read_alert_events_rejects_limit_above_max(client, temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    response = client.get(f"/runs/{run_id}/events?limit=501")

    assert response.status_code == 422


def test_read_alert_events_rejects_negative_offset(client, temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    response = client.get(f"/runs/{run_id}/events?offset=-1")

    assert response.status_code == 422


def test_read_predictions_rejects_limit_zero(client, temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    response = client.get(f"/runs/{run_id}/predictions?limit=0")

    assert response.status_code == 422


def test_read_predictions_rejects_limit_above_max(client, temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    response = client.get(f"/runs/{run_id}/predictions?limit=501")

    assert response.status_code == 422


def test_read_predictions_rejects_negative_offset(client, temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    response = client.get(f"/runs/{run_id}/predictions?offset=-1")

    assert response.status_code == 422


def test_read_sensor_readings_rejects_limit_zero(client, temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    response = client.get(f"/runs/{run_id}/machines/1/readings?limit=0")

    assert response.status_code == 422


def test_read_sensor_readings_rejects_limit_above_max(client, temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    response = client.get(f"/runs/{run_id}/machines/1/readings?limit=501")

    assert response.status_code == 422


def test_read_sensor_readings_rejects_negative_offset(client, temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    response = client.get(f"/runs/{run_id}/machines/1/readings?offset=-1")

    assert response.status_code == 422
    

def test_read_run_summary(client, temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    predictions = pd.DataFrame(
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
                "real_value": 1,
                "prediction": 1,
                "anomaly_score": 0.80,
                "threshold": 0.35,
                "anomaly_type": "drop",
                "target_sensor": "pressure",
            },
        ]
    )

    row_alerts = pd.DataFrame(
        [
            {
                "alert_id": 1,
                "step": 2,
                "machine_id": 1,
                "sensor": "temperature",
                "sensor_value": 105.0,
                "prediction": 1,
                "anomaly_score": 0.90,
                "severity": "CRITICAL",
                "alert_type": "model_and_threshold",
                "reason": "test alert one",
                "status": "open",
                "anomaly_type": "spike",
                "real_value": 1,
            },
            {
                "alert_id": 2,
                "step": 3,
                "machine_id": 2,
                "sensor": "pressure",
                "sensor_value": 20.0,
                "prediction": 1,
                "anomaly_score": 0.80,
                "severity": "WARNING",
                "alert_type": "model_anomaly",
                "reason": "test alert two",
                "status": "open",
                "anomaly_type": "drop",
                "real_value": 1,
            },
        ]
    )

    alert_events = pd.DataFrame(
        [
            {
                "event_id": 1,
                "machine_id": 1,
                "sensor": "temperature",
                "anomaly_type": "spike",
                "start_step": 2,
                "end_step": 2,
                "duration": 1,
                "alert_count": 1,
                "max_severity": "CRITICAL",
                "status": "open",
            },
            {
                "event_id": 2,
                "machine_id": 2,
                "sensor": "pressure",
                "anomaly_type": "drop",
                "start_step": 3,
                "end_step": 3,
                "duration": 1,
                "alert_count": 1,
                "max_severity": "WARNING",
                "status": "open",
            },
            {
                "event_id": 3,
                "machine_id": 2,
                "sensor": "vibration",
                "anomaly_type": "oscillation",
                "start_step": 4,
                "end_step": 4,
                "duration": 1,
                "alert_count": 1,
                "max_severity": "INFO",
                "status": "open",
            },
        ]
    )

    load_dataframe_to_table(temp_connection, predictions, "model_predictions", run_id)
    load_dataframe_to_table(temp_connection, row_alerts, "row_alerts", run_id)
    load_dataframe_to_table(temp_connection, alert_events, "alert_events", run_id)

    response = client.get(f"/runs/{run_id}/summary")

    assert response.status_code == 200

    data = response.json()

    assert data["run_id"] == run_id
    assert data["total_predictions"] == 3
    assert data["total_anomalies_predicted"] == 2
    assert data["total_row_alerts"] == 2
    assert data["total_alert_events"] == 3
    assert data["critical_alert_events"] == 1
    assert data["warning_alert_events"] == 1
    assert data["info_alert_events"] == 1
    assert data["machines_with_alerts"] == 2
    assert data["max_anomaly_score"] == 0.90
    assert data["mean_anomaly_score"] == pytest.approx(0.60)


def test_read_run_summary_returns_404_when_run_missing(client):
    response = client.get("/runs/999/summary")

    assert response.status_code == 404
    assert response.json()["detail"] == "Pipeline run not found"
    

def test_read_anomaly_type_distribution_for_run(client, temp_connection):
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
            },
            {
                "event_id": 2,
                "machine_id": 1,
                "sensor": "pressure",
                "anomaly_type": "spike",
                "start_step": 20,
                "end_step": 22,
                "duration": 3,
                "alert_count": 2,
                "max_severity": "WARNING",
                "status": "open",
            },
            {
                "event_id": 3,
                "machine_id": 2,
                "sensor": "vibration",
                "anomaly_type": "drift",
                "start_step": 30,
                "end_step": 35,
                "duration": 6,
                "alert_count": 3,
                "max_severity": "WARNING",
                "status": "open",
            },
            {
                "event_id": 4,
                "machine_id": 2,
                "sensor": "voltage",
                "anomaly_type": "drop",
                "start_step": 40,
                "end_step": 45,
                "duration": 6,
                "alert_count": 3,
                "max_severity": "INFO",
                "status": "open",
            },
        ]
    )

    load_dataframe_to_table(temp_connection, events, "alert_events", run_id)

    response = client.get(f"/runs/{run_id}/events/anomaly-type-distribution")

    assert response.status_code == 200

    data = response.json()

    assert data == [
        {"anomaly_type": "spike", "count": 2},
        {"anomaly_type": "drift", "count": 1},
        {"anomaly_type": "drop", "count": 1},
    ]


def test_read_anomaly_type_distribution_returns_404_when_run_missing(client):
    response = client.get("/runs/999/events/anomaly-type-distribution")

    assert response.status_code == 404
    assert response.json()["detail"] == "Pipeline run not found"
    
    
def test_read_sensor_distribution_for_run(client, temp_connection):
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
            },
            {
                "event_id": 2,
                "machine_id": 1,
                "sensor": "temperature",
                "anomaly_type": "drift",
                "start_step": 20,
                "end_step": 22,
                "duration": 3,
                "alert_count": 2,
                "max_severity": "WARNING",
                "status": "open",
            },
            {
                "event_id": 3,
                "machine_id": 2,
                "sensor": "pressure",
                "anomaly_type": "drop",
                "start_step": 30,
                "end_step": 35,
                "duration": 6,
                "alert_count": 3,
                "max_severity": "WARNING",
                "status": "open",
            },
            {
                "event_id": 4,
                "machine_id": 2,
                "sensor": "vibration",
                "anomaly_type": "oscillation",
                "start_step": 40,
                "end_step": 45,
                "duration": 6,
                "alert_count": 3,
                "max_severity": "INFO",
                "status": "open",
            },
        ]
    )

    load_dataframe_to_table(temp_connection, events, "alert_events", run_id)

    response = client.get(f"/runs/{run_id}/events/sensor-distribution")

    assert response.status_code == 200

    data = response.json()

    assert data == [
        {"sensor": "temperature", "count": 2},
        {"sensor": "pressure", "count": 1},
        {"sensor": "vibration", "count": 1},
    ]


def test_read_sensor_distribution_returns_404_when_run_missing(client):
    response = client.get("/runs/999/events/sensor-distribution")

    assert response.status_code == 404
    assert response.json()["detail"] == "Pipeline run not found"
    
    
def test_read_severity_distribution_for_run(client, temp_connection):
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
            {
                "event_id": 3,
                "machine_id": 2,
                "sensor": "vibration",
                "anomaly_type": "oscillation",
                "start_step": 30,
                "end_step": 35,
                "duration": 6,
                "alert_count": 3,
                "max_severity": "CRITICAL",
                "status": "open",
            },
            {
                "event_id": 4,
                "machine_id": 2,
                "sensor": "voltage",
                "anomaly_type": "drift",
                "start_step": 40,
                "end_step": 45,
                "duration": 6,
                "alert_count": 3,
                "max_severity": "INFO",
                "status": "open",
            },
        ]
    )

    load_dataframe_to_table(temp_connection, events, "alert_events", run_id)

    response = client.get(f"/runs/{run_id}/events/severity-distribution")

    assert response.status_code == 200

    data = response.json()

    assert data == [
        {"severity": "CRITICAL", "count": 2},
        {"severity": "WARNING", "count": 1},
        {"severity": "INFO", "count": 1},
    ]


def test_read_severity_distribution_returns_404_when_run_missing(client):
    response = client.get("/runs/999/events/severity-distribution")

    assert response.status_code == 404
    assert response.json()["detail"] == "Pipeline run not found"
    
    
def test_read_dashboard_run(client, temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    predictions = pd.DataFrame(
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
                "anomaly_score": 0.95,
                "threshold": 0.35,
                "anomaly_type": "spike",
                "target_sensor": "temperature",
            },
            {
                "step": 3,
                "machine_id": 2,
                "real_value": 1,
                "prediction": 1,
                "anomaly_score": 0.80,
                "threshold": 0.35,
                "anomaly_type": "drop",
                "target_sensor": "pressure",
            },
        ]
    )

    row_alerts = pd.DataFrame(
        [
            {
                "alert_id": 1,
                "step": 2,
                "machine_id": 1,
                "sensor": "temperature",
                "sensor_value": 105.0,
                "prediction": 1,
                "anomaly_score": 0.95,
                "severity": "CRITICAL",
                "alert_type": "model_and_threshold",
                "reason": "critical temperature spike",
                "status": "open",
                "anomaly_type": "spike",
                "real_value": 1,
            },
            {
                "alert_id": 2,
                "step": 3,
                "machine_id": 2,
                "sensor": "pressure",
                "sensor_value": 20.0,
                "prediction": 1,
                "anomaly_score": 0.80,
                "severity": "WARNING",
                "alert_type": "model_anomaly",
                "reason": "pressure drop",
                "status": "open",
                "anomaly_type": "drop",
                "real_value": 1,
            },
        ]
    )

    alert_events = pd.DataFrame(
        [
            {
                "event_id": 1,
                "machine_id": 1,
                "sensor": "temperature",
                "anomaly_type": "spike",
                "start_step": 2,
                "end_step": 2,
                "duration": 1,
                "alert_count": 1,
                "max_severity": "CRITICAL",
                "max_severity_reason": "critical temperature spike",
                "max_anomaly_score": 0.95,
                "mean_anomaly_score": 0.95,
                "min_sensor_value": 105.0,
                "max_sensor_value": 105.0,
                "first_reason": "critical temperature spike",
                "status": "open",
                "real_value": 1,
            },
            {
                "event_id": 2,
                "machine_id": 2,
                "sensor": "pressure",
                "anomaly_type": "drop",
                "start_step": 3,
                "end_step": 3,
                "duration": 1,
                "alert_count": 1,
                "max_severity": "WARNING",
                "max_severity_reason": "pressure drop",
                "max_anomaly_score": 0.80,
                "mean_anomaly_score": 0.80,
                "min_sensor_value": 20.0,
                "max_sensor_value": 20.0,
                "first_reason": "pressure drop",
                "status": "open",
                "real_value": 1,
            },
            {
                "event_id": 3,
                "machine_id": 2,
                "sensor": "vibration",
                "anomaly_type": "oscillation",
                "start_step": 4,
                "end_step": 4,
                "duration": 1,
                "alert_count": 1,
                "max_severity": "CRITICAL",
                "max_severity_reason": "vibration oscillation",
                "max_anomaly_score": 0.90,
                "mean_anomaly_score": 0.90,
                "min_sensor_value": 9.0,
                "max_sensor_value": 9.0,
                "first_reason": "vibration oscillation",
                "status": "open",
                "real_value": 1,
            },
        ]
    )

    load_dataframe_to_table(temp_connection, predictions, "model_predictions", run_id)
    load_dataframe_to_table(temp_connection, row_alerts, "row_alerts", run_id)
    load_dataframe_to_table(temp_connection, alert_events, "alert_events", run_id)

    response = client.get(f"/dashboard/runs/{run_id}")

    assert response.status_code == 200

    data = response.json()

    assert data["summary"]["run_id"] == run_id
    assert data["summary"]["total_predictions"] == 3
    assert data["summary"]["total_anomalies_predicted"] == 2
    assert data["summary"]["total_row_alerts"] == 2
    assert data["summary"]["total_alert_events"] == 3
    assert data["summary"]["critical_alert_events"] == 2
    assert data["summary"]["warning_alert_events"] == 1
    assert data["summary"]["info_alert_events"] == 0
    assert data["summary"]["machines_with_alerts"] == 2
    assert data["summary"]["max_anomaly_score"] == 0.95
    assert data["summary"]["mean_anomaly_score"] == pytest.approx(0.6166666667)

    assert data["anomaly_type_distribution"] == [
        {"anomaly_type": "drop", "count": 1},
        {"anomaly_type": "oscillation", "count": 1},
        {"anomaly_type": "spike", "count": 1},
    ]

    assert data["sensor_distribution"] == [
        {"sensor": "pressure", "count": 1},
        {"sensor": "temperature", "count": 1},
        {"sensor": "vibration", "count": 1},
    ]

    assert data["severity_distribution"] == [
        {"severity": "CRITICAL", "count": 2},
        {"severity": "WARNING", "count": 1},
    ]

    top_critical_event_ids = [
        event["event_id"] for event in data["top_critical_events"]
    ]

    assert top_critical_event_ids == [1, 3]
    assert all(
        event["max_severity"] == "CRITICAL"
        for event in data["top_critical_events"]
    )


def test_read_dashboard_run_returns_404_when_run_missing(client):
    response = client.get("/dashboard/runs/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Pipeline run not found"
    
    
def test_post_predict_returns_model_prediction(client, monkeypatch):
    def fake_predict_feature_row(features):
        assert features == {"feature_a": 1.0}

        return {
            "prediction": 1,
            "anomaly_score": 0.82,
            "threshold": 0.35,
            "is_anomaly": True,
        }

    monkeypatch.setattr("api.predict_feature_row", fake_predict_feature_row)

    response = client.post(
        "/predict",
        json={
            "features": {
                "feature_a": 1.0,
            }
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "prediction": 1,
        "anomaly_score": 0.82,
        "threshold": 0.35,
        "is_anomaly": True,
    }