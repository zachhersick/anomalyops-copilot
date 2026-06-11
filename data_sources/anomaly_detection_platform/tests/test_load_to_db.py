import sqlite3

import pandas as pd
import pytest

from db import create_tables
from load_to_db import (
    insert_pipeline_run,
    load_dataframe_to_table,
    load_csv_if_exists,
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


def test_insert_pipeline_run_creates_row(temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    assert run_id == 1

    row = temp_connection.execute(
        """
        SELECT *
        FROM pipeline_runs
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()

    assert row is not None
    assert row["notes"] == "Loaded CSV outputs into SQLite."
    assert row["fixed_seed"] == 295
    assert row["model_threshold"] == 0.35
    assert row["max_step_gap"] == 3


def test_load_dataframe_to_table_loads_sensor_rows(temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    df = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01 00:00:00",
                "step": 1,
                "machine_id": 1,
                "temperature": 75.0,
                "pressure": 50.0,
                "vibration": 2.5,
                "flow_rate": 1.2,
                "voltage": 120.0,
                "current": 10.0,
                "is_anomaly": 0,
                "anomaly_type": "normal",
                "target_sensor": "temperature",
            }
        ]
    )

    load_dataframe_to_table(
        conn=temp_connection,
        df=df,
        table_name="sensor_readings",
        run_id=run_id,
    )

    row = temp_connection.execute(
        """
        SELECT *
        FROM sensor_readings
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()

    assert row is not None
    assert row["run_id"] == run_id
    assert row["step"] == 1
    assert row["machine_id"] == 1
    assert row["temperature"] == 75.0
    assert row["target_sensor"] == "temperature"


def test_load_dataframe_to_table_adds_run_id(temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    df = pd.DataFrame(
        [
            {
                "step": 10,
                "machine_id": 2,
                "real_value": 1,
                "prediction": 1,
                "anomaly_score": 0.92,
                "threshold": 0.35,
                "anomaly_type": "spike",
                "target_sensor": "pressure",
            }
        ]
    )

    load_dataframe_to_table(
        conn=temp_connection,
        df=df,
        table_name="model_predictions",
        run_id=run_id,
    )

    row = temp_connection.execute(
        """
        SELECT *
        FROM model_predictions
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()

    assert row is not None
    assert row["run_id"] == run_id
    assert row["step"] == 10
    assert row["machine_id"] == 2
    assert row["prediction"] == 1
    assert row["anomaly_score"] == 0.92


def test_load_dataframe_to_table_ignores_extra_columns(temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    df = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01 00:00:00",
                "step": 5,
                "machine_id": 1,
                "temperature": 80.0,
                "pressure": 55.0,
                "vibration": 3.0,
                "flow_rate": 1.4,
                "voltage": 121.0,
                "current": 11.0,
                "is_anomaly": 0,
                "anomaly_type": "normal",
                "target_sensor": "voltage",
                "extra_debug_column": "should not be inserted",
            }
        ]
    )

    load_dataframe_to_table(
        conn=temp_connection,
        df=df,
        table_name="sensor_readings",
        run_id=run_id,
    )

    row = temp_connection.execute(
        """
        SELECT *
        FROM sensor_readings
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()

    assert row is not None
    assert row["step"] == 5
    assert row["machine_id"] == 1
    assert row["voltage"] == 121.0


def test_load_csv_if_exists_skips_missing_file(temp_connection, tmp_path):
    run_id = insert_pipeline_run(temp_connection)

    missing_csv_path = tmp_path / "missing_file.csv"

    load_csv_if_exists(
        conn=temp_connection,
        csv_path=missing_csv_path,
        table_name="sensor_readings",
        run_id=run_id,
    )

    row_count = temp_connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM sensor_readings
        """
    ).fetchone()["count"]

    assert row_count == 0


def test_row_alerts_allows_same_alert_id_across_different_runs(temp_connection):
    run_id_1 = insert_pipeline_run(temp_connection)
    run_id_2 = insert_pipeline_run(temp_connection)

    df = pd.DataFrame(
        [
            {
                "alert_id": 1,
                "step": 20,
                "machine_id": 1,
                "sensor": "temperature",
                "sensor_value": 105.0,
                "prediction": 1,
                "anomaly_score": 0.88,
                "severity": "CRITICAL",
                "alert_type": "model_and_threshold",
                "reason": "High temperature anomaly",
                "status": "open",
                "anomaly_type": "spike",
                "real_value": 1,
            }
        ]
    )

    load_dataframe_to_table(
        conn=temp_connection,
        df=df,
        table_name="row_alerts",
        run_id=run_id_1,
    )

    load_dataframe_to_table(
        conn=temp_connection,
        df=df,
        table_name="row_alerts",
        run_id=run_id_2,
    )

    row_count = temp_connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM row_alerts
        WHERE alert_id = 1
        """
    ).fetchone()["count"]

    assert row_count == 2