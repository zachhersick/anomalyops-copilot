import sqlite3

import pytest

from db import create_tables


@pytest.fixture
def temp_connection(tmp_path):
    """
    Create a temporary SQLite database for each test.
    """
    db_path = tmp_path / "test_anomaly_detection.db"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    yield conn

    conn.close()


def test_create_tables_creates_expected_indexes(temp_connection):
    create_tables(temp_connection)

    rows = temp_connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'index'
        """
    ).fetchall()

    index_names = {row["name"] for row in rows}

    expected_indexes = {
        "idx_alert_events_run_start",
        "idx_alert_events_run_max_start",
        "idx_alert_events_run_sensor_start",
        "idx_alert_events_run_anomaly_start",
        "idx_model_predictions_run_step_machine_target",
        "idx_model_predictions_run_machine_step",
        "idx_model_predictions_run_anomaly_step",
        "idx_model_predictions_run_target_step",
        "idx_sensor_readings_run_machine_step",
    }

    assert expected_indexes.issubset(index_names)