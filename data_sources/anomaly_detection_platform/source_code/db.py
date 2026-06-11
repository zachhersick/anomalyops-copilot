import sqlite3
from pathlib import Path

from config import DB_PATH


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    Create and return a SQLite database connection.
    """
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def create_tables(connection: sqlite3.Connection) -> None:
    """
    Create all database tables for storing pipeline outputs.
    """
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            notes TEXT,
            fixed_seed INTEGER,
            model_threshold REAL,
            max_step_gap INTEGER
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sensor_readings (
            reading_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            timestamp TEXT,
            step INTEGER NOT NULL,
            machine_id INTEGER NOT NULL,
            temperature REAL,
            pressure REAL,
            vibration REAL,
            flow_rate REAL,
            voltage REAL,
            current REAL,
            is_anomaly INTEGER,
            anomaly_type TEXT,
            target_sensor TEXT,
            FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS model_predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            step INTEGER NOT NULL,
            machine_id INTEGER NOT NULL,
            real_value INTEGER,
            prediction INTEGER,
            anomaly_score REAL,
            threshold REAL,
            anomaly_type TEXT,
            target_sensor TEXT,
            FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS row_alerts (
            run_id INTEGER NOT NULL,
            alert_id INTEGER NOT NULL,
            step INTEGER NOT NULL,
            machine_id INTEGER NOT NULL,
            sensor TEXT NOT NULL,
            sensor_value REAL,
            prediction INTEGER,
            anomaly_score REAL,
            severity TEXT,
            alert_type TEXT,
            reason TEXT,
            status TEXT,
            anomaly_type TEXT,
            real_value INTEGER,
            PRIMARY KEY (run_id, alert_id),
            FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS alert_events (
            run_id INTEGER NOT NULL,
            event_id INTEGER NOT NULL,
            machine_id INTEGER NOT NULL,
            sensor TEXT NOT NULL,
            anomaly_type TEXT,
            start_step INTEGER,
            end_step INTEGER,
            duration INTEGER,
            alert_count INTEGER,
            max_severity TEXT,
            max_severity_reason TEXT,
            max_anomaly_score REAL,
            mean_anomaly_score REAL,
            min_sensor_value REAL,
            max_sensor_value REAL,
            first_reason TEXT,
            status TEXT,
            real_value INTEGER,
            PRIMARY KEY (run_id, event_id),
            FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
        );
        """
    )
    
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_alert_events_run_start
        ON alert_events(run_id, start_step);
        """
    )
    
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_alert_events_run_max_start
        ON alert_events(run_id, max_severity, start_step);
        """
    )
    
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_alert_events_run_sensor_start
        ON alert_events(run_id, sensor, start_step);
        """
    )
    
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_alert_events_run_anomaly_start
        ON alert_events(run_id, anomaly_type, start_step);
        """
    )
    
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_model_predictions_run_step_machine_target
        ON model_predictions(run_id, step, machine_id, target_sensor);
        """
    )
    
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_model_predictions_run_machine_step
        ON model_predictions(run_id, machine_id, step);
        """
    )
    
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_model_predictions_run_anomaly_step
        ON model_predictions(run_id, anomaly_type, step);
        """
    )
    
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_model_predictions_run_target_step
        ON model_predictions(run_id, target_sensor, step);
        """
    )
    
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_sensor_readings_run_machine_step
        ON sensor_readings(run_id, machine_id, step);
        """
    )

    connection.commit()


def main() -> None:
    connection = get_connection()
    create_tables(connection)
    connection.close()

    print(f"Database initialized: {DB_PATH}")


if __name__ == "__main__":
    main()