from pathlib import Path
import sqlite3

import pandas as pd

from db import create_tables, get_connection
from config import DB_PATH


SENSOR_DATA_RAW_PATH = Path("outputs/sensor_data_raw.csv")
PREDICTIONS_PATH = Path("outputs/predictions.csv")
ALERTS_PATH = Path("outputs/alerts.csv")
ALERT_EVENTS_PATH = Path("outputs/alert_events.csv")


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    """
    Return the column names for a SQLite table.
    """
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()

    if not rows:
        raise ValueError(f"Table '{table_name}' does not exist or has no columns.")

    return {row[1] for row in rows}


def insert_pipeline_run(
    conn: sqlite3.Connection,
    notes: str = "Loaded CSV outputs into SQLite.",
    fixed_seed: int = 295,
    model_threshold: float = 0.35,
    max_step_gap: int = 3,
) -> int:
    """
    Insert one pipeline run and return its run_id.
    """
    cursor = conn.execute(
        """
        INSERT INTO pipeline_runs (
            notes,
            fixed_seed,
            model_threshold,
            max_step_gap
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            notes,
            fixed_seed,
            model_threshold,
            max_step_gap,
        ),
    )

    conn.commit()
    return cursor.lastrowid


def load_dataframe_to_table(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    table_name: str,
    run_id: int,
) -> None:
    """
    Load a dataframe into a SQLite table.

    Only columns that exist in both the dataframe and database table are inserted.
    A run_id column is added when the table supports it.
    """
    table_columns = get_table_columns(conn, table_name)

    df_to_insert = df.copy()

    if "run_id" in table_columns:
        df_to_insert["run_id"] = run_id

    matching_columns = [
        col for col in df_to_insert.columns
        if col in table_columns
    ]

    if not matching_columns:
        raise ValueError(
            f"No matching columns found between dataframe and table '{table_name}'."
        )

    df_to_insert = df_to_insert[matching_columns]

    df_to_insert.to_sql(
        table_name,
        conn,
        if_exists="append",
        index=False,
    )

    conn.commit()


def load_csv_if_exists(
    conn: sqlite3.Connection,
    csv_path: Path,
    table_name: str,
    run_id: int,
) -> None:
    """
    Load one CSV file into one SQLite table if the CSV exists.
    """
    if not csv_path.exists():
        print(f"Skipped missing file: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if df.empty:
        print(f"Skipped empty file: {csv_path}")
        return

    load_dataframe_to_table(conn, df, table_name, run_id)

    print(f"Loaded {len(df)} rows into {table_name} from {csv_path}")


def main() -> None:
    """
    Create database tables and load existing pipeline CSV outputs into SQLite.
    """
    with get_connection(DB_PATH) as conn:
        create_tables(conn)

        run_id = insert_pipeline_run(conn)

        load_csv_if_exists(
            conn=conn,
            csv_path=SENSOR_DATA_RAW_PATH,
            table_name="sensor_readings",
            run_id=run_id,
        )

        load_csv_if_exists(
            conn=conn,
            csv_path=PREDICTIONS_PATH,
            table_name="model_predictions",
            run_id=run_id,
        )

        load_csv_if_exists(
            conn=conn,
            csv_path=ALERTS_PATH,
            table_name="row_alerts",
            run_id=run_id,
        )

        load_csv_if_exists(
            conn=conn,
            csv_path=ALERT_EVENTS_PATH,
            table_name="alert_events",
            run_id=run_id,
        )

    print(f"Finished loading CSV outputs into {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()