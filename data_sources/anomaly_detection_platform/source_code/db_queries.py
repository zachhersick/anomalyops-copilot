import sqlite3

def get_latest_run_id(conn):
    """
    Return the most recent pipeline run_id
    """
    row = conn.execute(
        """
        SELECT run_id
        FROM pipeline_runs
        ORDER BY run_id DESC
        LIMIT 1
        """
    ).fetchone()
    
    if row is None:
        return None
    
    return row['run_id']

def get_pipeline_runs(conn):
    """
    return all pipeline runs
    """
    rows = conn.execute(
        """
        SELECT *
        FROM pipeline_runs
        ORDER BY run_id DESC
        """
    ).fetchall()
    
    return rows

def get_alert_events_for_run(conn, run_id):
    """
    return grouped alert events for one pipeline run
    """
    rows = conn.execute(
        """
        SELECT *
        FROM alert_events
        WHERE run_id = ?
        ORDER BY start_step ASC
        """
    , (run_id, )).fetchall()
    
    return rows

def get_critical_alert_events(conn, run_id):
    """
    return only critical alert events for one run
    """
    rows = conn.execute(
        """
        SELECT *
        FROM alert_events
        WHERE run_id = ?
            AND max_severity = ?
        ORDER BY start_step ASC
        """
    , (run_id, 'CRITICAL', )).fetchall()
    
    return rows

def get_row_alerts_for_event(conn, run_id, event_id):
    """
    given one grouped event, return the individual row alerts inside that event
    """
    event_row = conn.execute(
        """
        SELECT *
        FROM alert_events
        WHERE run_id = ?
            AND event_id = ?
        """
    , (run_id, event_id, )).fetchone()
    
    if event_row is None:
        return []
    
    event_machine_id = event_row['machine_id']
    event_sensor = event_row['sensor']
    event_anomaly_type = event_row['anomaly_type']
    event_start_step = event_row['start_step']
    event_end_step = event_row['end_step']
    
    alert_rows = conn.execute(
        """
        SELECT *
        FROM row_alerts
        WHERE run_id = ?
            AND machine_id = ?
            AND sensor = ?
            AND anomaly_type = ?
            AND step BETWEEN ? AND ?
        ORDER BY step ASC
        """
    , (run_id, event_machine_id, event_sensor, event_anomaly_type, event_start_step, event_end_step, )).fetchall()
    
    return alert_rows

def get_sensor_readings_for_machine(
    conn,
    run_id: int,
    machine_id: int,
    limit: int | None = None,
    offset: int = 0,
):
    """
    return sensor readings for one machine in one run
    """
    if limit is None:
        rows = conn.execute(
            """
            SELECT *
            FROM sensor_readings
            WHERE run_id = ?
                AND machine_id = ?
            ORDER BY step ASC
            """
        , (run_id, machine_id, )).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT *
            FROM sensor_readings
            WHERE run_id = ?
                AND machine_id = ?
            ORDER BY step ASC
            LIMIT ?
            OFFSET ?
            """
        , (run_id, machine_id, limit, offset, )).fetchall()
    
    return rows

def get_predictions_for_run(conn, run_id, limit=None):
    """
    return model predictions for one run
    """
    return get_filtered_predictions_for_run(
        conn=conn,
        run_id=run_id,
        limit=limit,
    )

def run_exists(conn, run_id):
    """
    Return True if a pipeline run exists.
    Return False otherwise.
    """
    row = conn.execute(
        """
        SELECT 1
        FROM pipeline_runs
        WHERE run_id = ?
        LIMIT 1
        """,
        (run_id,),
    ).fetchone()

    return row is not None

def event_exists(conn,run_id, event_id):
    """
    Return True if an alert event exists.
    Return False otherwise.
    """
    row = conn.execute(
        """
        SELECT 1
        FROM alert_events
        WHERE run_id = ?
            AND event_id = ?
        LIMIT 1
        """,
        (run_id, event_id, ),
    ).fetchone()

    return row is not None

def get_filtered_alert_events_for_run(
    conn,
    run_id,
    severity=None,
    sensor=None,
    anomaly_type=None,
    limit=100,
    offset: int = 0,
):
    sql = """
        SELECT *
        FROM alert_events
        WHERE run_id = ?
    """
    
    params = [run_id]
    
    if severity is not None:
        sql += " AND max_severity = ?"
        params.append(severity)
    
    if sensor is not None:
        sql += " AND sensor = ?"
        params.append(sensor)
    
    if anomaly_type is not None:
        sql += " AND anomaly_type = ?"
        params.append(anomaly_type)
        
    sql += " ORDER BY start_step ASC"
    
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.append(limit)
        params.append(offset)
    
    rows = conn.execute(sql, params).fetchall()
    return rows

def get_filtered_predictions_for_run(
    conn, 
    run_id: int,
    machine_id: int | None = None,
    anomaly_type: str | None = None,
    target_sensor: str | None = None, 
    limit=None,
    offset: int = 0,
):
    sql = """
        SELECT *
        FROM model_predictions
        WHERE run_id = ?
    """
    
    params = [run_id]
    
    if machine_id is not None:
        sql += " AND machine_id = ?"
        params.append(machine_id)
    
    if anomaly_type is not None:
        sql += " AND anomaly_type = ?"
        params.append(anomaly_type)
    
    if target_sensor is not None:
        sql += " AND target_sensor = ?"
        params.append(target_sensor)
    
    sql += " ORDER BY step ASC, machine_id ASC, target_sensor ASC"
    
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.append(limit)
        params.append(offset)
    
    rows = conn.execute(sql, params).fetchall()
    return rows

def get_run_summary(conn, run_id: int):
    row = conn.execute(
        """
        SELECT COUNT(*) AS total_predictions
        FROM model_predictions
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()

    total_predictions = row["total_predictions"]
    
    row = conn.execute(
        """
        SELECT COUNT(*) AS total_anomalies_predicted
        FROM model_predictions
        WHERE run_id = ?
            AND prediction = 1
        """,
        (run_id,),
    ).fetchone()
    
    total_anomalies_predicted = row['total_anomalies_predicted']
    
    row = conn.execute(
        """
        SELECT
            MAX(anomaly_score) AS max_anomaly_score,
            AVG(anomaly_score) AS mean_anomaly_score
        FROM model_predictions
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    
    max_anomaly_score = row['max_anomaly_score']
    mean_anomaly_score = row['mean_anomaly_score']
    
    rows = conn.execute(
        """
        SELECT max_severity, COUNT(*) AS count
        FROM alert_events
        WHERE run_id = ?
        GROUP BY max_severity
        """,
        (run_id,),
    ).fetchall()

    severity_counts = {
        "CRITICAL": 0,
        "WARNING": 0,
        "INFO": 0,
    }

    for row in rows:
        severity_counts[row["max_severity"]] = row["count"]
        
    row = conn.execute(
        """
        SELECT COUNT(*) as total_row_alerts
        FROM row_alerts
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    
    total_row_alerts = row['total_row_alerts']
    
    row = conn.execute(
        """
        SELECT COUNT(*) as total_alert_events
        FROM alert_events
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    
    total_alert_events = row['total_alert_events']
    
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT machine_id) AS machines_with_alerts
        FROM alert_events
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()

    machines_with_alerts = row["machines_with_alerts"]
        
    return {
        "run_id": run_id,
        "total_predictions": total_predictions,
        "total_anomalies_predicted": total_anomalies_predicted,
        "total_row_alerts": total_row_alerts,
        "total_alert_events": total_alert_events,
        "critical_alert_events": severity_counts["CRITICAL"],
        "warning_alert_events": severity_counts["WARNING"],
        "info_alert_events": severity_counts["INFO"],
        "machines_with_alerts": machines_with_alerts,
        "max_anomaly_score": max_anomaly_score,
        "mean_anomaly_score": mean_anomaly_score,
    }
    
    
def get_anomaly_type_distribution_for_run(conn, run_id: int):
    rows = conn.execute(
        """
        SELECT anomaly_type, COUNT(*) AS count
        FROM alert_events
        WHERE run_id = ?
            AND anomaly_type IS NOT NULL
        GROUP BY anomaly_type
        ORDER BY count DESC, anomaly_type ASC
        """,
        (run_id,),
    ).fetchall()
    
    distribution = []
    
    for row in rows:
        distribution.append(
            {
                'anomaly_type': row['anomaly_type'],
                'count': row['count'],
            }
        )
        
    return distribution


def get_sensor_distribution_for_run(conn, run_id: int):
    rows = conn.execute(
        """
        SELECT sensor, COUNT(*) AS count
        FROM alert_events
        WHERE run_id = ?
            AND sensor IS NOT NULL
        GROUP BY sensor
        ORDER BY count DESC, sensor ASC
        """,
        (run_id,),
    ).fetchall()
    
    distribution = []
    
    for row in rows:
        distribution.append(
            {
                'sensor': row['sensor'],
                'count': row['count'],
            }
        )
        
    return distribution


def get_severity_distribution_for_run(conn, run_id: int):
    rows = conn.execute(
        """
        SELECT max_severity, COUNT(*) as count
        FROM alert_events
        WHERE run_id = ?
            AND max_severity IS NOT NULL
        GROUP BY max_severity
        ORDER BY
            CASE max_severity
                WHEN 'CRITICAL' THEN 1
                WHEN 'WARNING' THEN 2
                WHEN 'INFO' THEN 3
                ELSE 4
            END
        """,
        (run_id,),
    ).fetchall()
    
    distribution = []
    
    for row in rows:
        distribution.append(
            {
                'severity': row['max_severity'],
                'count': row['count'],
            }
        )
        
    return distribution


def get_top_critical_events_for_run(conn, run_id: int, limit: int = 5):
    rows = conn.execute(
        """
        SELECT *
        FROM alert_events
        WHERE run_id = ?
            AND max_severity = ?
        ORDER BY max_anomaly_score DESC, start_step ASC
        LIMIT ?
        """,
        (run_id, 'CRITICAL', limit,),
    ).fetchall()
    
    return rows