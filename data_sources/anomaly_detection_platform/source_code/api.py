from fastapi import Depends, FastAPI, HTTPException, Query
from typing import Annotated, Literal
import sqlite3

from db import get_connection
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
    get_filtered_alert_events_for_run,
    get_filtered_predictions_for_run,
    get_run_summary,
    get_anomaly_type_distribution_for_run,
    get_sensor_distribution_for_run,
    get_severity_distribution_for_run,
    get_top_critical_events_for_run,
)


from model_serving import predict_feature_row

from schemas import (
    HealthResponse,
    PipelineRunResponse,
    LatestRunResponse,
    AlertEventResponse,
    RowAlertResponse,
    SensorReadingResponse,
    PredictionResponse,
    RunSummaryResponse,
    AnomalyTypeDistributionResponse,
    SensorDistributionResponse,
    SeverityDistributionResponse,
    DashboardRunResponse,
    PredictionRequest,
    PredictionResultResponse,
)


app = FastAPI(
    title="Industrial Anomaly Detection API",
    description="API for reading pipeline runs, alert events, and anomaly detection outputs.",
    version="0.1.0",
)


def get_db_connection():
    """
    Create a database connection for one API request.
    Close it after the request finishes.
    """
    conn = get_connection()

    try:
        yield conn
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row) -> dict:
    """
    Convert one sqlite3.Row into a normal dictionary.
    """
    return dict(row)


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    """
    Convert a list of sqlite3.Row objects into normal dictionaries.
    """
    return [row_to_dict(row) for row in rows]


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Return basic API health status.
    """
    return {"status": "ok"}


@app.get("/runs", response_model=list[PipelineRunResponse])
def read_pipeline_runs(conn=Depends(get_db_connection)):
    """
    Return all stored pipeline runs, newest first.
    """
    rows = get_pipeline_runs(conn)
    return rows_to_dicts(rows)


@app.get("/runs/latest", response_model=LatestRunResponse)
def read_latest_run(conn=Depends(get_db_connection)):
    """
    Return the latest pipeline run_id.
    """
    latest_run_id = get_latest_run_id(conn)

    if latest_run_id is None:
        raise HTTPException(status_code=404, detail="No pipeline runs found.")

    return {"run_id": latest_run_id}


@app.get("/runs/{run_id}/events", response_model=list[AlertEventResponse])
def read_alert_events_for_run(
    run_id: int,
    severity: str | None = None,
    sensor: str | None = None,
    anomaly_type: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    conn=Depends(get_db_connection),
):
    """
    Return grouped alert events for one pipeline run.
    """
    ensure_run_exists(conn, run_id)
    
    rows = get_filtered_alert_events_for_run(
        conn,
        run_id=run_id,
        severity=severity,
        sensor=sensor,
        anomaly_type=anomaly_type,
        limit=limit,
        offset=offset
    )
    return rows_to_dicts(rows)


@app.get("/runs/{run_id}/events/critical", response_model=list[AlertEventResponse])
def read_critical_alert_events_for_run(
    run_id: int,
    conn=Depends(get_db_connection),
):
    """
    Return critical grouped alert events for one pipeline run.
    """
    ensure_run_exists(conn, run_id)
    
    rows = get_critical_alert_events(conn, run_id)
    return rows_to_dicts(rows)


@app.get("/runs/{run_id}/events/{event_id}/alerts", response_model=list[RowAlertResponse])
def read_row_alerts_for_event(
    run_id: int,
    event_id: int,
    conn=Depends(get_db_connection),
):
    """
    Return individual row alerts inside one grouped alert event.
    """
    ensure_run_exists(conn, run_id)
    ensure_event_exists(conn, event_id, run_id)
    
    rows = get_row_alerts_for_event(conn, run_id, event_id)
    return rows_to_dicts(rows)


@app.get("/runs/{run_id}/machines/{machine_id}/readings", response_model=list[SensorReadingResponse])
def read_sensor_readings_for_machine(
    run_id: int,
    machine_id: int,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    conn=Depends(get_db_connection),
):
    """
    Return sensor readings for one machine in one run.
    """
    ensure_run_exists(conn, run_id)
        
    rows = get_sensor_readings_for_machine(
        conn=conn,
        run_id=run_id,
        machine_id=machine_id,
        limit=limit,
        offset=offset,
    )

    return rows_to_dicts(rows)


@app.get("/runs/{run_id}/predictions", response_model=list[PredictionResponse])
def read_predictions_for_run(
    run_id: int,
    machine_id: int | None = None,
    anomaly_type: str  | None = None,
    target_sensor: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    conn=Depends(get_db_connection),
):
    """
    Return model predictions for one run.
    """
    ensure_run_exists(conn, run_id)
    
    rows = get_filtered_predictions_for_run(
        conn=conn,
        run_id=run_id,
        machine_id=machine_id,
        anomaly_type=anomaly_type,
        target_sensor=target_sensor,
        limit=limit,
        offset=offset,
    )

    return rows_to_dicts(rows)

@app.get("/runs/{run_id}/summary", response_model=RunSummaryResponse)
def read_run_summary(run_id: int, conn=Depends(get_db_connection)):
    ensure_run_exists(conn, run_id)
    
    summary = get_run_summary(conn, run_id)
    
    return summary


@app.get("/runs/{run_id}/events/anomaly-type-distribution", response_model=list[AnomalyTypeDistributionResponse])
def read_anomaly_type_distribution_for_run(run_id: int, conn=Depends(get_db_connection)):
    ensure_run_exists(conn, run_id)
    
    distribution = get_anomaly_type_distribution_for_run(conn, run_id)
    
    return distribution


@app.get("/runs/{run_id}/events/sensor-distribution")
def read_sensor_distribution_for_run(run_id: int, conn=Depends(get_db_connection)):
    ensure_run_exists(conn, run_id)
    
    distribution = get_sensor_distribution_for_run(conn, run_id)
    
    return distribution


@app.get("/runs/{run_id}/events/severity-distribution")
def read_severity_distribution_for_run(run_id: int, conn=Depends(get_db_connection)):
    ensure_run_exists(conn, run_id)
    
    distribution = get_severity_distribution_for_run(conn, run_id)
    
    return distribution


@app.get("/dashboard/runs/{run_id}", response_model=DashboardRunResponse)
def read_dashboard_run(run_id: int, conn=Depends(get_db_connection)):
    ensure_run_exists(conn, run_id)
    
    return {
        "summary": get_run_summary(conn, run_id),
        "anomaly_type_distribution": get_anomaly_type_distribution_for_run(conn, run_id),
        "sensor_distribution": get_sensor_distribution_for_run(conn, run_id),
        "severity_distribution": get_severity_distribution_for_run(conn, run_id),
        "top_critical_events": rows_to_dicts(
            get_top_critical_events_for_run(conn, run_id, limit=5)
        ),
    }
    

@app.post("/predict", response_model=PredictionResultResponse)
def post_model_serve(request: PredictionRequest):
    try:
        response = predict_feature_row(request.features)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    
    return response

def ensure_run_exists(conn, run_id: int):
    response = run_exists(conn, run_id)
    
    if response == False:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    
def ensure_event_exists(conn, event_id, run_id: int):
    response = event_exists(conn, run_id, event_id)
    
    if response == False:
        raise HTTPException(status_code=404, detail="Alert event not found")