from pydantic import BaseModel


class LatestRun(BaseModel):
    run_id: int
    
    
class RunSummary(BaseModel):
    run_id: int
    total_predictions: int
    total_anomalies_predicted: int
    total_row_alerts: int
    total_alert_events: int
    critical_alert_events: int
    warning_alert_events: int
    info_alert_events: int
    machines_with_alerts: int
    max_anomaly_score: float | None
    mean_anomaly_score: float | None
    
    
class AlertEvent(BaseModel):
    run_id: int
    event_id: int
    machine_id: int
    sensor: str
    anomaly_type: str | None
    start_step: int | None
    end_step: int | None
    duration: int | None
    alert_count: int | None
    max_severity: str | None
    max_severity_reason: str | None
    max_anomaly_score: float | None
    mean_anomaly_score: float | None
    min_sensor_value: float | None
    max_sensor_value: float | None
    first_reason: str | None
    status: str | None
    real_value: int | None
    
    
class RowAlert(BaseModel):
    run_id: int
    alert_id: int
    step: int
    machine_id: int
    sensor: str
    sensor_value: float | None
    prediction: int | None
    anomaly_score: float | None
    severity: str | None
    alert_type: str | None
    reason: str | None
    status: str | None
    anomaly_type: str | None
    real_value: int | None