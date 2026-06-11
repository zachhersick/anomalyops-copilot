from pydantic import BaseModel


class PipelineRunResponse(BaseModel):
    """
    Response model for one pipeline run.
    """
    run_id: int
    created_at: str
    notes: str | None = None
    fixed_seed: int | None = None
    model_threshold: float | None = None
    max_step_gap: int | None = None


class LatestRunResponse(BaseModel):
    """
    Response model for the latest run.
    """
    run_id: int


class HealthResponse(BaseModel):
    """
    Response model for health check.
    """
    status: str


class AlertEventResponse(BaseModel):
    """
    Response model for one grouped alert event.
    """
    run_id: int
    event_id: int
    machine_id: int
    sensor: str
    anomaly_type: str | None = None
    start_step: int | None = None
    end_step: int | None = None
    duration: int | None = None
    alert_count: int | None = None
    max_severity: str | None = None
    max_severity_reason: str | None = None
    max_anomaly_score: float | None = None
    mean_anomaly_score: float | None = None
    min_sensor_value: float | None = None
    max_sensor_value: float | None = None
    first_reason: str | None = None
    status: str | None = None
    real_value: int | None = None


class RowAlertResponse(BaseModel):
    """
    Response model for one row-level alert.
    """
    run_id: int
    alert_id: int
    step: int
    machine_id: int
    sensor: str
    sensor_value: float | None = None
    prediction: int | None = None
    anomaly_score: float | None = None
    severity: str | None = None
    alert_type: str | None = None
    reason: str | None = None
    status: str | None = None
    anomaly_type: str | None = None
    real_value: int | None = None


class SensorReadingResponse(BaseModel):
    """
    Response model for one sensor reading.
    """
    reading_id: int
    run_id: int
    timestamp: str | None = None
    step: int
    machine_id: int
    temperature: float | None = None
    pressure: float | None = None
    vibration: float | None = None
    flow_rate: float | None = None
    voltage: float | None = None
    current: float | None = None
    is_anomaly: int | None = None
    anomaly_type: str | None = None
    target_sensor: str | None = None


class PredictionResponse(BaseModel):
    """
    Response model for one model prediction.
    """
    prediction_id: int
    run_id: int
    step: int
    machine_id: int
    real_value: int | None = None
    prediction: int | None = None
    anomaly_score: float | None = None
    threshold: float | None = None
    anomaly_type: str | None = None
    target_sensor: str | None = None
    
class RunSummaryResponse(BaseModel):
    """
    Response model for one run summary
    """
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
    
class AnomalyTypeDistributionResponse(BaseModel):
    anomaly_type: str
    count: int
    
class SensorDistributionResponse(BaseModel):
    sensor: str
    count: int
    
class SeverityDistributionResponse(BaseModel):
    severity: str
    count: int
    
class DashboardRunResponse(BaseModel):
    summary: RunSummaryResponse
    anomaly_type_distribution: list[AnomalyTypeDistributionResponse]
    sensor_distribution: list[SensorDistributionResponse]
    severity_distribution: list[SeverityDistributionResponse]
    top_critical_events: list[AlertEventResponse]
    
    
class PredictionRequest(BaseModel):
    features: dict[str, float]

class PredictionResultResponse(BaseModel):
    prediction: int
    anomaly_score: float
    threshold: float
    is_anomaly: bool