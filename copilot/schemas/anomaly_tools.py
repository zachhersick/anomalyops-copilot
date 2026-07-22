from pydantic import BaseModel, Field

from copilot.schemas.anomaly import (
    LatestRun,
    RunSummary,
    AlertEvent,
    RowAlert,
)


class GetLatestRunInput(BaseModel):
    None


class GetRunSummaryInput(BaseModel):
    run_id: int = Field(gt=0)
    
    
class ListAlertEventsInput(BaseModel):
    run_id: int = Field(gt=0)
    severity: str | None = None
    sensor: str | None = None
    anomaly_type: str | None = None
    limit: int = Field(100, gt=0, le=500)
    offset: int = Field(0, ge=0)
    
    
class GetEventAlertsInput(BaseModel):
    run_id: int = Field(gt=0)
    event_id: int = Field(gt=0)
    
    
class GetLatestRunOutput(BaseModel):
    run: LatestRun
    
    
class GetRunSummaryOutput(BaseModel):
    summary: RunSummary
    
    
class ListAlertEventsOutput(BaseModel):
    events: list[AlertEvent]
    
    
class GetEventAlertsOutput(BaseModel):
    alerts: list[RowAlert]