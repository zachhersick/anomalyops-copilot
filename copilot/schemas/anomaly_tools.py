from pydantic import BaseModel, Field, ConfigDict

from copilot.schemas.anomaly import (
    LatestRun,
    RunSummary,
    AlertEvent,
    RowAlert,
)


class GetLatestRunInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetRunSummaryInput(BaseModel):
    run_id: int = Field(gt=0)
    model_config = ConfigDict(extra="forbid")
    
    
class ListAlertEventsInput(BaseModel):
    run_id: int = Field(gt=0)
    severity: str | None = None
    sensor: str | None = None
    anomaly_type: str | None = None
    limit: int = Field(100, gt=0, le=500)
    offset: int = Field(0, ge=0)
    model_config = ConfigDict(extra="forbid")
    
    
class GetEventAlertsInput(BaseModel):
    run_id: int = Field(gt=0)
    event_id: int = Field(gt=0)
    model_config = ConfigDict(extra="forbid")
    
    
class GetLatestRunOutput(BaseModel):
    run: LatestRun
    model_config = ConfigDict(extra="forbid")
    
    
class GetRunSummaryOutput(BaseModel):
    summary: RunSummary
    model_config = ConfigDict(extra="forbid")
    
    
class ListAlertEventsOutput(BaseModel):
    events: list[AlertEvent]
    model_config = ConfigDict(extra="forbid")
    
    
class GetEventAlertsOutput(BaseModel):
    alerts: list[RowAlert]
    model_config = ConfigDict(extra="forbid")