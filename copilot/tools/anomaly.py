from typing import NoReturn

from copilot.clients.anomaly_api import (
    AnomalyApiClient,
    AnomalyApiError,
    AnomalyResourceNotFoundError,
)
from copilot.schemas.anomaly_tools import (
    GetLatestRunInput,
    GetLatestRunOutput,
    GetRunSummaryInput,
    GetRunSummaryOutput,
    ListAlertEventsInput,
    ListAlertEventsOutput,
    GetEventAlertsInput,
    GetEventAlertsOutput,
)


class OperationalToolError(Exception):
    pass


class OperationalResourceNotFoundError(OperationalToolError):
    pass


class AnomalyOperationalTools:
    
    
    def __init__(
        self,
        client: AnomalyApiClient,
    ) -> None:
        self._client = client
        
        
    def _raise_operational_error(
        self,
        exc: AnomalyApiError,
    ) -> NoReturn:
        if isinstance(exc, AnomalyResourceNotFoundError):
            raise OperationalResourceNotFoundError(
                "The requested operational resource was not found."
            ) from exc
        raise OperationalToolError(
            "The operational tool call failed."
        ) from exc
        
        
    def get_latest_run(
        self,
        tool_input: GetLatestRunInput,
    ) -> GetLatestRunOutput:
        try:
            run = self._client.get_latest_run()
        except AnomalyApiError as exc:
            self._raise_operational_error(exc)
        
        return GetLatestRunOutput(run=run)
    
    
    def get_run_summary(
        self,
        tool_input: GetRunSummaryInput,
    ) -> GetRunSummaryOutput:
        try:
            summary = self._client.get_run_summary(run_id=tool_input.run_id)
        except AnomalyApiError as exc:
            self._raise_operational_error(exc)
            
        return GetRunSummaryOutput(summary=summary)
    
    
    def list_alert_events(
        self,
        tool_input: ListAlertEventsInput,
    ) -> ListAlertEventsOutput:
        try:
            events = self._client.list_alert_events(
                run_id=tool_input.run_id,
                severity=tool_input.severity,
                sensor=tool_input.sensor,
                anomaly_type=tool_input.anomaly_type,
                limit=tool_input.limit,
                offset=tool_input.offset,
            )
        except AnomalyApiError as exc:
            self._raise_operational_error(exc)
        
        return ListAlertEventsOutput(events=events)
    
    
    def get_event_alerts(
        self,
        tool_input: GetEventAlertsInput,
    ) -> GetEventAlertsOutput:
        try:
            alerts = self._client.get_event_alerts(
                run_id=tool_input.run_id,
                event_id=tool_input.event_id
            )
        except AnomalyApiError as exc:
            self._raise_operational_error(exc)
        
        return GetEventAlertsOutput(alerts=alerts)