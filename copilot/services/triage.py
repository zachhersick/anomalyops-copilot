from copilot.tools.anomaly import (
    AnomalyOperationalTools,
    OperationalResourceNotFoundError,
    OperationalToolError,
)
from copilot.schemas.triage import (
    TriageRequest,
    TriageReport,
    TriageEvidence,
    TriageFinding,
)
from copilot.schemas.anomaly_tools import (
    GetLatestRunInput,
    GetRunSummaryInput,
    ListAlertEventsInput,
    GetEventAlertsInput,
)


class TriageServiceError(Exception):
    pass


class TriageRunNotFoundError(TriageServiceError):
    pass


class TriageService:
    
    
    def __init__(
        self,
        tools: AnomalyOperationalTools,
    ) -> None:
        self._tool = tools
        
        
    def triage(
        self,
        request: TriageRequest,
    ) -> TriageReport:
        try:
            if request.run_id is None:
                result = self._tool.get_latest_run(GetLatestRunInput())
                run_id = result.run.run_id
            else:
                run_id = request.run_id
            
            summary_result = self._tool.get_run_summary(
                GetRunSummaryInput(
                    run_id=run_id
                )
            )
            summary = summary_result.summary
            
            if summary.total_alert_events == 0:
                return TriageReport(
                    run_id=run_id,
                    status="no_alerts",
                    run_summary=summary,
                    findings=[],
                    evidence=[],
                )
            
            critical_result = self._tool.list_alert_events(
                ListAlertEventsInput(
                    run_id=run_id,
                    severity="critical",
                    limit=request.max_events,
                    offset=0,
                )
            )
            
            warning_result = self._tool.list_alert_events(
                ListAlertEventsInput(
                    run_id=run_id,
                    severity="warning",
                    limit=request.max_events,
                    offset=0,
                )
            )
            
            events = critical_result.events + warning_result.events
            
            severity_rank = {
                "critical": 0,
                "warning": 1,
            }
            
            selected_events = sorted(
                events,
                key=lambda event: (
                    severity_rank.get(
                        (event.max_severity or "").lower(),
                        2,
                    ),
                    event.max_anomaly_score is None,
                    -(
                        event.max_anomaly_score
                        if event.max_anomaly_score is not None
                        else 0.0
                    ),
                    event.event_id,
                ),
            )[:request.max_events]
            
            findings: list[TriageFinding] = []
            evidence: list[TriageEvidence] = []
            
            for event in selected_events:
                alerts_result = self._tool.get_event_alerts(
                    GetEventAlertsInput(
                        run_id=run_id,
                        event_id=event.event_id,
                    )
                )
                
                sorted_alerts = sorted(
                    alerts_result.alerts,
                    key=lambda alert: (
                        alert.step,
                        alert.alert_id,
                    ),
                )
                
                evidence_id = f"event-{event.event_id}"
                
                evidence.append(
                    TriageEvidence(
                        evidence_id=evidence_id,
                        event=event,
                        alerts=sorted_alerts,
                    )
                )
                
                severity = event.max_severity or "unknown"
                anomaly_type = event.anomaly_type or "alert"
                
                findings.append(
                    TriageFinding(
                        finding_id=f"finding-{event.event_id}",
                        severity=severity,
                        machine_id=event.machine_id,
                        sensor=event.sensor,
                        anomaly_type=event.anomaly_type,
                        summary=(
                            f"{severity.capitalize()} "
                            f"{anomaly_type} event on machine "
                            f"{event.machine_id} "
                            f"{event.sensor} sensor."
                        ),
                        evidence_ids=[evidence_id],
                    )
                )
                
            return TriageReport(
                run_id=run_id,
                status="completed",
                run_summary=summary,
                findings=findings,
                evidence=evidence,
            )
            
        except OperationalResourceNotFoundError as exc:
            raise TriageRunNotFoundError(
                "The requested run was not found."
            ) from exc
        except OperationalToolError as exc:
            raise TriageServiceError(
                "Triage failed while retrieving operational evidence."
            ) from exc