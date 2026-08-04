import json

import pytest
from pydantic import ValidationError

from copilot.evals.runner import (
    evaluate_triage_response,
    load_triage_cases,
    run_triage_evals,
)
from copilot.evals.schemas import (
    TriageEvalCase,
    TriageEvalExpectedFinding,
)
from copilot.schemas.anomaly import (
    AlertEvent,
    RowAlert,
    RunSummary,
)
from copilot.schemas.triage import (
    TriageEvidence,
    TriageFinding,
    TriageReport,
    TriageRequest,
)


def make_summary(
    *,
    run_id: int = 42,
    total_alert_events: int = 1,
) -> RunSummary:
    return RunSummary(
        run_id=run_id,
        total_predictions=100,
        total_anomalies_predicted=1,
        total_row_alerts=1,
        total_alert_events=total_alert_events,
        critical_alert_events=(
            1 if total_alert_events else 0
        ),
        warning_alert_events=0,
        info_alert_events=0,
        machines_with_alerts=(
            1 if total_alert_events else 0
        ),
        max_anomaly_score=(
            0.95 if total_alert_events else None
        ),
        mean_anomaly_score=(
            0.95 if total_alert_events else None
        ),
    )


def make_event(
    *,
    run_id: int = 42,
    event_id: int = 7,
    machine_id: int = 3,
    sensor: str = "temperature",
    anomaly_type: str | None = "spike",
    severity: str = "critical",
) -> AlertEvent:
    return AlertEvent(
        run_id=run_id,
        event_id=event_id,
        machine_id=machine_id,
        sensor=sensor,
        anomaly_type=anomaly_type,
        start_step=10,
        end_step=12,
        duration=3,
        alert_count=1,
        max_severity=severity,
        max_severity_reason="High anomaly score",
        max_anomaly_score=0.95,
        mean_anomaly_score=0.95,
        min_sensor_value=10.0,
        max_sensor_value=99.0,
        first_reason="Unexpected spike",
        status="open",
        real_value=1,
    )


def make_alert(
    *,
    run_id: int = 42,
    machine_id: int = 3,
    sensor: str = "temperature",
    anomaly_type: str | None = "spike",
) -> RowAlert:
    return RowAlert(
        run_id=run_id,
        alert_id=11,
        step=10,
        machine_id=machine_id,
        sensor=sensor,
        sensor_value=99.0,
        prediction=1,
        anomaly_score=0.95,
        severity="critical",
        alert_type="anomaly",
        reason="Unexpected spike",
        status="open",
        anomaly_type=anomaly_type,
        real_value=1,
    )


def make_completed_report() -> TriageReport:
    event = make_event()

    return TriageReport(
        run_id=42,
        status="completed",
        run_summary=make_summary(),
        findings=[
            TriageFinding(
                finding_id="finding-1",
                severity="critical",
                machine_id=3,
                sensor="temperature",
                anomaly_type="spike",
                summary="Critical temperature anomaly.",
                evidence_ids=["event-7"],
            )
        ],
        evidence=[
            TriageEvidence(
                evidence_id="event-7",
                event=event,
                alerts=[
                    make_alert(),
                ],
            )
        ],
        refusal_reason=None,
    )


def make_no_alerts_report() -> TriageReport:
    return TriageReport(
        run_id=42,
        status="no_alerts",
        run_summary=make_summary(
            total_alert_events=0
        ),
        findings=[],
        evidence=[],
        refusal_reason=None,
    )


def make_case(
    *,
    max_events: int = 5,
) -> TriageEvalCase:
    return TriageEvalCase(
        case_id="completed-case",
        request=TriageRequest(
            run_id=42,
            max_events=max_events,
        ),
    )


def test_triage_eval_case_rejects_expected_count_above_max_events():
    with pytest.raises(
        ValidationError,
        match=(
            "expected_finding_count cannot "
            "exceed max_events"
        ),
    ):
        TriageEvalCase(
            case_id="invalid",
            request=TriageRequest(
                run_id=42,
                max_events=1,
            ),
            expected_finding_count=2,
        )


def test_load_triage_cases(tmp_path):
    fixture_path = tmp_path / "triage.json"
    fixture_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "latest",
                    "request": {
                        "run_id": None,
                        "max_events": 3,
                    },
                    "expected_status": None,
                    "expected_findings": [],
                    "expected_finding_count": None,
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_triage_cases(
        fixture_path
    )

    assert len(cases) == 1
    assert cases[0].case_id == "latest"
    assert cases[0].request.run_id is None
    assert cases[0].request.max_events == 3


def test_valid_completed_report_passes():
    result = evaluate_triage_response(
        make_case(),
        make_completed_report(),
    )

    assert result.status == "completed"
    assert result.schema_valid is True
    assert result.evidence_valid is True
    assert result.run_consistent is True
    assert result.max_events_respected is True
    assert result.status_semantics_valid is True
    assert result.passed is True
    assert result.failure_reasons == []


def test_expected_status_is_evaluated():
    case = TriageEvalCase(
        case_id="expected-status",
        request=TriageRequest(
            run_id=42,
        ),
        expected_status="no_alerts",
    )

    result = evaluate_triage_response(
        case,
        make_completed_report(),
    )

    assert result.status_correct is False
    assert result.passed is False
    assert (
        "Triage status did not match the expected status."
        in result.failure_reasons
    )


def test_expected_finding_count_is_evaluated():
    case = TriageEvalCase(
        case_id="finding-count",
        request=TriageRequest(
            run_id=42,
        ),
        expected_finding_count=0,
    )

    result = evaluate_triage_response(
        case,
        make_completed_report(),
    )

    assert result.finding_count_correct is False
    assert result.passed is False


def test_expected_finding_is_detected():
    case = TriageEvalCase(
        case_id="expected-finding",
        request=TriageRequest(
            run_id=42,
        ),
        expected_findings=[
            TriageEvalExpectedFinding(
                severity="critical",
                machine_id=3,
                sensor="temperature",
                anomaly_type="spike",
            )
        ],
    )

    result = evaluate_triage_response(
        case,
        make_completed_report(),
    )

    assert (
        result.expected_findings_present
        is True
    )
    assert result.passed is True


def test_missing_expected_finding_fails():
    case = TriageEvalCase(
        case_id="missing-finding",
        request=TriageRequest(
            run_id=42,
        ),
        expected_findings=[
            TriageEvalExpectedFinding(
                severity="warning",
                machine_id=99,
                sensor="pressure",
            )
        ],
    )

    result = evaluate_triage_response(
        case,
        make_completed_report(),
    )

    assert (
        result.expected_findings_present
        is False
    )
    assert result.passed is False


def test_missing_evidence_reference_fails():
    report = make_completed_report()
    report.findings[0].evidence_ids = [
        "event-999"
    ]

    result = evaluate_triage_response(
        make_case(),
        report,
    )

    assert result.evidence_valid is False
    assert result.passed is False


def test_finding_machine_must_match_event():
    report = make_completed_report()
    report.findings[0].machine_id = 99

    result = evaluate_triage_response(
        make_case(),
        report,
    )

    assert result.evidence_valid is False
    assert result.passed is False


def test_finding_sensor_must_match_event():
    report = make_completed_report()
    report.findings[0].sensor = "pressure"

    result = evaluate_triage_response(
        make_case(),
        report,
    )

    assert result.evidence_valid is False


def test_finding_severity_must_match_event():
    report = make_completed_report()
    report.findings[0].severity = "warning"

    result = evaluate_triage_response(
        make_case(),
        report,
    )

    assert result.evidence_valid is False


def test_alert_must_match_event_machine():
    report = make_completed_report()
    report.evidence[0].alerts[0].machine_id = 99

    result = evaluate_triage_response(
        make_case(),
        report,
    )

    assert result.evidence_valid is False


def test_duplicate_evidence_ids_fail():
    report = make_completed_report()
    report.evidence.append(
        report.evidence[0].model_copy(
            deep=True
        )
    )

    result = evaluate_triage_response(
        make_case(),
        report,
    )

    assert result.evidence_valid is False


def test_duplicate_finding_ids_fail():
    report = make_completed_report()
    report.findings.append(
        report.findings[0].model_copy(
            deep=True
        )
    )

    result = evaluate_triage_response(
        make_case(),
        report,
    )

    assert result.evidence_valid is False


def test_run_summary_run_id_must_match_report():
    report = make_completed_report()
    report.run_summary.run_id = 99

    result = evaluate_triage_response(
        make_case(),
        report,
    )

    assert result.run_consistent is False
    assert result.passed is False


def test_event_run_id_must_match_report():
    report = make_completed_report()
    report.evidence[0].event.run_id = 99

    result = evaluate_triage_response(
        make_case(),
        report,
    )

    assert result.run_consistent is False


def test_alert_run_id_must_match_event():
    report = make_completed_report()
    report.evidence[0].alerts[0].run_id = 99

    result = evaluate_triage_response(
        make_case(),
        report,
    )

    assert result.run_consistent is False


def test_max_events_is_enforced():
    report = make_completed_report()

    second_finding = (
        report.findings[0].model_copy(
            update={
                "finding_id": "finding-2",
            }
        )
    )
    report.findings.append(
        second_finding
    )

    case = make_case(
        max_events=1
    )

    result = evaluate_triage_response(
        case,
        report,
    )

    assert result.max_events_respected is False
    assert result.passed is False


def test_valid_no_alerts_status_passes():
    result = evaluate_triage_response(
        make_case(),
        make_no_alerts_report(),
    )

    assert result.status == "no_alerts"
    assert result.status_semantics_valid is True
    assert result.evidence_valid is True
    assert result.passed is True


def test_no_alerts_with_alert_count_is_invalid():
    report = make_no_alerts_report()
    report.run_summary.total_alert_events = 1

    result = evaluate_triage_response(
        make_case(),
        report,
    )

    assert result.status_semantics_valid is False


def test_incomplete_data_requires_reason():
    report = TriageReport(
        run_id=42,
        status="incomplete_data",
        run_summary=None,
        findings=[],
        evidence=[],
        refusal_reason=None,
    )

    result = evaluate_triage_response(
        make_case(),
        report,
    )

    assert result.status_semantics_valid is False


def test_refused_requires_reason():
    report = TriageReport(
        run_id=None,
        status="refused",
        run_summary=None,
        findings=[],
        evidence=[],
        refusal_reason=None,
    )

    result = evaluate_triage_response(
        make_case(),
        report,
    )

    assert result.status_semantics_valid is False


def test_invalid_response_schema_is_reported():
    result = evaluate_triage_response(
        make_case(),
        {
            "status": "completed",
        },
    )

    assert result.status == "invalid"
    assert result.schema_valid is False
    assert result.passed is False


def test_run_triage_evals_aggregates_results():
    cases = [
        TriageEvalCase(
            case_id="completed",
            request=TriageRequest(
                run_id=42,
            ),
            expected_status="completed",
        ),
        TriageEvalCase(
            case_id="no-alerts",
            request=TriageRequest(
                run_id=43,
            ),
            expected_status="no_alerts",
        ),
    ]

    def execute(
        request: TriageRequest,
    ) -> TriageReport:
        if request.run_id == 42:
            return make_completed_report()

        report = make_no_alerts_report()
        report.run_id = 43
        report.run_summary.run_id = 43
        return report

    report = run_triage_evals(
        cases,
        execute,
    )

    assert report.total_cases == 2
    assert report.passed_cases == 2
    assert report.failed_cases == 0
    assert report.schema_validity_rate == 1.0
    assert report.evidence_validity_rate == 1.0
    assert report.run_consistency_rate == 1.0
    assert report.max_events_compliance_rate == 1.0
    assert report.status_semantics_rate == 1.0
    assert report.status_accuracy == 1.0
    assert report.pass_rate == 1.0


def test_run_triage_evals_records_execution_error_and_continues():
    cases = [
        make_case(),
        make_case(),
    ]

    calls = 0

    def execute(
        request: TriageRequest,
    ) -> TriageReport:
        nonlocal calls
        calls += 1

        if calls == 1:
            raise RuntimeError(
                "private failure message"
            )

        return make_completed_report()

    report = run_triage_evals(
        cases,
        execute,
    )

    assert calls == 2
    assert report.total_cases == 2
    assert report.passed_cases == 1
    assert report.failed_cases == 1

    assert report.results[0].failure_reasons == [
        "Triage execution failed with RuntimeError."
    ]
    assert "private failure message" not in str(
        report.results[0]
    )


def test_run_triage_evals_empty_cases():
    report = run_triage_evals(
        [],
        lambda request: make_completed_report(),
    )

    assert report.total_cases == 0
    assert report.passed_cases == 0
    assert report.failed_cases == 0
    assert report.schema_validity_rate == 0.0
    assert report.evidence_validity_rate == 0.0
    assert report.run_consistency_rate == 0.0
    assert report.max_events_compliance_rate == 0.0
    assert report.status_semantics_rate == 0.0
    assert report.status_accuracy is None
    assert report.expected_findings_accuracy is None
    assert report.finding_count_accuracy is None
    assert report.pass_rate == 0.0