import pytest
from pydantic import ValidationError

from copilot.schemas.anomaly import RunSummary
from copilot.schemas.triage import (
    TriageFindingDraft,
    TriageReport,
    TriageReportDraft,
)


def make_run_summary(
    run_id: int = 42,
    total_alert_events: int = 2,
) -> RunSummary:
    return RunSummary(
        run_id=run_id,
        total_predictions=100,
        total_anomalies_predicted=10,
        total_row_alerts=8,
        total_alert_events=total_alert_events,
        critical_alert_events=1,
        warning_alert_events=1,
        info_alert_events=0,
        machines_with_alerts=2,
        max_anomaly_score=0.97,
        mean_anomaly_score=0.74,
    )


def make_finding_draft(
    **overrides,
) -> dict[str, object]:
    values: dict[str, object] = {
        "severity": "critical",
        "machine_id": 3,
        "sensor": "temperature",
        "anomaly_type": "spike",
        "summary": "Critical temperature spike detected.",
        "evidence_ids": ["event-7"],
    }
    values.update(overrides)
    return values


def test_existing_deterministic_triage_report_still_validates():
    report = TriageReport(
        run_id=42,
        status="completed",
        run_summary=make_run_summary(),
        findings=[],
        evidence=[],
    )

    assert report.run_id == 42
    assert report.status == "completed"
    assert report.run_summary is not None
    assert report.run_summary.run_id == 42
    assert report.findings == []
    assert report.evidence == []
    assert report.refusal_reason is None


@pytest.mark.parametrize(
    "status",
    [
        "completed",
        "no_alerts",
        "incomplete_data",
        "refused",
    ],
)
def test_triage_report_accepts_all_supported_statuses(status: str):
    report = TriageReport(
        run_id=None,
        status=status,
        run_summary=None,
        findings=[],
        evidence=[],
    )

    assert report.status == status


def test_triage_report_accepts_nullable_run_id_and_summary():
    report = TriageReport(
        run_id=None,
        status="refused",
        run_summary=None,
        findings=[],
        evidence=[],
        refusal_reason="Unable to complete triage.",
    )

    assert report.run_id is None
    assert report.run_summary is None


def test_triage_report_accepts_refusal_reason():
    report = TriageReport(
        run_id=None,
        status="refused",
        run_summary=None,
        findings=[],
        evidence=[],
        refusal_reason="Insufficient trusted operational data.",
    )

    assert report.refusal_reason == (
        "Insufficient trusted operational data."
    )


def test_triage_report_rejects_invalid_status():
    with pytest.raises(ValidationError):
        TriageReport(
            run_id=42,
            status="invalid",
            run_summary=make_run_summary(),
            findings=[],
            evidence=[],
        )


def test_triage_finding_draft_validates():
    draft = TriageFindingDraft(
        **make_finding_draft()
    )

    assert draft.severity == "critical"
    assert draft.machine_id == 3
    assert draft.sensor == "temperature"
    assert draft.anomaly_type == "spike"
    assert draft.summary == (
        "Critical temperature spike detected."
    )
    assert draft.evidence_ids == ["event-7"]


def test_triage_finding_draft_accepts_explicit_null_anomaly_type():
    draft = TriageFindingDraft(
        **make_finding_draft(anomaly_type=None)
    )

    assert draft.anomaly_type is None


def test_triage_finding_draft_requires_anomaly_type_even_when_nullable():
    values = make_finding_draft()
    values.pop("anomaly_type")

    with pytest.raises(ValidationError):
        TriageFindingDraft(**values)


def test_triage_finding_draft_rejects_extra_fields():
    values = make_finding_draft()
    values["unexpected"] = "value"

    with pytest.raises(ValidationError):
        TriageFindingDraft(**values)


@pytest.mark.parametrize(
    "machine_id",
    [0, -1],
)
def test_triage_finding_draft_requires_positive_machine_id(
    machine_id: int,
):
    with pytest.raises(ValidationError):
        TriageFindingDraft(
            **make_finding_draft(machine_id=machine_id)
        )


def test_triage_finding_draft_rejects_empty_sensor():
    with pytest.raises(ValidationError):
        TriageFindingDraft(
            **make_finding_draft(sensor="")
        )


def test_triage_finding_draft_rejects_empty_summary():
    with pytest.raises(ValidationError):
        TriageFindingDraft(
            **make_finding_draft(summary="")
        )


def test_triage_finding_draft_requires_at_least_one_evidence_id():
    with pytest.raises(ValidationError):
        TriageFindingDraft(
            **make_finding_draft(evidence_ids=[])
        )


@pytest.mark.parametrize(
    "severity",
    [
        "info",
        "high",
        "invalid",
    ],
)
def test_triage_finding_draft_rejects_invalid_severity(
    severity: str,
):
    with pytest.raises(ValidationError):
        TriageFindingDraft(
            **make_finding_draft(severity=severity)
        )


@pytest.mark.parametrize(
    "severity",
    [
        "critical",
        "warning",
        "unknown",
    ],
)
def test_triage_finding_draft_accepts_supported_severity(
    severity: str,
):
    draft = TriageFindingDraft(
        **make_finding_draft(severity=severity)
    )

    assert draft.severity == severity


def test_triage_report_draft_validates():
    draft = TriageReportDraft(
        status="completed",
        findings=[
            TriageFindingDraft(
                **make_finding_draft()
            )
        ],
        refusal_reason=None,
    )

    assert draft.status == "completed"
    assert len(draft.findings) == 1
    assert draft.refusal_reason is None


@pytest.mark.parametrize(
    "status",
    [
        "completed",
        "no_alerts",
        "incomplete_data",
        "refused",
    ],
)
def test_triage_report_draft_accepts_all_supported_statuses(
    status: str,
):
    draft = TriageReportDraft(
        status=status,
        findings=[],
        refusal_reason=None,
    )

    assert draft.status == status


def test_triage_report_draft_requires_refusal_reason_field():
    with pytest.raises(ValidationError):
        TriageReportDraft(
            status="completed",
            findings=[],
        )


def test_triage_report_draft_rejects_extra_fields():
    with pytest.raises(ValidationError):
        TriageReportDraft(
            status="completed",
            findings=[],
            refusal_reason=None,
            run_id=42,
        )


def test_triage_report_draft_rejects_run_summary():
    with pytest.raises(ValidationError):
        TriageReportDraft(
            status="completed",
            findings=[],
            refusal_reason=None,
            run_summary=make_run_summary(),
        )


def test_triage_report_draft_rejects_evidence():
    with pytest.raises(ValidationError):
        TriageReportDraft(
            status="completed",
            findings=[],
            refusal_reason=None,
            evidence=[],
        )


def test_triage_report_draft_rejects_invalid_status():
    with pytest.raises(ValidationError):
        TriageReportDraft(
            status="invalid",
            findings=[],
            refusal_reason=None,
        )