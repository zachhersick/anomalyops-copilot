from unittest.mock import Mock

import pytest

from copilot.providers.deterministic_triage import (
    DeterministicTriageAgent,
)
from copilot.providers.errors import (
    TriageAgentResourceNotFoundError,
    TriageAgentToolError,
)
from copilot.providers.interfaces import ToolCallingTriageAgent
from copilot.schemas.anomaly import RunSummary
from copilot.schemas.triage import TriageReport, TriageRequest
from copilot.services.triage import (
    TriageRunNotFoundError,
    TriageService,
    TriageServiceError,
)
from copilot.tools.anomaly import AnomalyOperationalTools


def make_report() -> TriageReport:
    return TriageReport(
        run_id=42,
        status="no_alerts",
        run_summary=RunSummary(
            run_id=42,
            total_predictions=100,
            total_anomalies_predicted=0,
            total_row_alerts=0,
            total_alert_events=0,
            critical_alert_events=0,
            warning_alert_events=0,
            info_alert_events=0,
            machines_with_alerts=0,
            max_anomaly_score=None,
            mean_anomaly_score=None,
        ),
        findings=[],
        evidence=[],
    )


def test_deterministic_triage_agent_satisfies_protocol():
    tools = Mock(spec=AnomalyOperationalTools)

    agent = DeterministicTriageAgent(tools)

    assert isinstance(agent, ToolCallingTriageAgent)


def test_deterministic_triage_agent_has_expected_metadata():
    tools = Mock(spec=AnomalyOperationalTools)

    agent = DeterministicTriageAgent(tools)

    assert agent.provider_name == "deterministic"
    assert agent.model_name == "deterministic-triage-v1"


def test_deterministic_triage_agent_delegates_request(monkeypatch):
    tools = Mock(spec=AnomalyOperationalTools)
    expected_report = make_report()
    triage = Mock(return_value=expected_report)

    monkeypatch.setattr(
        TriageService,
        "triage",
        triage,
    )

    agent = DeterministicTriageAgent(tools)
    request = TriageRequest(
        run_id=42,
        max_events=5,
    )

    result = agent.triage(request)

    assert result == expected_report
    triage.assert_called_once_with(request)


def test_deterministic_triage_agent_translates_run_not_found(
    monkeypatch,
):
    tools = Mock(spec=AnomalyOperationalTools)

    original_error = TriageRunNotFoundError(
        "missing"
    )

    monkeypatch.setattr(
        TriageService,
        "triage",
        Mock(side_effect=original_error),
    )

    agent = DeterministicTriageAgent(tools)

    with pytest.raises(
        TriageAgentResourceNotFoundError
    ) as exc_info:
        agent.triage(
            TriageRequest(run_id=42)
        )

    assert exc_info.value.__cause__ is original_error


def test_deterministic_triage_agent_translates_service_error(
    monkeypatch,
):
    tools = Mock(spec=AnomalyOperationalTools)

    original_error = TriageServiceError(
        "failed"
    )

    monkeypatch.setattr(
        TriageService,
        "triage",
        Mock(side_effect=original_error),
    )

    agent = DeterministicTriageAgent(tools)

    with pytest.raises(
        TriageAgentToolError
    ) as exc_info:
        agent.triage(
            TriageRequest(run_id=42)
        )

    assert exc_info.value.__cause__ is original_error