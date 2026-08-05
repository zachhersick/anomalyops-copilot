import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import OpenAIError

from copilot.providers.errors import (
    InvalidTriageAgentResponseError,
    TriageAgentConfigurationError,
    TriageAgentProviderError,
    TriageAgentResourceNotFoundError,
    TriageAgentToolError,
)
from copilot.providers.interfaces import (
    ToolCallingTriageAgent,
)
from copilot.providers.openai_triage import (
    OpenAITriageAgent,
)
from copilot.schemas.anomaly import (
    AlertEvent,
    RowAlert,
    RunSummary,
)
from copilot.schemas.anomaly_tools import (
    GetEventAlertsOutput,
    GetLatestRunOutput,
    GetRunSummaryOutput,
    ListAlertEventsOutput,
)
from copilot.schemas.triage import (
    TriageFindingDraft,
    TriageReportDraft,
    TriageRequest,
)
from copilot.schemas.anomaly import LatestRun
from copilot.tools.anomaly import (
    AnomalyOperationalTools,
    OperationalResourceNotFoundError,
    OperationalToolError,
)


def make_summary(
    *,
    run_id: int = 42,
    total_alert_events: int = 1,
) -> RunSummary:
    return RunSummary(
        run_id=run_id,
        total_predictions=100,
        total_anomalies_predicted=10,
        total_row_alerts=5,
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
            0.97 if total_alert_events else None
        ),
        mean_anomaly_score=(
            0.80 if total_alert_events else None
        ),
    )


def make_event(
    *,
    run_id: int = 42,
    event_id: int = 7,
    machine_id: int = 3,
    sensor: str = "temperature",
    anomaly_type: str | None = "spike",
    severity: str | None = "critical",
) -> AlertEvent:
    return AlertEvent(
        run_id=run_id,
        event_id=event_id,
        machine_id=machine_id,
        sensor=sensor,
        anomaly_type=anomaly_type,
        start_step=100,
        end_step=110,
        duration=11,
        alert_count=3,
        max_severity=severity,
        max_severity_reason="High anomaly score",
        max_anomaly_score=0.97,
        mean_anomaly_score=0.90,
        min_sensor_value=70.0,
        max_sensor_value=105.0,
        first_reason="Outside range",
        status="open",
        real_value=1,
    )


def make_alert(
    *,
    run_id: int = 42,
    alert_id: int = 10,
    step: int = 105,
    machine_id: int = 3,
    sensor: str = "temperature",
) -> RowAlert:
    return RowAlert(
        run_id=run_id,
        alert_id=alert_id,
        step=step,
        machine_id=machine_id,
        sensor=sensor,
        sensor_value=103.0,
        prediction=1,
        anomaly_score=0.97,
        severity="critical",
        alert_type="anomaly",
        reason="Outside range",
        status="open",
        anomaly_type="spike",
        real_value=1,
    )


def function_call(
    name: str,
    arguments: dict[str, object] | str,
    call_id: str = "call-1",
):
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)

    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=arguments,
        call_id=call_id,
    )


def tool_response(*calls):
    return SimpleNamespace(
        output=list(calls),
        output_parsed=None,
    )


def final_response(
    draft: TriageReportDraft,
):
    return SimpleNamespace(
        output=[],
        output_parsed=draft,
    )


def make_client(*responses):
    client = Mock()
    client.responses.parse.side_effect = list(
        responses
    )
    return client


def make_tools() -> Mock:
    return Mock(spec=AnomalyOperationalTools)


def make_completed_draft(
    **overrides,
) -> TriageReportDraft:
    values = {
        "status": "completed",
        "findings": [
            TriageFindingDraft(
                severity="critical",
                machine_id=3,
                sensor="temperature",
                anomaly_type="spike",
                summary=(
                    "Critical temperature spike."
                ),
                evidence_ids=["event-7"],
            )
        ],
        "refusal_reason": None,
    }
    values.update(overrides)

    return TriageReportDraft(**values)


def test_openai_triage_agent_satisfies_protocol():
    agent = OpenAITriageAgent(
        model_name="gpt-test",
        client=Mock(),
        tools=make_tools(),
    )

    assert isinstance(
        agent,
        ToolCallingTriageAgent,
    )


def test_openai_triage_agent_metadata():
    agent = OpenAITriageAgent(
        model_name="gpt-test",
        client=Mock(),
        tools=make_tools(),
    )

    assert agent.provider_name == "openai"
    assert agent.model_name == "gpt-test"


@pytest.mark.parametrize(
    "model_name",
    [
        "",
        "   ",
    ],
)
def test_openai_triage_agent_rejects_blank_model(
    model_name,
):
    with pytest.raises(
        TriageAgentConfigurationError
    ):
        OpenAITriageAgent(
            model_name=model_name,
            client=Mock(),
            tools=make_tools(),
        )


@pytest.mark.parametrize(
    "max_tool_rounds",
    [
        0,
        -1,
    ],
)
def test_openai_triage_agent_requires_positive_round_limit(
    max_tool_rounds,
):
    with pytest.raises(
        TriageAgentConfigurationError
    ):
        OpenAITriageAgent(
            model_name="gpt-test",
            client=Mock(),
            tools=make_tools(),
            max_tool_rounds=max_tool_rounds,
        )


def test_request_uses_expected_model_tools_and_format():
    draft = TriageReportDraft(
        status="refused",
        findings=[],
        refusal_reason="Cannot inspect data.",
    )

    client = make_client(
        final_response(draft)
    )

    agent = OpenAITriageAgent(
        model_name="gpt-test",
        client=client,
        tools=make_tools(),
    )

    agent.triage(
        TriageRequest(
            run_id=42,
            max_events=5,
        )
    )

    kwargs = (
        client.responses.parse.call_args.kwargs
    )

    assert kwargs["model"] == "gpt-test"
    assert kwargs["text_format"] is TriageReportDraft
    assert kwargs["parallel_tool_calls"] is False

    tools = kwargs["tools"]

    assert [
        tool["name"]
        for tool in tools
    ] == [
        "get_latest_run",
        "get_run_summary",
        "list_alert_events",
        "get_event_alerts",
    ]

    assert all(
        tool["type"] == "function"
        for tool in tools
    )
    assert all(
        tool["strict"] is True
        for tool in tools
    )


def test_tool_schemas_disallow_extra_properties():
    client = make_client(
        final_response(
            TriageReportDraft(
                status="refused",
                findings=[],
                refusal_reason="Cannot inspect.",
            )
        )
    )

    agent = OpenAITriageAgent(
        model_name="gpt-test",
        client=client,
        tools=make_tools(),
    )

    agent.triage(
        TriageRequest(run_id=42)
    )

    tools = (
        client.responses.parse.call_args.kwargs[
            "tools"
        ]
    )

    for tool in tools:
        assert (
            tool["parameters"][
                "additionalProperties"
            ]
            is False
        )


def test_prompt_contains_request_configuration():
    client = make_client(
        final_response(
            TriageReportDraft(
                status="refused",
                findings=[],
                refusal_reason="No evidence.",
            )
        )
    )

    agent = OpenAITriageAgent(
        model_name="gpt-test",
        client=client,
        tools=make_tools(),
    )

    agent.triage(
        TriageRequest(
            run_id=42,
            max_events=9,
        )
    )

    input_items = (
        client.responses.parse.call_args.kwargs[
            "input"
        ]
    )

    assert "Requested run ID: 42" in (
        input_items[0]["content"]
    )
    assert "Maximum findings/events: 9" in (
        input_items[0]["content"]
    )


def test_valid_explicit_run_completed_flow():
    tools = make_tools()

    tools.get_run_summary.return_value = (
        GetRunSummaryOutput(
            summary=make_summary()
        )
    )
    tools.list_alert_events.return_value = (
        ListAlertEventsOutput(
            events=[make_event()]
        )
    )
    tools.get_event_alerts.return_value = (
        GetEventAlertsOutput(
            alerts=[
                make_alert(
                    alert_id=2,
                    step=105,
                ),
                make_alert(
                    alert_id=1,
                    step=101,
                ),
            ]
        )
    )

    client = make_client(
        tool_response(
            function_call(
                "get_run_summary",
                {"run_id": 42},
                "summary-call",
            )
        ),
        tool_response(
            function_call(
                "list_alert_events",
                {
                    "run_id": 42,
                    "severity": "critical",
                    "sensor": None,
                    "anomaly_type": None,
                    "limit": 5,
                    "offset": 0,
                },
                "events-call",
            )
        ),
        tool_response(
            function_call(
                "get_event_alerts",
                {
                    "run_id": 42,
                    "event_id": 7,
                },
                "alerts-call",
            )
        ),
        final_response(
            make_completed_draft()
        ),
    )

    agent = OpenAITriageAgent(
        model_name="gpt-test",
        client=client,
        tools=tools,
    )

    report = agent.triage(
        TriageRequest(
            run_id=42,
            max_events=5,
        )
    )

    assert report.status == "completed"
    assert report.run_id == 42
    assert report.run_summary == make_summary()

    assert len(report.findings) == 1
    assert (
        report.findings[0].finding_id
        == "finding-1"
    )

    assert len(report.evidence) == 1
    assert (
        report.evidence[0].evidence_id
        == "event-7"
    )

    assert [
        alert.alert_id
        for alert in report.evidence[0].alerts
    ] == [1, 2]


def test_latest_run_is_resolved_before_summary():
    tools = make_tools()

    tools.get_latest_run.return_value = (
        GetLatestRunOutput(
            run=LatestRun(run_id=42)
        )
    )
    tools.get_run_summary.return_value = (
        GetRunSummaryOutput(
            summary=make_summary(
                total_alert_events=0
            )
        )
    )

    client = make_client(
        tool_response(
            function_call(
                "get_latest_run",
                {},
                "latest-call",
            )
        ),
        tool_response(
            function_call(
                "get_run_summary",
                {"run_id": 42},
                "summary-call",
            )
        ),
        final_response(
            TriageReportDraft(
                status="no_alerts",
                findings=[],
                refusal_reason=None,
            )
        ),
    )

    agent = OpenAITriageAgent(
        model_name="gpt-test",
        client=client,
        tools=tools,
    )

    report = agent.triage(
        TriageRequest(
            run_id=None,
        )
    )

    assert report.run_id == 42
    assert report.status == "no_alerts"

    tools.get_latest_run.assert_called_once()
    tools.get_run_summary.assert_called_once()


def test_function_outputs_preserve_call_id():
    tools = make_tools()

    tools.get_run_summary.return_value = (
        GetRunSummaryOutput(
            summary=make_summary(
                total_alert_events=0
            )
        )
    )

    client = make_client(
        tool_response(
            function_call(
                "get_run_summary",
                {"run_id": 42},
                "abc-123",
            )
        ),
        final_response(
            TriageReportDraft(
                status="no_alerts",
                findings=[],
                refusal_reason=None,
            )
        ),
    )

    agent = OpenAITriageAgent(
        model_name="gpt-test",
        client=client,
        tools=tools,
    )

    agent.triage(
        TriageRequest(run_id=42)
    )

    second_input = (
        client.responses.parse.call_args_list[
            1
        ].kwargs["input"]
    )

    outputs = [
        item
        for item in second_input
        if (
            isinstance(item, dict)
            and item.get("type")
            == "function_call_output"
        )
    ]

    assert len(outputs) == 1
    assert outputs[0]["call_id"] == "abc-123"


def test_valid_no_alerts_report_has_no_evidence():
    tools = make_tools()

    tools.get_run_summary.return_value = (
        GetRunSummaryOutput(
            summary=make_summary(
                total_alert_events=0
            )
        )
    )

    client = make_client(
        tool_response(
            function_call(
                "get_run_summary",
                {"run_id": 42},
            )
        ),
        final_response(
            TriageReportDraft(
                status="no_alerts",
                findings=[],
                refusal_reason=None,
            )
        ),
    )

    report = OpenAITriageAgent(
        "gpt-test",
        client,
        tools,
    ).triage(
        TriageRequest(run_id=42)
    )

    assert report.status == "no_alerts"
    assert report.findings == []
    assert report.evidence == []
    assert report.refusal_reason is None


def test_valid_refused_report():
    draft = TriageReportDraft(
        status="refused",
        findings=[],
        refusal_reason=(
            "Operational evidence unavailable."
        ),
    )

    report = OpenAITriageAgent(
        "gpt-test",
        make_client(
            final_response(draft)
        ),
        make_tools(),
    ).triage(
        TriageRequest(run_id=42)
    )

    assert report.status == "refused"
    assert report.findings == []
    assert report.evidence == []
    assert report.refusal_reason == (
        "Operational evidence unavailable."
    )


def test_unknown_tool_is_rejected():
    client = make_client(
        tool_response(
            function_call(
                "delete_everything",
                {},
            )
        )
    )

    agent = OpenAITriageAgent(
        "gpt-test",
        client,
        make_tools(),
    )

    with pytest.raises(
        InvalidTriageAgentResponseError
    ):
        agent.triage(
            TriageRequest(run_id=42)
        )


def test_malformed_tool_arguments_are_rejected():
    client = make_client(
        tool_response(
            function_call(
                "get_run_summary",
                "{not-json",
            )
        )
    )

    agent = OpenAITriageAgent(
        "gpt-test",
        client,
        make_tools(),
    )

    with pytest.raises(
        InvalidTriageAgentResponseError
    ):
        agent.triage(
            TriageRequest(run_id=42)
        )


def test_invalid_tool_arguments_are_rejected():
    client = make_client(
        tool_response(
            function_call(
                "get_run_summary",
                {"run_id": 0},
            )
        )
    )

    with pytest.raises(
        InvalidTriageAgentResponseError
    ):
        OpenAITriageAgent(
            "gpt-test",
            client,
            make_tools(),
        ).triage(
            TriageRequest(run_id=42)
        )


def test_explicit_run_cannot_use_latest_lookup():
    client = make_client(
        tool_response(
            function_call(
                "get_latest_run",
                {},
            )
        )
    )

    with pytest.raises(
        InvalidTriageAgentResponseError
    ):
        OpenAITriageAgent(
            "gpt-test",
            client,
            make_tools(),
        ).triage(
            TriageRequest(run_id=42)
        )


def test_run_specific_tool_cannot_precede_latest_resolution():
    client = make_client(
        tool_response(
            function_call(
                "get_run_summary",
                {"run_id": 42},
            )
        )
    )

    with pytest.raises(
        InvalidTriageAgentResponseError
    ):
        OpenAITriageAgent(
            "gpt-test",
            client,
            make_tools(),
        ).triage(
            TriageRequest(run_id=None)
        )


def test_tool_cannot_target_different_run():
    client = make_client(
        tool_response(
            function_call(
                "get_run_summary",
                {"run_id": 99},
            )
        )
    )

    with pytest.raises(
        InvalidTriageAgentResponseError
    ):
        OpenAITriageAgent(
            "gpt-test",
            client,
            make_tools(),
        ).triage(
            TriageRequest(run_id=42)
        )


def test_list_event_limit_cannot_exceed_request_limit():
    client = make_client(
        tool_response(
            function_call(
                "list_alert_events",
                {
                    "run_id": 42,
                    "severity": None,
                    "sensor": None,
                    "anomaly_type": None,
                    "limit": 6,
                    "offset": 0,
                },
            )
        )
    )

    with pytest.raises(
        InvalidTriageAgentResponseError
    ):
        OpenAITriageAgent(
            "gpt-test",
            client,
            make_tools(),
        ).triage(
            TriageRequest(
                run_id=42,
                max_events=5,
            )
        )


def test_event_alerts_require_previously_listed_event():
    client = make_client(
        tool_response(
            function_call(
                "get_event_alerts",
                {
                    "run_id": 42,
                    "event_id": 7,
                },
            )
        )
    )

    with pytest.raises(
        InvalidTriageAgentResponseError
    ):
        OpenAITriageAgent(
            "gpt-test",
            client,
            make_tools(),
        ).triage(
            TriageRequest(run_id=42)
        )


def test_event_from_wrong_run_is_rejected():
    tools = make_tools()

    tools.list_alert_events.return_value = (
        ListAlertEventsOutput(
            events=[
                make_event(
                    run_id=99
                )
            ]
        )
    )

    client = make_client(
        tool_response(
            function_call(
                "list_alert_events",
                {
                    "run_id": 42,
                    "severity": None,
                    "sensor": None,
                    "anomaly_type": None,
                    "limit": 5,
                    "offset": 0,
                },
            )
        )
    )

    with pytest.raises(
        InvalidTriageAgentResponseError
    ):
        OpenAITriageAgent(
            "gpt-test",
            client,
            tools,
        ).triage(
            TriageRequest(run_id=42)
        )


def test_model_cannot_reference_uninspected_evidence():
    tools = make_tools()

    tools.get_run_summary.return_value = (
        GetRunSummaryOutput(
            summary=make_summary()
        )
    )
    tools.list_alert_events.return_value = (
        ListAlertEventsOutput(
            events=[make_event()]
        )
    )

    client = make_client(
        tool_response(
            function_call(
                "get_run_summary",
                {"run_id": 42},
                "summary",
            )
        ),
        tool_response(
            function_call(
                "list_alert_events",
                {
                    "run_id": 42,
                    "severity": None,
                    "sensor": None,
                    "anomaly_type": None,
                    "limit": 5,
                    "offset": 0,
                },
                "events",
            )
        ),
        final_response(
            make_completed_draft()
        ),
    )

    with pytest.raises(
        InvalidTriageAgentResponseError
    ):
        OpenAITriageAgent(
            "gpt-test",
            client,
            tools,
        ).triage(
            TriageRequest(run_id=42)
        )


def test_finding_machine_must_match_evidence():
    tools = make_tools()

    tools.get_run_summary.return_value = (
        GetRunSummaryOutput(
            summary=make_summary()
        )
    )
    tools.list_alert_events.return_value = (
        ListAlertEventsOutput(
            events=[make_event()]
        )
    )
    tools.get_event_alerts.return_value = (
        GetEventAlertsOutput(
            alerts=[make_alert()]
        )
    )

    bad_draft = make_completed_draft(
        findings=[
            TriageFindingDraft(
                severity="critical",
                machine_id=999,
                sensor="temperature",
                anomaly_type="spike",
                summary="Bad machine.",
                evidence_ids=["event-7"],
            )
        ]
    )

    client = make_client(
        tool_response(
            function_call(
                "get_run_summary",
                {"run_id": 42},
            )
        ),
        tool_response(
            function_call(
                "list_alert_events",
                {
                    "run_id": 42,
                    "severity": None,
                    "sensor": None,
                    "anomaly_type": None,
                    "limit": 5,
                    "offset": 0,
                },
            )
        ),
        tool_response(
            function_call(
                "get_event_alerts",
                {
                    "run_id": 42,
                    "event_id": 7,
                },
            )
        ),
        final_response(bad_draft),
    )

    with pytest.raises(
        InvalidTriageAgentResponseError
    ):
        OpenAITriageAgent(
            "gpt-test",
            client,
            tools,
        ).triage(
            TriageRequest(run_id=42)
        )


def test_finding_cannot_invent_missing_anomaly_type():
    tools = make_tools()
    tools.get_run_summary.return_value = GetRunSummaryOutput(
        summary=make_summary()
    )
    tools.list_alert_events.return_value = ListAlertEventsOutput(
        events=[make_event(anomaly_type=None)]
    )
    tools.get_event_alerts.return_value = GetEventAlertsOutput(
        alerts=[make_alert()]
    )

    client = make_client(
        tool_response(
            function_call("get_run_summary", {"run_id": 42})
        ),
        tool_response(
            function_call(
                "list_alert_events",
                {
                    "run_id": 42,
                    "severity": None,
                    "sensor": None,
                    "anomaly_type": None,
                    "limit": 5,
                    "offset": 0,
                },
            )
        ),
        tool_response(
            function_call(
                "get_event_alerts",
                {"run_id": 42, "event_id": 7},
            )
        ),
        final_response(make_completed_draft()),
    )

    with pytest.raises(InvalidTriageAgentResponseError):
        OpenAITriageAgent(
            "gpt-test",
            client,
            tools,
        ).triage(TriageRequest(run_id=42))


def test_unique_evidence_events_cannot_exceed_request_limit():
    tools = make_tools()
    tools.get_run_summary.return_value = GetRunSummaryOutput(
        summary=make_summary(total_alert_events=2)
    )
    tools.list_alert_events.side_effect = [
        ListAlertEventsOutput(events=[make_event(event_id=7)]),
        ListAlertEventsOutput(events=[make_event(event_id=8)]),
    ]
    tools.get_event_alerts.side_effect = [
        GetEventAlertsOutput(alerts=[make_alert(alert_id=10)]),
        GetEventAlertsOutput(alerts=[make_alert(alert_id=11)]),
    ]
    draft = make_completed_draft(
        findings=[
            TriageFindingDraft(
                severity="critical",
                machine_id=3,
                sensor="temperature",
                anomaly_type="spike",
                summary="Two events.",
                evidence_ids=["event-7", "event-8"],
            )
        ]
    )
    list_arguments = {
        "run_id": 42,
        "severity": None,
        "sensor": None,
        "anomaly_type": None,
        "limit": 1,
        "offset": 0,
    }
    client = make_client(
        tool_response(
            function_call("get_run_summary", {"run_id": 42})
        ),
        tool_response(
            function_call(
                "list_alert_events",
                list_arguments,
                "list-1",
            ),
            function_call(
                "list_alert_events",
                list_arguments,
                "list-2",
            ),
        ),
        tool_response(
            function_call(
                "get_event_alerts",
                {"run_id": 42, "event_id": 7},
                "alerts-1",
            ),
            function_call(
                "get_event_alerts",
                {"run_id": 42, "event_id": 8},
                "alerts-2",
            ),
        ),
        final_response(draft),
    )

    with pytest.raises(InvalidTriageAgentResponseError):
        OpenAITriageAgent(
            "gpt-test",
            client,
            tools,
        ).triage(
            TriageRequest(run_id=42, max_events=1)
        )


def test_completed_requires_run_summary():
    with pytest.raises(
        InvalidTriageAgentResponseError
    ):
        OpenAITriageAgent(
            "gpt-test",
            make_client(
                final_response(
                    make_completed_draft()
                )
            ),
            make_tools(),
        ).triage(
            TriageRequest(run_id=42)
        )


def test_completed_requires_findings():
    tools = make_tools()

    tools.get_run_summary.return_value = (
        GetRunSummaryOutput(
            summary=make_summary()
        )
    )

    client = make_client(
        tool_response(
            function_call(
                "get_run_summary",
                {"run_id": 42},
            )
        ),
        final_response(
            TriageReportDraft(
                status="completed",
                findings=[],
                refusal_reason=None,
            )
        ),
    )

    with pytest.raises(
        InvalidTriageAgentResponseError
    ):
        OpenAITriageAgent(
            "gpt-test",
            client,
            tools,
        ).triage(
            TriageRequest(run_id=42)
        )


def test_no_alerts_requires_zero_summary_events():
    tools = make_tools()

    tools.get_run_summary.return_value = (
        GetRunSummaryOutput(
            summary=make_summary(
                total_alert_events=1
            )
        )
    )

    client = make_client(
        tool_response(
            function_call(
                "get_run_summary",
                {"run_id": 42},
            )
        ),
        final_response(
            TriageReportDraft(
                status="no_alerts",
                findings=[],
                refusal_reason=None,
            )
        ),
    )

    with pytest.raises(
        InvalidTriageAgentResponseError
    ):
        OpenAITriageAgent(
            "gpt-test",
            client,
            tools,
        ).triage(
            TriageRequest(run_id=42)
        )


def test_refused_requires_reason():
    client = make_client(
        final_response(
            TriageReportDraft(
                status="refused",
                findings=[],
                refusal_reason=None,
            )
        )
    )

    with pytest.raises(
        InvalidTriageAgentResponseError
    ):
        OpenAITriageAgent(
            "gpt-test",
            client,
            make_tools(),
        ).triage(
            TriageRequest(run_id=42)
        )


def test_response_cannot_mix_tool_calls_and_final_output():
    response = SimpleNamespace(
        output=[
            function_call(
                "get_run_summary",
                {"run_id": 42},
            )
        ],
        output_parsed=(
            TriageReportDraft(
                status="refused",
                findings=[],
                refusal_reason="No.",
            )
        ),
    )

    with pytest.raises(
        InvalidTriageAgentResponseError
    ):
        OpenAITriageAgent(
            "gpt-test",
            make_client(response),
            make_tools(),
        ).triage(
            TriageRequest(run_id=42)
        )


def test_missing_final_parsed_output_is_rejected():
    response = SimpleNamespace(
        output=[],
        output_parsed=None,
    )

    with pytest.raises(
        InvalidTriageAgentResponseError
    ):
        OpenAITriageAgent(
            "gpt-test",
            make_client(response),
            make_tools(),
        ).triage(
            TriageRequest(run_id=42)
        )


def test_tool_round_limit_is_enforced():
    tools = make_tools()

    tools.get_run_summary.return_value = (
        GetRunSummaryOutput(
            summary=make_summary(
                total_alert_events=0
            )
        )
    )

    repeated_call = tool_response(
        function_call(
            "get_run_summary",
            {"run_id": 42},
        )
    )

    client = make_client(
        repeated_call,
        repeated_call,
    )

    agent = OpenAITriageAgent(
        "gpt-test",
        client,
        tools,
        max_tool_rounds=1,
    )

    with pytest.raises(
        InvalidTriageAgentResponseError
    ):
        agent.triage(
            TriageRequest(run_id=42)
        )


def test_operational_not_found_error_is_translated():
    tools = make_tools()

    original = (
        OperationalResourceNotFoundError(
            "missing"
        )
    )

    tools.get_run_summary.side_effect = original

    client = make_client(
        tool_response(
            function_call(
                "get_run_summary",
                {"run_id": 42},
            )
        )
    )

    with pytest.raises(
        TriageAgentResourceNotFoundError
    ) as exc_info:
        OpenAITriageAgent(
            "gpt-test",
            client,
            tools,
        ).triage(
            TriageRequest(run_id=42)
        )

    assert exc_info.value.__cause__ is original


def test_operational_tool_error_is_translated():
    tools = make_tools()

    original = OperationalToolError(
        "failed"
    )

    tools.get_run_summary.side_effect = original

    client = make_client(
        tool_response(
            function_call(
                "get_run_summary",
                {"run_id": 42},
            )
        )
    )

    with pytest.raises(
        TriageAgentToolError
    ) as exc_info:
        OpenAITriageAgent(
            "gpt-test",
            client,
            tools,
        ).triage(
            TriageRequest(run_id=42)
        )

    assert exc_info.value.__cause__ is original


def test_openai_error_is_translated():
    original = OpenAIError(
        "provider failed"
    )

    client = Mock()
    client.responses.parse.side_effect = (
        original
    )

    with pytest.raises(
        TriageAgentProviderError
    ) as exc_info:
        OpenAITriageAgent(
            "gpt-test",
            client,
            make_tools(),
        ).triage(
            TriageRequest(run_id=42)
        )

    assert exc_info.value.__cause__ is original
