from unittest.mock import Mock

import pytest

import scripts.run_triage_evals as script_module
from copilot.api.settings import ApiSettings
from copilot.evals.schemas import (
    TriageEvalCase,
    TriageEvalReport,
)
from copilot.schemas.triage import (
    TriageRequest,
)
from scripts.run_triage_evals import (
    format_optional_rate,
    main,
)


def make_report(
    *,
    failed_cases: int = 0,
) -> TriageEvalReport:
    total = 2
    passed = total - failed_cases

    return TriageEvalReport(
        total_cases=total,
        passed_cases=passed,
        failed_cases=failed_cases,
        schema_validity_rate=1.0,
        evidence_validity_rate=1.0,
        run_consistency_rate=1.0,
        max_events_compliance_rate=1.0,
        status_semantics_rate=1.0,
        status_accuracy=None,
        expected_findings_accuracy=None,
        finding_count_accuracy=None,
        pass_rate=passed / total,
        results=[],
    )


class FakeClient:
    def __init__(
        self,
        base_url: str,
    ):
        self.base_url = base_url

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        return None


def configure(
    monkeypatch,
    report: TriageEvalReport,
):
    settings = ApiSettings(
        anomaly_api_base_url=(
            "http://anomaly-api.test"
        ),
    )

    cases = [
        TriageEvalCase(
            case_id="latest",
            request=TriageRequest(
                max_events=5,
            ),
        )
    ]

    agent = Mock()
    agent.provider_name = "deterministic"
    agent.model_name = (
        "deterministic-triage-v1"
    )

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        script_module,
        "load_dotenv",
        lambda: None,
    )
    monkeypatch.setattr(
        script_module,
        "load_api_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        script_module,
        "AnomalyApiClient",
        FakeClient,
    )
    monkeypatch.setattr(
        script_module,
        "load_triage_cases",
        lambda fixture_path: cases,
    )

    def fake_create_triage_agent(
        received_settings,
        tools,
    ):
        captured["settings"] = (
            received_settings
        )
        captured["tools"] = tools
        return agent

    monkeypatch.setattr(
        script_module,
        "create_triage_agent",
        fake_create_triage_agent,
    )

    def fake_run_triage_evals(
        received_cases,
        executor,
    ):
        captured["cases"] = received_cases
        captured["executor"] = executor
        return report

    monkeypatch.setattr(
        script_module,
        "run_triage_evals",
        fake_run_triage_evals,
    )

    return captured, agent


def test_format_optional_rate():
    assert (
        format_optional_rate(None)
        == "N/A"
    )
    assert (
        format_optional_rate(0.5)
        == "50.00%"
    )


def test_main_requires_anomaly_api_base_url(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        script_module,
        "load_dotenv",
        lambda: None,
    )
    monkeypatch.setattr(
        script_module,
        "load_api_settings",
        lambda: ApiSettings(
            anomaly_api_base_url=None,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "ANOMALYOPS_ANOMALY_API_BASE_URL "
            "is required"
        ),
    ):
        main(
            [
                str(
                    tmp_path
                    / "triage_cases.json"
                )
            ]
        )


def test_main_returns_zero_by_default_when_eval_fails(
    monkeypatch,
    tmp_path,
):
    configure(
        monkeypatch,
        make_report(
            failed_cases=1,
        ),
    )

    exit_code = main(
        [
            str(
                tmp_path
                / "triage_cases.json"
            )
        ]
    )

    assert exit_code == 0


def test_main_strict_returns_one_on_failure(
    monkeypatch,
    tmp_path,
):
    configure(
        monkeypatch,
        make_report(
            failed_cases=1,
        ),
    )

    exit_code = main(
        [
            str(
                tmp_path
                / "triage_cases.json"
            ),
            "--strict",
        ]
    )

    assert exit_code == 1


def test_main_strict_returns_zero_when_all_pass(
    monkeypatch,
    tmp_path,
):
    configure(
        monkeypatch,
        make_report(),
    )

    exit_code = main(
        [
            str(
                tmp_path
                / "triage_cases.json"
            ),
            "--strict",
        ]
    )

    assert exit_code == 0


def test_main_prints_provider_and_metrics(
    monkeypatch,
    tmp_path,
    capsys,
):
    configure(
        monkeypatch,
        make_report(),
    )

    exit_code = main(
        [
            str(
                tmp_path
                / "triage_cases.json"
            )
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Triage evaluation" in output
    assert "Provider: deterministic" in output
    assert (
        "Model: deterministic-triage-v1"
        in output
    )
    assert "Total: 2" in output
    assert "Passed: 2" in output
    assert "Failed: 0" in output
    assert "Schema validity: 100.00%" in output
    assert "Evidence validity: 100.00%" in output
    assert "Run consistency: 100.00%" in output
    assert (
        "Max-events compliance: 100.00%"
        in output
    )
    assert "Status semantics: 100.00%" in output
    assert "Status accuracy: N/A" in output
    assert "Pass rate: 100.00%" in output


def test_main_json_outputs_report(
    monkeypatch,
    tmp_path,
    capsys,
):
    configure(
        monkeypatch,
        make_report(),
    )

    exit_code = main(
        [
            str(
                tmp_path
                / "triage_cases.json"
            ),
            "--json",
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"total_cases": 2' in output
    assert '"passed_cases": 2' in output
    assert '"failed_cases": 0' in output


def test_main_passes_agent_triage_to_runner(
    monkeypatch,
    tmp_path,
):
    captured, agent = configure(
        monkeypatch,
        make_report(),
    )

    main(
        [
            str(
                tmp_path
                / "triage_cases.json"
            )
        ]
    )

    executor = captured["executor"]

    assert executor == agent.triage