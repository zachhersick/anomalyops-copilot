import json

import pytest

import scripts.run_rag_evals as script_module
from copilot.api.settings import ApiSettings
from copilot.evals.schemas import (
    RagEvalCase,
    RagEvalReport,
)
from copilot.schemas.query import (
    QueryRequest,
    QueryResponse,
)
from scripts.run_rag_evals import main


def make_report(
    *,
    failed_cases: int,
) -> RagEvalReport:
    total_cases = 2
    passed_cases = (
        total_cases - failed_cases
    )

    return RagEvalReport(
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        supported_cases=1,
        refusal_cases=1,
        schema_validity_rate=1.0,
        retrieval_hit_rate=0.5,
        citation_validity_rate=1.0,
        citation_hit_rate=0.5,
        refusal_accuracy=1.0,
        pass_rate=passed_cases / total_cases,
        results=[],
    )


def configure_script(
    monkeypatch,
    report: RagEvalReport,
):
    settings = ApiSettings(
        ai_provider="openai",
        openai_api_key="test-key",
        grounded_answer_model="gpt-test",
    )
    generator = object()
    cases = [
        RagEvalCase(
            case_id="test-case",
            query="test query",
            expected_source_paths=[
                "source_code/api.py",
            ],
        )
    ]
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

    def fake_create_generator(
        resolved_settings,
    ):
        captured["generator_settings"] = (
            resolved_settings
        )
        return generator

    monkeypatch.setattr(
        script_module,
        "create_grounded_answer_generator",
        fake_create_generator,
    )
    monkeypatch.setattr(
        script_module,
        "load_rag_cases",
        lambda fixture_path: cases,
    )

    def fake_run_rag_evals(
        received_cases,
        execute_query,
    ):
        captured["cases"] = received_cases
        captured["execute_query"] = execute_query
        return report

    monkeypatch.setattr(
        script_module,
        "run_rag_evals",
        fake_run_rag_evals,
    )

    return captured, generator, cases


def test_main_prints_readable_report_and_returns_zero_by_default(
    monkeypatch,
    tmp_path,
    capsys,
):
    report = make_report(
        failed_cases=1,
    )
    captured, _, cases = configure_script(
        monkeypatch,
        report,
    )

    exit_code = main(
        [
            str(tmp_path / "chunks.json"),
            str(tmp_path / "cases.json"),
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "RAG evaluation" in output
    assert "Total: 2" in output
    assert "Passed: 1" in output
    assert "Failed: 1" in output
    assert "Schema validity: 100.00%" in output
    assert "Retrieval hit rate: 50.00%" in output
    assert "Citation validity: 100.00%" in output
    assert "Citation hit rate: 50.00%" in output
    assert "Refusal accuracy: 100.00%" in output
    assert "Pass rate: 50.00%" in output
    assert captured["cases"] == cases
    assert captured["generator_settings"].ai_provider == "deterministic"


def test_main_strict_returns_one_when_cases_fail(
    monkeypatch,
    tmp_path,
):
    configure_script(
        monkeypatch,
        make_report(
            failed_cases=1,
        ),
    )

    exit_code = main(
        [
            str(tmp_path / "chunks.json"),
            str(tmp_path / "cases.json"),
            "--strict",
        ]
    )

    assert exit_code == 1


def test_main_strict_returns_zero_when_all_cases_pass(
    monkeypatch,
    tmp_path,
):
    configure_script(
        monkeypatch,
        make_report(
            failed_cases=0,
        ),
    )

    exit_code = main(
        [
            str(tmp_path / "chunks.json"),
            str(tmp_path / "cases.json"),
            "--strict",
        ]
    )

    assert exit_code == 0


def test_main_json_outputs_valid_report(
    monkeypatch,
    tmp_path,
    capsys,
):
    configure_script(
        monkeypatch,
        make_report(
            failed_cases=1,
        ),
    )

    exit_code = main(
        [
            str(tmp_path / "chunks.json"),
            str(tmp_path / "cases.json"),
            "--json",
        ]
    )

    payload = json.loads(
        capsys.readouterr().out
    )

    assert exit_code == 0
    assert payload["total_cases"] == 2
    assert payload["passed_cases"] == 1
    assert payload["failed_cases"] == 1
    assert payload["pass_rate"] == pytest.approx(
        0.5
    )


def test_main_forces_manifest_retrieval_settings(
    monkeypatch,
    tmp_path,
):
    report = make_report(
        failed_cases=0,
    )
    captured, _, _ = configure_script(
        monkeypatch,
        report,
    )

    manifest_path = tmp_path / "chunks.json"

    exit_code = main(
        [
            str(manifest_path),
            str(tmp_path / "cases.json"),
        ]
    )

    resolved_settings = captured[
        "generator_settings"
    ]

    assert exit_code == 0
    assert resolved_settings.retrieval_backend == (
        "manifest"
    )
    assert resolved_settings.manifest_path == (
        manifest_path
    )


def test_main_query_executor_uses_configured_generator(
    monkeypatch,
    tmp_path,
):
    report = make_report(
        failed_cases=0,
    )
    captured, generator, _ = configure_script(
        monkeypatch,
        report,
    )

    expected_response = QueryResponse(
        answer="Supported answer [1]",
        confidence=0.9,
        citations=[],
        refusal_reason=None,
        context=None,
        context_snippets=[],
    )
    query_calls: list[dict[str, object]] = []

    def fake_query_service(
        settings,
        request,
        *,
        grounded_answer_generator,
    ):
        query_calls.append(
            {
                "settings": settings,
                "request": request,
                "generator": (
                    grounded_answer_generator
                ),
            }
        )
        return expected_response

    monkeypatch.setattr(
        script_module,
        "query_service",
        fake_query_service,
    )

    manifest_path = tmp_path / "chunks.json"

    exit_code = main(
        [
            str(manifest_path),
            str(tmp_path / "cases.json"),
        ]
    )

    execute_query = captured[
        "execute_query"
    ]
    request = QueryRequest(
        query="prediction API",
        show_context=True,
    )

    result = execute_query(request)

    assert exit_code == 0
    assert result is expected_response
    assert len(query_calls) == 1
    assert query_calls[0]["request"] == request
    assert query_calls[0]["generator"] is generator
    assert (
        query_calls[0]["settings"].manifest_path
        == manifest_path
    )
