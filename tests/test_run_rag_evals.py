import json

from types import SimpleNamespace

import pytest

import scripts.run_rag_evals as script_module
from copilot.api.settings import ApiSettings
from copilot.evals.schemas import (
    RagEvalCase,
    RagEvalReport,
    RagEvalResult,
)
from copilot.schemas.query import (
    QueryRequest,
    QueryResponse,
)
from copilot.schemas.chunk import SourceChunk
from scripts.run_rag_evals import (
    _build_snapshot,
    main,
    validate_semantic_index,
)


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
    generator = SimpleNamespace(
        provider_name="deterministic",
        model_name="deterministic-grounded-answer-v1",
    )
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
    monkeypatch.setattr(
        script_module,
        "load_chunk_manifest",
        lambda manifest_path: [],
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


def test_main_output_writes_sanitized_snapshot(
    monkeypatch,
    tmp_path,
):
    configure_script(
        monkeypatch,
        make_report(failed_cases=0),
    )
    manifest_path = tmp_path / "chunks.json"
    output_path = tmp_path / "result.json"
    manifest_path.write_text("[]\n", encoding="utf-8")

    exit_code = main(
        [
            str(manifest_path),
            str(tmp_path / "cases.json"),
            "--output",
            str(output_path),
        ]
    )
    payload = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert exit_code == 0
    assert payload["metadata"]["mode"] == (
        "deterministic-manifest"
    )
    assert payload["report"]["passed_cases"] == 2


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


def test_main_rejects_json_and_output_together(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                str(tmp_path / "chunks.json"),
                str(tmp_path / "cases.json"),
                "--json",
                "--output",
                str(tmp_path / "result.json"),
            ]
        )


def test_semantic_mode_wires_pgvector_and_disposes_engine(
    monkeypatch,
    tmp_path,
):
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url="postgresql://secret-database",
        ai_provider="openai",
        openai_api_key="secret-key",
        embedding_model="embedding-test",
        grounded_answer_model="answer-test",
    )
    embedding_provider = SimpleNamespace(
        provider_name="openai",
        model_name="embedding-test",
        dimensions=16,
    )
    generator = SimpleNamespace(
        provider_name="openai",
        model_name="answer-test",
    )
    engine = SimpleNamespace(disposed=False)
    engine.dispose = lambda: setattr(
        engine,
        "disposed",
        True,
    )
    session_factory = object()
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
        "load_rag_cases",
        lambda path: [],
    )
    monkeypatch.setattr(
        script_module,
        "load_chunk_manifest",
        lambda path: [],
    )
    monkeypatch.setattr(
        script_module,
        "create_embedding_provider",
        lambda resolved: embedding_provider,
    )
    monkeypatch.setattr(
        script_module,
        "create_grounded_answer_generator",
        lambda resolved: generator,
    )
    monkeypatch.setattr(
        script_module,
        "create_engine_from_url",
        lambda url: engine,
    )
    monkeypatch.setattr(
        script_module,
        "create_session_factory",
        lambda received_engine: session_factory,
    )

    def fake_validate(
        received_factory,
        chunks,
        provider,
    ):
        captured["preflight"] = (
            received_factory,
            chunks,
            provider,
        )

    monkeypatch.setattr(
        script_module,
        "validate_semantic_index",
        fake_validate,
    )

    def fake_run(cases, execute_query):
        execute_query(QueryRequest(query="test"))
        return make_report(failed_cases=0)

    monkeypatch.setattr(
        script_module,
        "run_rag_evals",
        fake_run,
    )

    def fake_query_service(
        resolved_settings,
        request,
        session_factory=None,
        *,
        embedding_provider=None,
        grounded_answer_generator=None,
    ):
        captured["query"] = (
            resolved_settings,
            session_factory,
            embedding_provider,
            grounded_answer_generator,
        )
        return QueryResponse(
            answer="",
            confidence=0.0,
            citations=[],
            refusal_reason="unsupported",
            context=None,
            context_snippets=[],
        )

    monkeypatch.setattr(
        script_module,
        "query_service",
        fake_query_service,
    )

    exit_code = main(
        [
            str(tmp_path / "chunks.json"),
            str(tmp_path / "cases.json"),
            "--mode",
            "semantic",
        ]
    )

    assert exit_code == 0
    assert captured["preflight"] == (
        session_factory,
        [],
        embedding_provider,
    )
    assert captured["query"] == (
        settings,
        session_factory,
        embedding_provider,
        generator,
    )
    assert engine.disposed is True


def test_semantic_mode_sanitizes_setup_failure_and_disposes(
    monkeypatch,
    tmp_path,
    capsys,
):
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url="postgresql://secret-database",
        ai_provider="openai",
        openai_api_key="secret-key",
        embedding_model="embedding-test",
        grounded_answer_model="answer-test",
    )
    provider = SimpleNamespace(
        provider_name="openai",
        model_name="embedding-test",
        dimensions=16,
    )
    generator = SimpleNamespace(
        provider_name="openai",
        model_name="answer-test",
    )
    engine = SimpleNamespace(disposed=False)
    engine.dispose = lambda: setattr(
        engine,
        "disposed",
        True,
    )

    monkeypatch.setattr(script_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(script_module, "load_api_settings", lambda: settings)
    monkeypatch.setattr(script_module, "load_rag_cases", lambda path: [])
    monkeypatch.setattr(script_module, "load_chunk_manifest", lambda path: [])
    monkeypatch.setattr(
        script_module,
        "create_embedding_provider",
        lambda resolved: provider,
    )
    monkeypatch.setattr(
        script_module,
        "create_grounded_answer_generator",
        lambda resolved: generator,
    )
    monkeypatch.setattr(
        script_module,
        "create_engine_from_url",
        lambda url: engine,
    )
    monkeypatch.setattr(
        script_module,
        "create_session_factory",
        lambda received: object(),
    )
    monkeypatch.setattr(
        script_module,
        "validate_semantic_index",
        lambda *args: (_ for _ in ()).throw(
            RuntimeError("private database detail")
        ),
    )

    exit_code = main(
        [
            str(tmp_path / "chunks.json"),
            str(tmp_path / "cases.json"),
            "--mode",
            "semantic",
        ]
    )
    error = capsys.readouterr().err

    assert exit_code == 2
    assert engine.disposed is True
    assert "RuntimeError" in error
    assert "reindex_embeddings.py" in error
    assert "private database detail" not in error
    assert "secret" not in error


def test_validate_semantic_index_checks_ids_and_configuration():
    chunk = SourceChunk(
        chunk_id="chunk-1",
        source_id="api.py",
        project_name="platform",
        source_type="python",
        source_path="api.py",
        chunk_index=0,
        content="content",
        start_line=1,
        end_line=2,
    )
    provider = SimpleNamespace(
        provider_name="openai",
        model_name="embedding-test",
        dimensions=16,
    )

    class Result:
        def __init__(self, rows):
            self.rows = rows

        def all(self):
            return self.rows

    class Session:
        def __init__(self, rows):
            self.rows = rows

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def execute(self, statement):
            return Result(self.rows)

    matching_row = SimpleNamespace(
        chunk_id="chunk-1",
        embedding_provider="openai",
        embedding_model="embedding-test",
        embedding_dimensions=16,
    )

    validate_semantic_index(
        lambda: Session([matching_row]),
        [chunk],
        provider,
    )

    with pytest.raises(
        RuntimeError,
        match="does not match",
    ):
        validate_semantic_index(
            lambda: Session(
                [
                    SimpleNamespace(
                        **{
                            **matching_row.__dict__,
                            "chunk_id": "stale",
                        }
                    )
                ]
            ),
            [chunk],
            provider,
        )


def test_snapshot_is_sanitized(tmp_path):
    manifest_path = tmp_path / "chunks.json"
    manifest_path.write_text("[]\n", encoding="utf-8")
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url="postgresql://secret-database",
        ai_provider="openai",
        openai_api_key="secret-key",
    )
    provider = SimpleNamespace(
        provider_name="openai",
        model_name="embedding-test",
        dimensions=16,
    )
    generator = SimpleNamespace(
        provider_name="openai",
        model_name="answer-test",
    )
    report = make_report(failed_cases=0)
    report.results = [
        RagEvalResult(
            case_id="safe-case-id",
            status="answered",
            expected_source_paths=["private/source.py"],
            retrieved_source_paths=["private/source.py"],
            cited_source_paths=["private/source.py"],
            schema_valid=True,
            retrieval_hit=True,
            citations_valid=True,
            citation_hit=True,
            refusal_correct=True,
            first_relevant_rank=1,
            relevant_source_recall=1.0,
            answer_terms_present=True,
            passed=True,
            failure_reasons=["private failure"],
        )
    ]

    snapshot = _build_snapshot(
        manifest_path=manifest_path,
        chunks=[],
        cases_count=20,
        top_k_values=[5],
        settings=settings,
        answer_generator=generator,
        report=report,
        duration_seconds=1.25,
        embedding_provider=provider,
    )
    serialized = json.dumps(snapshot)

    assert snapshot["metadata"]["case_count"] == 20
    assert snapshot["metadata"]["mode"] == "openai-pgvector"
    assert "secret-key" not in serialized
    assert "secret-database" not in serialized
    assert "source content" not in serialized
    assert "private/source.py" not in serialized
    assert "private failure" not in serialized
