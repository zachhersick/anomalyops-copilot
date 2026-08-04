import json

import pytest
from pydantic import ValidationError

from copilot.evals.runner import (
    evaluate_rag_response,
    load_rag_cases,
    run_rag_evals,
)
from copilot.evals.schemas import RagEvalCase
from copilot.schemas.answer import Citation
from copilot.schemas.query import (
    ContextSnippet,
    QueryRequest,
    QueryResponse,
)


def make_case(
    *,
    case_id: str = "prediction-api",
    expected_source_paths: list[str] | None = None,
    expect_refusal: bool = False,
) -> RagEvalCase:
    return RagEvalCase(
        case_id=case_id,
        query="How does the prediction API work?",
        expected_source_paths=(
            expected_source_paths
            if expected_source_paths is not None
            else ["source_code/api.py"]
        ),
        expect_refusal=expect_refusal,
        top_k=3,
        min_score=0.0,
    )


def make_answered_response(
    *,
    source_path: str = "source_code/api.py",
    citation_source_path: str | None = None,
    answer: str = "The API accepts engineered features [1].",
    citation_id: int = 1,
    context_citation_id: int = 1,
    citation_start_line: int = 10,
    citation_end_line: int = 20,
    context_start_line: int = 10,
    context_end_line: int = 20,
) -> QueryResponse:
    return QueryResponse(
        answer=answer,
        confidence=0.9,
        citations=[
            Citation(
                citation_id=citation_id,
                source_path=(
                    citation_source_path
                    if citation_source_path is not None
                    else source_path
                ),
                start_line=citation_start_line,
                end_line=citation_end_line,
            )
        ],
        refusal_reason=None,
        context="retrieved context",
        context_snippets=[
            ContextSnippet(
                citation_id=context_citation_id,
                source_path=source_path,
                start_line=context_start_line,
                end_line=context_end_line,
                content="The API accepts engineered features.",
                score=0.9,
            )
        ],
    )


def make_refusal_response() -> QueryResponse:
    return QueryResponse(
        answer="",
        confidence=0.0,
        citations=[],
        refusal_reason="The retrieved context does not support an answer.",
        context=None,
        context_snippets=[],
    )


def test_rag_eval_case_requires_sources_for_supported_case():
    with pytest.raises(
        ValidationError,
        match="Supported RAG cases require expected source paths",
    ):
        RagEvalCase(
            case_id="invalid-supported-case",
            query="What does the API do?",
            expected_source_paths=[],
            expect_refusal=False,
        )


def test_rag_eval_case_allows_empty_sources_for_refusal_case():
    case = RagEvalCase(
        case_id="unsupported-question",
        query="What is tomorrow's weather?",
        expected_source_paths=[],
        expect_refusal=True,
    )

    assert case.expect_refusal is True
    assert case.expected_source_paths == []


def test_load_rag_cases_validates_fixture(tmp_path):
    fixture_path = tmp_path / "rag_cases.json"
    fixture_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "prediction-api",
                    "query": "How does the prediction API work?",
                    "expected_source_paths": [
                        "source_code/api.py",
                    ],
                    "expect_refusal": False,
                    "top_k": 4,
                    "min_score": 0.2,
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_rag_cases(fixture_path)

    assert len(cases) == 1
    assert cases[0].case_id == "prediction-api"
    assert cases[0].top_k == 4
    assert cases[0].min_score == pytest.approx(0.2)


def test_evaluate_supported_answer_passes():
    result = evaluate_rag_response(
        make_case(),
        make_answered_response(),
    )

    assert result.status == "answered"
    assert result.schema_valid is True
    assert result.retrieval_hit is True
    assert result.citations_valid is True
    assert result.citation_hit is True
    assert result.refusal_correct is True
    assert result.passed is True
    assert result.failure_reasons == []


def test_source_path_matching_accepts_windows_absolute_paths():
    source_path = (
        r"C:\Dev\anomalyops-copilot\data_sources"
        r"\anomaly_detection_platform\source_code\api.py"
    )

    result = evaluate_rag_response(
        make_case(),
        make_answered_response(
            source_path=source_path,
        ),
    )

    assert result.retrieval_hit is True
    assert result.citation_hit is True
    assert result.passed is True


def test_source_path_matching_is_case_insensitive():
    result = evaluate_rag_response(
        make_case(
            expected_source_paths=[
                "SOURCE_CODE/API.PY",
            ]
        ),
        make_answered_response(
            source_path=(
                "data_sources/anomaly_detection_platform/"
                "source_code/api.py"
            ),
        ),
    )

    assert result.retrieval_hit is True
    assert result.citation_hit is True
    assert result.passed is True


def test_evaluate_answer_reports_retrieval_and_citation_misses():
    result = evaluate_rag_response(
        make_case(),
        make_answered_response(
            source_path="source_code/dashboard.py",
        ),
    )

    assert result.schema_valid is True
    assert result.retrieval_hit is False
    assert result.citations_valid is True
    assert result.citation_hit is False
    assert result.refusal_correct is True
    assert result.passed is False
    assert result.failure_reasons == [
        "Expected source was not retrieved.",
        "No citation referenced an expected source.",
    ]


def test_evaluate_detects_citation_metadata_mismatch():
    result = evaluate_rag_response(
        make_case(),
        make_answered_response(
            citation_start_line=11,
            context_start_line=10,
        ),
    )

    assert result.retrieval_hit is True
    assert result.citations_valid is False
    assert result.citation_hit is True
    assert result.passed is False
    assert result.failure_reasons == [
        "Citations were missing or inconsistent with context.",
    ]


def test_evaluate_detects_missing_inline_citation():
    result = evaluate_rag_response(
        make_case(),
        make_answered_response(
            answer="The API accepts engineered features.",
        ),
    )

    assert result.citations_valid is False
    assert result.passed is False


def test_evaluate_allows_additional_bracketed_numbers():
    result = evaluate_rag_response(
        make_case(),
        make_answered_response(
            answer=(
                "The 2026 API accepts engineered features "
                "[1], with a limit of [100]."
            ),
        ),
    )

    assert result.citations_valid is True
    assert result.passed is True


def test_evaluate_rejects_duplicate_citation_ids():
    response = make_answered_response()
    response.citations.append(
        Citation(
            citation_id=1,
            source_path="source_code/api.py",
            start_line=10,
            end_line=20,
        )
    )

    result = evaluate_rag_response(
        make_case(),
        response,
    )

    assert result.citations_valid is False
    assert result.passed is False


def test_evaluate_correct_refusal_passes():
    case = make_case(
        case_id="unsupported-weather",
        expected_source_paths=[],
        expect_refusal=True,
    )

    result = evaluate_rag_response(
        case,
        make_refusal_response(),
    )

    assert result.status == "refused"
    assert result.schema_valid is True
    assert result.retrieval_hit is True
    assert result.citations_valid is True
    assert result.citation_hit is True
    assert result.refusal_correct is True
    assert result.passed is True


def test_evaluate_unexpected_refusal_fails_supported_case():
    result = evaluate_rag_response(
        make_case(),
        make_refusal_response(),
    )

    assert result.status == "refused"
    assert result.refusal_correct is False
    assert result.citation_hit is False
    assert result.passed is False
    assert (
        "Refusal behavior did not match the case expectation."
        in result.failure_reasons
    )


def test_evaluate_answer_fails_expected_refusal_case():
    case = make_case(
        case_id="unsupported-weather",
        expected_source_paths=[],
        expect_refusal=True,
    )

    result = evaluate_rag_response(
        case,
        make_answered_response(),
    )

    assert result.status == "answered"
    assert result.refusal_correct is False
    assert result.passed is False
    assert result.failure_reasons == [
        "Refusal behavior did not match the case expectation.",
    ]


def test_evaluate_invalid_response_schema():
    result = evaluate_rag_response(
        make_case(),
        {
            "answer": "Incomplete response",
        },
    )

    assert result.status == "invalid"
    assert result.schema_valid is False
    assert result.passed is False
    assert result.failure_reasons == [
        "Response did not match the QueryResponse schema.",
    ]


def test_run_rag_evals_builds_requests_and_aggregates_rates():
    cases = [
        RagEvalCase(
            case_id="supported-pass",
            query="prediction API",
            expected_source_paths=[
                "source_code/api.py",
            ],
        ),
        RagEvalCase(
            case_id="supported-fail",
            query="model threshold",
            expected_source_paths=[
                "source_code/config.py",
            ],
        ),
        RagEvalCase(
            case_id="refusal-pass",
            query="tomorrow weather",
            expected_source_paths=[],
            expect_refusal=True,
        ),
    ]

    received_requests: list[QueryRequest] = []

    def execute_query(
        request: QueryRequest,
    ) -> QueryResponse:
        received_requests.append(request)

        if request.query == "prediction API":
            return make_answered_response()

        if request.query == "model threshold":
            return make_answered_response(
                source_path="source_code/dashboard.py",
            )

        return make_refusal_response()

    report = run_rag_evals(
        cases,
        execute_query,
    )

    assert len(received_requests) == 3

    for request, case in zip(
        received_requests,
        cases,
        strict=True,
    ):
        assert request.query == case.query
        assert request.top_k == case.top_k
        assert request.min_score == case.min_score
        assert request.show_context is True

    assert report.total_cases == 3
    assert report.passed_cases == 2
    assert report.failed_cases == 1
    assert report.supported_cases == 2
    assert report.refusal_cases == 1
    assert report.schema_validity_rate == pytest.approx(1.0)
    assert report.retrieval_hit_rate == pytest.approx(0.5)
    assert report.citation_validity_rate == pytest.approx(1.0)
    assert report.citation_hit_rate == pytest.approx(0.5)
    assert report.refusal_accuracy == pytest.approx(1.0)
    assert report.pass_rate == pytest.approx(2 / 3)


def test_run_rag_evals_records_executor_error_and_continues():
    cases = [
        make_case(case_id="broken"),
        make_case(
            case_id="refusal",
            expected_source_paths=[],
            expect_refusal=True,
        ),
    ]

    call_count = 0

    def execute_query(
        request: QueryRequest,
    ) -> QueryResponse:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            raise RuntimeError(
                "private provider failure"
            )

        return make_refusal_response()

    report = run_rag_evals(
        cases,
        execute_query,
    )

    assert call_count == 2
    assert report.total_cases == 2
    assert report.passed_cases == 1
    assert report.failed_cases == 1

    failed_result = report.results[0]

    assert failed_result.status == "invalid"
    assert failed_result.failure_reasons == [
        "Query execution failed with RuntimeError.",
    ]
    assert "private provider failure" not in str(
        failed_result
    )


def test_run_rag_evals_handles_empty_cases():
    report = run_rag_evals(
        [],
        lambda request: make_refusal_response(),
    )

    assert report.total_cases == 0
    assert report.passed_cases == 0
    assert report.failed_cases == 0
    assert report.supported_cases == 0
    assert report.refusal_cases == 0
    assert report.schema_validity_rate == 0.0
    assert report.retrieval_hit_rate == 0.0
    assert report.citation_validity_rate == 0.0
    assert report.citation_hit_rate == 0.0
    assert report.refusal_accuracy == 0.0
    assert report.pass_rate == 0.0
    assert report.results == []