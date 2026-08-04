import json
import re

from pathlib import Path, PurePosixPath
from collections.abc import Callable
from pydantic import ValidationError

from copilot.retrieval.search import retrieve_relevant_chunks
from copilot.evals.schemas import (
    RetrievalEvalCase,
    RetrievalEvalReport,
    RetrievalEvalResult,
    RagEvalCase,
    RagEvalReport,
    RagEvalResult,
)
from copilot.schemas.chunk import SourceChunk
from copilot.schemas.query import (
    QueryRequest,
    QueryResponse,
)


def load_retrieval_cases(fixture_path: Path) -> list[RetrievalEvalCase]:
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    
    validated_cases = [
        RetrievalEvalCase(**case)
        for case in cases
    ]
    
    return validated_cases


def run_retrieval_evals(chunks: list[SourceChunk], cases: list[RetrievalEvalCase]) -> RetrievalEvalReport:
    retrieval_eval_results = []
    
    for case in cases:
        scored_chunks = retrieve_relevant_chunks(
            case.query,
            chunks,
            case.top_k,
        )
        
        retrieved_source_paths = [
            scored_chunk.chunk.source_path
            for scored_chunk in scored_chunks
        ]
        
        passed = bool(
            set(case.expected_source_paths)
            & set(retrieved_source_paths)
        )
        
        retrieval_eval_results.append(
            RetrievalEvalResult(
                case_id=case.case_id,
                query=case.query,
                expected_source_paths=case.expected_source_paths,
                retrieved_source_paths=retrieved_source_paths,
                passed=passed,
            )
        )
    
    total_cases = len(retrieval_eval_results)
    passed_cases = sum(result.passed for result in retrieval_eval_results)
    failed_cases=sum(not result.passed for result in retrieval_eval_results)
    
    retrieval_eval_report = RetrievalEvalReport(
        total_cases=len(retrieval_eval_results),
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        hit_rate=passed_cases / total_cases if total_cases else 0.0,
        results=retrieval_eval_results,
    )
    
    return retrieval_eval_report


RagQueryExecutor = Callable[
    [QueryRequest],
    QueryResponse | dict[str, object],
]


def load_rag_cases(
    fixture_path: Path,
) -> list[RagEvalCase]:
    raw_cases = json.loads(
        fixture_path.read_text(
            encoding="utf-8"
        )
    )

    return [
        RagEvalCase.model_validate(case)
        for case in raw_cases
    ]


def _unique_paths(
    paths: list[str],
) -> list[str]:
    return list(dict.fromkeys(paths))


def _normalize_source_path(
    source_path: str,
) -> str:
    normalized = source_path.replace(
        "\\",
        "/",
    ).lstrip("./")

    return str(
        PurePosixPath(normalized)
    ).lower()


def _source_path_matches(
    actual_path: str,
    expected_path: str,
) -> bool:
    actual = _normalize_source_path(
        actual_path
    )
    expected = _normalize_source_path(
        expected_path
    )

    return (
        actual == expected
        or actual.endswith(
            f"/{expected}"
        )
    )


def _contains_expected_source(
    actual_paths: list[str],
    expected_paths: list[str],
) -> bool:
    return any(
        _source_path_matches(
            actual_path,
            expected_path,
        )
        for actual_path in actual_paths
        for expected_path in expected_paths
    )


def _validate_response_citations(
    response: QueryResponse,
) -> bool:
    refused = (
        response.refusal_reason is not None
    )

    if refused:
        return (
            response.answer == ""
            and response.citations == []
        )

    if (
        not response.answer.strip()
        or not response.citations
    ):
        return False

    context_by_id = {
        snippet.citation_id: snippet
        for snippet in response.context_snippets
    }

    if len(context_by_id) != len(
        response.context_snippets
    ):
        return False

    citation_ids = [
        citation.citation_id
        for citation in response.citations
    ]

    if len(citation_ids) != len(
        set(citation_ids)
    ):
        return False

    for citation in response.citations:
        snippet = context_by_id.get(
            citation.citation_id
        )

        if snippet is None:
            return False

        if (
            citation.source_path
            != snippet.source_path
            or citation.start_line
            != snippet.start_line
            or citation.end_line
            != snippet.end_line
        ):
            return False

    inline_ids = {
        int(match)
        for match in re.findall(
            r"\[(\d+)\]",
            response.answer,
        )
    }

    return set(citation_ids).issubset(
        inline_ids
    )


def _invalid_rag_result(
    case: RagEvalCase,
    reason: str,
) -> RagEvalResult:
    return RagEvalResult(
        case_id=case.case_id,
        status="invalid",
        expected_source_paths=(
            case.expected_source_paths
        ),
        retrieved_source_paths=[],
        cited_source_paths=[],
        schema_valid=False,
        retrieval_hit=False,
        citations_valid=False,
        citation_hit=False,
        refusal_correct=False,
        passed=False,
        failure_reasons=[reason],
    )


def evaluate_rag_response(
    case: RagEvalCase,
    raw_response: (
        QueryResponse
        | dict[str, object]
    ),
) -> RagEvalResult:
    try:
        response = QueryResponse.model_validate(
            raw_response
        )
    except ValidationError:
        return _invalid_rag_result(
            case,
            "Response did not match the QueryResponse schema.",
        )

    retrieved_source_paths = _unique_paths(
        [
            snippet.source_path
            for snippet in response.context_snippets
        ]
    )
    cited_source_paths = _unique_paths(
        [
            citation.source_path
            for citation in response.citations
        ]
    )

    retrieval_hit = (
        True
        if case.expect_refusal
        else _contains_expected_source(
            retrieved_source_paths,
            case.expected_source_paths,
        )
    )

    citations_valid = (
        _validate_response_citations(
            response
        )
    )

    citation_hit = (
        True
        if case.expect_refusal
        else _contains_expected_source(
            cited_source_paths,
            case.expected_source_paths,
        )
    )

    actual_refusal = (
        response.refusal_reason is not None
    )
    refusal_correct = (
        actual_refusal
        == case.expect_refusal
    )

    failure_reasons: list[str] = []

    if not retrieval_hit:
        failure_reasons.append(
            "Expected source was not retrieved."
        )

    if not citations_valid:
        failure_reasons.append(
            "Citations were missing or inconsistent with context."
        )

    if not citation_hit:
        failure_reasons.append(
            "No citation referenced an expected source."
        )

    if not refusal_correct:
        failure_reasons.append(
            "Refusal behavior did not match the case expectation."
        )

    passed = not failure_reasons

    return RagEvalResult(
        case_id=case.case_id,
        status=(
            "refused"
            if actual_refusal
            else "answered"
        ),
        expected_source_paths=(
            case.expected_source_paths
        ),
        retrieved_source_paths=(
            retrieved_source_paths
        ),
        cited_source_paths=(
            cited_source_paths
        ),
        schema_valid=True,
        retrieval_hit=retrieval_hit,
        citations_valid=citations_valid,
        citation_hit=citation_hit,
        refusal_correct=refusal_correct,
        passed=passed,
        failure_reasons=failure_reasons,
    )


def run_rag_evals(
    cases: list[RagEvalCase],
    execute_query: RagQueryExecutor,
) -> RagEvalReport:
    results: list[RagEvalResult] = []

    for case in cases:
        request = QueryRequest(
            query=case.query,
            top_k=case.top_k,
            min_score=case.min_score,
            show_context=True,
        )

        try:
            response = execute_query(
                request
            )
        except Exception as exc:
            results.append(
                _invalid_rag_result(
                    case,
                    (
                        "Query execution failed with "
                        f"{type(exc).__name__}."
                    ),
                )
            )
            continue

        results.append(
            evaluate_rag_response(
                case,
                response,
            )
        )

    total_cases = len(results)
    passed_cases = sum(
        result.passed
        for result in results
    )
    failed_cases = (
        total_cases - passed_cases
    )

    supported_results = [
        result
        for case, result in zip(
            cases,
            results,
            strict=True,
        )
        if not case.expect_refusal
    ]
    refusal_results = [
        result
        for case, result in zip(
            cases,
            results,
            strict=True,
        )
        if case.expect_refusal
    ]

    supported_cases = len(
        supported_results
    )
    refusal_cases = len(
        refusal_results
    )

    def rate(
        numerator: int,
        denominator: int,
    ) -> float:
        if denominator == 0:
            return 0.0

        return numerator / denominator

    return RagEvalReport(
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        supported_cases=supported_cases,
        refusal_cases=refusal_cases,
        schema_validity_rate=rate(
            sum(
                result.schema_valid
                for result in results
            ),
            total_cases,
        ),
        retrieval_hit_rate=rate(
            sum(
                result.retrieval_hit
                for result in supported_results
            ),
            supported_cases,
        ),
        citation_validity_rate=rate(
            sum(
                result.citations_valid
                for result in supported_results
            ),
            supported_cases,
        ),
        citation_hit_rate=rate(
            sum(
                result.citation_hit
                for result in supported_results
            ),
            supported_cases,
        ),
        refusal_accuracy=rate(
            sum(
                result.refusal_correct
                for result in results
            ),
            total_cases,
        ),
        pass_rate=rate(
            passed_cases,
            total_cases,
        ),
        results=results,
    )