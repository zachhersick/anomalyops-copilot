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
    TriageEvalCase,
    TriageEvalReport,
    TriageEvalResult,
    TriageEvalExpectedFinding,
)
from copilot.schemas.chunk import SourceChunk
from copilot.schemas.query import (
    QueryRequest,
    QueryResponse,
)
from copilot.schemas.triage import (
    TriageFinding,
    TriageReport,
    TriageRequest,
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


def _relevant_source_stats(
    actual_paths: list[str],
    expected_paths: list[str],
) -> tuple[int | None, float]:
    actual_paths = actual_paths[:5]
    first_rank = next(
        (
            rank
            for rank, actual_path in enumerate(
                actual_paths,
                start=1,
            )
            if any(
                _source_path_matches(
                    actual_path,
                    expected_path,
                )
                for expected_path in expected_paths
            )
        ),
        None,
    )
    matched_expected = sum(
        any(
            _source_path_matches(
                actual_path,
                expected_path,
            )
            for actual_path in actual_paths
        )
        for expected_path in expected_paths
    )

    return (
        first_rank,
        matched_expected / len(expected_paths),
    )


def _normalize_answer_text(value: str) -> str:
    return " ".join(
        value.lower()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )


def _answer_terms_present(
    answer: str,
    expected_terms: list[str],
) -> bool | None:
    if not expected_terms:
        return None

    normalized_answer = _normalize_answer_text(
        answer
    )

    return all(
        _normalize_answer_text(term)
        in normalized_answer
        for term in expected_terms
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
        relevant_source_recall=(
            None
            if case.expect_refusal
            else 0.0
        ),
        answer_terms_present=(
            False
            if case.expected_answer_terms
            else None
        ),
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

    ranked_source_paths = [
        snippet.source_path
        for snippet in response.context_snippets
    ]
    retrieved_source_paths = _unique_paths(
        ranked_source_paths
    )
    cited_source_paths = _unique_paths(
        [
            citation.source_path
            for citation in response.citations
        ]
    )

    if case.expect_refusal:
        first_relevant_rank = None
        relevant_source_recall = None
    else:
        (
            first_relevant_rank,
            relevant_source_recall,
        ) = _relevant_source_stats(
            ranked_source_paths,
            case.expected_source_paths,
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
    answer_terms_present = (
        _answer_terms_present(
            response.answer,
            case.expected_answer_terms,
        )
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

    if answer_terms_present is False:
        failure_reasons.append(
            "Answer did not contain all expected terms."
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
        first_relevant_rank=(
            first_relevant_rank
        ),
        relevant_source_recall=(
            relevant_source_recall
        ),
        answer_terms_present=(
            answer_terms_present
        ),
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

    answer_term_results = [
        result
        for result in supported_results
        if result.answer_terms_present
        is not None
    ]
    actual_refusals = [
        result
        for result in results
        if result.status == "refused"
    ]
    true_refusals = sum(
        result.status == "refused"
        for result in refusal_results
    )

    def optional_rate(
        numerator: int,
        denominator: int,
    ) -> float | None:
        if denominator == 0:
            return None

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
        hit_rate_at_3=rate(
            sum(
                result.first_relevant_rank
                is not None
                and result.first_relevant_rank <= 3
                for result in supported_results
            ),
            supported_cases,
        ),
        hit_rate_at_5=rate(
            sum(
                result.first_relevant_rank
                is not None
                and result.first_relevant_rank <= 5
                for result in supported_results
            ),
            supported_cases,
        ),
        mean_reciprocal_rank_at_5=rate(
            sum(
                1 / result.first_relevant_rank
                for result in supported_results
                if result.first_relevant_rank
                is not None
                and result.first_relevant_rank <= 5
            ),
            supported_cases,
        ),
        mean_source_recall_at_5=rate(
            sum(
                result.relevant_source_recall
                or 0.0
                for result in supported_results
            ),
            supported_cases,
        ),
        answer_term_accuracy=optional_rate(
            sum(
                result.answer_terms_present
                is True
                for result in answer_term_results
            ),
            len(answer_term_results),
        ),
        refusal_precision=optional_rate(
            true_refusals,
            len(actual_refusals),
        ),
        refusal_recall=optional_rate(
            true_refusals,
            refusal_cases,
        ),
        results=results,
    )
    
    
TriageExecutor = Callable[
    [TriageRequest],
    TriageReport | dict[str, object],
]


def load_triage_cases(
    fixture_path: Path,
) -> list[TriageEvalCase]:
    raw_cases = json.loads(
        fixture_path.read_text(
            encoding="utf-8",
        )
    )

    return [
        TriageEvalCase.model_validate(case)
        for case in raw_cases
    ]


def _finding_matches_expected(
    finding: TriageFinding,
    expected: TriageEvalExpectedFinding,
) -> bool:
    if (
        finding.severity.lower()
        != expected.severity.lower()
    ):
        return False

    if finding.machine_id != expected.machine_id:
        return False

    if finding.sensor != expected.sensor:
        return False

    if (
        expected.anomaly_type is not None
        and finding.anomaly_type
        != expected.anomaly_type
    ):
        return False

    return True


def _expected_findings_present(
    report: TriageReport,
    expected_findings: list[
        TriageEvalExpectedFinding
    ],
) -> bool:
    unmatched_indices = set(
        range(len(report.findings))
    )

    for expected in expected_findings:
        matched_index = next(
            (
                index
                for index in unmatched_indices
                if _finding_matches_expected(
                    report.findings[index],
                    expected,
                )
            ),
            None,
        )

        if matched_index is None:
            return False

        unmatched_indices.remove(
            matched_index
        )

    return True


def _run_is_consistent(
    report: TriageReport,
) -> bool:
    if (
        report.run_summary is not None
        and report.run_id
        != report.run_summary.run_id
    ):
        return False

    for evidence in report.evidence:
        if (
            report.run_id is not None
            and evidence.event.run_id
            != report.run_id
        ):
            return False

        for alert in evidence.alerts:
            if (
                alert.run_id
                != evidence.event.run_id
            ):
                return False

    return True


def _evidence_is_valid(
    report: TriageReport,
) -> bool:
    finding_ids = [
        finding.finding_id
        for finding in report.findings
    ]

    if len(finding_ids) != len(
        set(finding_ids)
    ):
        return False

    evidence_ids = [
        evidence.evidence_id
        for evidence in report.evidence
    ]

    if len(evidence_ids) != len(
        set(evidence_ids)
    ):
        return False

    evidence_by_id = {
        evidence.evidence_id: evidence
        for evidence in report.evidence
    }

    for finding in report.findings:
        if not finding.evidence_ids:
            return False

        if len(finding.evidence_ids) != len(
            set(finding.evidence_ids)
        ):
            return False

        for evidence_id in (
            finding.evidence_ids
        ):
            evidence = evidence_by_id.get(
                evidence_id
            )

            if evidence is None:
                return False

            event = evidence.event

            if (
                finding.machine_id
                != event.machine_id
            ):
                return False

            if finding.sensor != event.sensor:
                return False

            if (
                finding.anomaly_type
                != event.anomaly_type
            ):
                return False

            if (
                event.max_severity is not None
                and finding.severity.lower()
                != event.max_severity.lower()
            ):
                return False

            for alert in evidence.alerts:
                if (
                    alert.machine_id
                    != event.machine_id
                ):
                    return False

                if alert.sensor != event.sensor:
                    return False

                if (
                    alert.anomaly_type is not None
                    and event.anomaly_type
                    is not None
                    and alert.anomaly_type
                    != event.anomaly_type
                ):
                    return False

    return True


def _status_semantics_are_valid(
    report: TriageReport,
) -> bool:
    if report.status == "completed":
        return (
            report.run_id is not None
            and report.run_summary is not None
            and bool(report.findings)
            and report.refusal_reason is None
        )

    if report.status == "no_alerts":
        return (
            report.run_id is not None
            and report.run_summary is not None
            and report.run_summary.total_alert_events
            == 0
            and report.findings == []
            and report.evidence == []
            and report.refusal_reason is None
        )

    if report.status == "incomplete_data":
        return bool(
            report.refusal_reason
            and report.refusal_reason.strip()
        )

    if report.status == "refused":
        return (
            bool(
                report.refusal_reason
                and report.refusal_reason.strip()
            )
            and report.findings == []
        )

    return False


def _invalid_triage_result(
    case: TriageEvalCase,
    reason: str,
) -> TriageEvalResult:
    return TriageEvalResult(
        case_id=case.case_id,
        status="invalid",
        schema_valid=False,
        status_correct=(
            False
            if case.expected_status
            is not None
            else None
        ),
        finding_count_correct=(
            False
            if case.expected_finding_count
            is not None
            else None
        ),
        expected_findings_present=(
            False
            if case.expected_findings
            else None
        ),
        evidence_valid=False,
        run_consistent=False,
        max_events_respected=False,
        status_semantics_valid=False,
        passed=False,
        failure_reasons=[reason],
    )


def evaluate_triage_response(
    case: TriageEvalCase,
    raw_report: (
        TriageReport
        | dict[str, object]
    ),
) -> TriageEvalResult:
    try:
        report = TriageReport.model_validate(
            raw_report
        )
    except ValidationError:
        return _invalid_triage_result(
            case,
            (
                "Response did not match the "
                "TriageReport schema."
            ),
        )

    status_correct = (
        report.status
        == case.expected_status
        if case.expected_status is not None
        else None
    )

    finding_count_correct = (
        len(report.findings)
        == case.expected_finding_count
        if case.expected_finding_count
        is not None
        else None
    )

    expected_findings_present = (
        _expected_findings_present(
            report,
            case.expected_findings,
        )
        if case.expected_findings
        else None
    )

    evidence_valid = _evidence_is_valid(
        report
    )
    run_consistent = _run_is_consistent(
        report
    )
    max_events_respected = (
        len(report.findings)
        <= case.request.max_events
    )
    status_semantics_valid = (
        _status_semantics_are_valid(
            report
        )
    )

    failure_reasons: list[str] = []

    if status_correct is False:
        failure_reasons.append(
            "Triage status did not match the expected status."
        )

    if finding_count_correct is False:
        failure_reasons.append(
            "Finding count did not match the expected count."
        )

    if expected_findings_present is False:
        failure_reasons.append(
            "Expected triage findings were missing."
        )

    if not evidence_valid:
        failure_reasons.append(
            "Finding evidence was missing or inconsistent."
        )

    if not run_consistent:
        failure_reasons.append(
            "Run identifiers were inconsistent."
        )

    if not max_events_respected:
        failure_reasons.append(
            "Triage report exceeded max_events."
        )

    if not status_semantics_valid:
        failure_reasons.append(
            "Triage status semantics were invalid."
        )

    return TriageEvalResult(
        case_id=case.case_id,
        status=report.status,
        schema_valid=True,
        status_correct=status_correct,
        finding_count_correct=(
            finding_count_correct
        ),
        expected_findings_present=(
            expected_findings_present
        ),
        evidence_valid=evidence_valid,
        run_consistent=run_consistent,
        max_events_respected=(
            max_events_respected
        ),
        status_semantics_valid=(
            status_semantics_valid
        ),
        passed=not failure_reasons,
        failure_reasons=failure_reasons,
    )


def run_triage_evals(
    cases: list[TriageEvalCase],
    execute_triage: TriageExecutor,
) -> TriageEvalReport:
    results: list[TriageEvalResult] = []

    for case in cases:
        try:
            report = execute_triage(
                case.request
            )
        except Exception as exc:
            results.append(
                _invalid_triage_result(
                    case,
                    (
                        "Triage execution failed "
                        f"with {type(exc).__name__}."
                    ),
                )
            )
            continue

        results.append(
            evaluate_triage_response(
                case,
                report,
            )
        )

    total_cases = len(results)
    passed_cases = sum(
        result.passed
        for result in results
    )

    def rate(
        values: list[bool],
    ) -> float | None:
        if not values:
            return None

        return sum(values) / len(values)

    def required_rate(
        values: list[bool],
    ) -> float:
        if not values:
            return 0.0

        return sum(values) / len(values)

    status_results = [
        result.status_correct
        for result in results
        if result.status_correct is not None
    ]
    finding_results = [
        result.expected_findings_present
        for result in results
        if result.expected_findings_present
        is not None
    ]
    count_results = [
        result.finding_count_correct
        for result in results
        if result.finding_count_correct
        is not None
    ]

    return TriageEvalReport(
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=(
            total_cases - passed_cases
        ),
        schema_validity_rate=required_rate(
            [
                result.schema_valid
                for result in results
            ]
        ),
        evidence_validity_rate=required_rate(
            [
                result.evidence_valid
                for result in results
            ]
        ),
        run_consistency_rate=required_rate(
            [
                result.run_consistent
                for result in results
            ]
        ),
        max_events_compliance_rate=required_rate(
            [
                result.max_events_respected
                for result in results
            ]
        ),
        status_semantics_rate=required_rate(
            [
                result.status_semantics_valid
                for result in results
            ]
        ),
        status_accuracy=rate(
            status_results
        ),
        expected_findings_accuracy=rate(
            finding_results
        ),
        finding_count_accuracy=rate(
            count_results
        ),
        pass_rate=required_rate(
            [
                result.passed
                for result in results
            ]
        ),
        results=results,
    )
