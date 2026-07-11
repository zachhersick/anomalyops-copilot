import json

from pathlib import Path

from copilot.retrieval.search import retrieve_relevant_chunks
from copilot.evals.schemas import RetrievalEvalCase, RetrievalEvalReport, RetrievalEvalResult
from copilot.schemas.chunk import SourceChunk


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