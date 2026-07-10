import json

from pathlib import Path

from copilot.evals.schemas import RetrievalEvalCase


def test_retrieval_eval_fixtures_are_valid():
    fixture_path = Path("evals/retrieval_cases.json")
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    
    assert len(cases) >= 5
    
    validated_cases = [
        RetrievalEvalCase(**case)
        for case in cases
    ]
    
    case_ids = [case.case_id for case in validated_cases]
    
    assert len(set(case_ids)) == len(case_ids)
    assert all(case.expected_source_paths for case in validated_cases)
    assert all(case.top_k > 0 for case in validated_cases)
    