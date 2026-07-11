import argparse
from pathlib import Path

from copilot.evals.runner import load_retrieval_cases, run_retrieval_evals
from copilot.ingestion.manifest import load_chunk_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path")
    parser.add_argument("fixture_path")
    
    args = parser.parse_args(argv)
    
    chunks = load_chunk_manifest(Path(args.manifest_path))
    cases = load_retrieval_cases(Path(args.fixture_path))
    retrieval_eval_report = run_retrieval_evals(chunks, cases)
    
    print("Retrieval evaluation")
    print(f"Total: {retrieval_eval_report.total_cases}")
    print(f"Passed: {retrieval_eval_report.passed_cases}")
    print(f"Failed: {retrieval_eval_report.failed_cases}")
    print(f"Hit rate: {retrieval_eval_report.hit_rate:.2%}")
    
    for result in retrieval_eval_report.results:
        if result.passed:
            print(f"PASS {result.case_id}")
        else:
            print(f"FAIL {result.case_id}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())