from copilot.evals.runner import run_retrieval_evals
from copilot.evals.schemas import RetrievalEvalCase
from copilot.schemas.chunk import SourceChunk
from copilot.schemas.retrieval import ScoredChunk


def make_scored_chunk(
    chunk_id: str,
    source_path: str,
    score: float,
) -> ScoredChunk:
    return ScoredChunk(
        chunk=SourceChunk(
            chunk_id=chunk_id,
            source_id=f"anomaly_detection_platform:{source_path}",
            project_name="anomaly_detection_platform",
            source_type="python",
            source_path=source_path,
            chunk_index=0,
            content=f"Content from {source_path}",
            start_line=1,
            end_line=5,
        ),
        score=score,
    )


def test_run_retrieval_evals_reports_passes_failures_and_path_order(
    monkeypatch,
):
    def fake_retrieve_relevant_chunks(query, chunks, top_k):
        if query == "model threshold":
            return [
                make_scored_chunk(
                    "chunk-1",
                    "source_code/config.py",
                    0.9,
                ),
                make_scored_chunk(
                    "chunk-2",
                    "README.md",
                    0.8,
                ),
            ]

        return [
            make_scored_chunk(
                "chunk-3",
                "source_code/dashboard.py",
                0.7,
            ),
        ]

    monkeypatch.setattr(
        "copilot.evals.runner.retrieve_relevant_chunks",
        fake_retrieve_relevant_chunks,
    )

    cases = [
        RetrievalEvalCase(
            case_id="model-threshold",
            query="model threshold",
            expected_source_paths=["source_code/config.py"],
            top_k=3,
        ),
        RetrievalEvalCase(
            case_id="alert-severity",
            query="alert severity",
            expected_source_paths=["source_code/alerts.py"],
            top_k=3,
        ),
    ]

    report = run_retrieval_evals([], cases)

    assert report.total_cases == 2
    assert report.passed_cases == 1
    assert report.failed_cases == 1
    assert report.hit_rate == 0.5

    assert report.results[0].passed is True
    assert report.results[0].retrieved_source_paths == [
        "source_code/config.py",
        "README.md",
    ]

    assert report.results[1].passed is False
    assert report.results[1].retrieved_source_paths == [
        "source_code/dashboard.py",
    ]


def test_run_retrieval_evals_handles_empty_cases():
    report = run_retrieval_evals([], [])

    assert report.total_cases == 0
    assert report.passed_cases == 0
    assert report.failed_cases == 0
    assert report.hit_rate == 0.0
    assert report.results == []