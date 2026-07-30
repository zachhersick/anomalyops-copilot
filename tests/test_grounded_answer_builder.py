import pytest

from copilot.answering.grounded import build_grounded_answer
from copilot.schemas.answer import Citation, GroundedAnswer
from copilot.schemas.chunk import SourceChunk
from copilot.schemas.retrieval import ScoredChunk


class RecordingGroundedAnswerGenerator:
    provider_name = "recording"
    model_name = "recording-test"

    def __init__(
        self,
        result: GroundedAnswer,
    ) -> None:
        self.result = result
        self.calls: list[dict] = []

    def generate(
        self,
        query: str,
        context: list[ScoredChunk],
    ) -> GroundedAnswer:
        self.calls.append(
            {
                "query": query,
                "context": context,
            }
        )
        return self.result


def make_scored_chunk(
    chunk_id: str,
    content: str,
    score: float,
    *,
    source_path: str = "source.py",
    start_line: int = 1,
    end_line: int = 5,
    chunk_index: int = 0,
) -> ScoredChunk:
    return ScoredChunk(
        chunk=SourceChunk(
            chunk_id=chunk_id,
            source_id=source_path,
            project_name="test-project",
            source_type="python",
            source_path=source_path,
            chunk_index=chunk_index,
            content=content,
            start_line=start_line,
            end_line=end_line,
        ),
        score=score,
    )


def make_generated_answer() -> GroundedAnswer:
    return GroundedAnswer(
        answer="Generated answer. [2]",
        citations=[
            Citation(
                citation_id=2,
                source_path="second.py",
                start_line=20,
                end_line=30,
            )
        ],
        confidence=0.87,
        refusal_reason=None,
    )


def test_build_grounded_answer_refuses_when_context_is_empty():
    generator = RecordingGroundedAnswerGenerator(
        make_generated_answer()
    )

    result = build_grounded_answer(
        query="Where is the endpoint?",
        scored_chunks=[],
        min_score=0.0,
        generator=generator,
    )

    assert result == GroundedAnswer(
        answer="",
        citations=[],
        confidence=0.0,
        refusal_reason="No relevant context was retrieved.",
    )
    assert generator.calls == []


def test_build_grounded_answer_refuses_below_minimum_score():
    generator = RecordingGroundedAnswerGenerator(
        make_generated_answer()
    )
    chunks = [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="Some context.",
            score=0.49,
        ),
    ]

    result = build_grounded_answer(
        query="Question",
        scored_chunks=chunks,
        min_score=0.5,
        generator=generator,
    )

    assert result == GroundedAnswer(
        answer="",
        citations=[],
        confidence=0.49,
        refusal_reason=(
            "Retrieved context was below the confidence threshold."
        ),
    )
    assert generator.calls == []


def test_build_grounded_answer_uses_highest_score_for_threshold():
    generator = RecordingGroundedAnswerGenerator(
        make_generated_answer()
    )
    chunks = [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="Low-scoring context.",
            score=0.2,
        ),
        make_scored_chunk(
            chunk_id="chunk-2",
            content="High-scoring context.",
            score=0.8,
        ),
    ]

    result = build_grounded_answer(
        query="Question",
        scored_chunks=chunks,
        min_score=0.5,
        generator=generator,
    )

    assert result is generator.result
    assert len(generator.calls) == 1


def test_build_grounded_answer_allows_score_equal_to_threshold():
    generator = RecordingGroundedAnswerGenerator(
        make_generated_answer()
    )
    chunks = [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="Supporting context.",
            score=0.5,
        ),
    ]

    result = build_grounded_answer(
        query="Question",
        scored_chunks=chunks,
        min_score=0.5,
        generator=generator,
    )

    assert result is generator.result
    assert len(generator.calls) == 1


@pytest.mark.parametrize(
    ("score", "expected_confidence"),
    [
        (1.4, 1.0),
        (-0.4, 0.0),
    ],
)
def test_build_grounded_answer_clamps_refusal_confidence(
    score,
    expected_confidence,
):
    generator = RecordingGroundedAnswerGenerator(
        make_generated_answer()
    )
    chunks = [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="Context.",
            score=score,
        ),
    ]

    min_score = 2.0 if score > 0 else 0.5

    result = build_grounded_answer(
        query="Question",
        scored_chunks=chunks,
        min_score=min_score,
        generator=generator,
    )

    assert result.confidence == expected_confidence
    assert result.refusal_reason == (
        "Retrieved context was below the confidence threshold."
    )
    assert generator.calls == []


def test_build_grounded_answer_passes_query_and_context_unchanged():
    generated_answer = make_generated_answer()
    generator = RecordingGroundedAnswerGenerator(
        generated_answer
    )
    chunks = [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="First context.",
            score=0.7,
            source_path="first.py",
            chunk_index=0,
        ),
        make_scored_chunk(
            chunk_id="chunk-2",
            content="Second context.",
            score=0.9,
            source_path="second.py",
            chunk_index=1,
        ),
    ]

    result = build_grounded_answer(
        query="Exact query text",
        scored_chunks=chunks,
        min_score=0.5,
        generator=generator,
    )

    assert result is generated_answer
    assert generator.calls == [
        {
            "query": "Exact query text",
            "context": chunks,
        }
    ]
    assert generator.calls[0]["context"] is chunks


def test_build_grounded_answer_returns_generator_result_unchanged():
    generated_answer = GroundedAnswer(
        answer="Provider-specific answer. [1]",
        citations=[
            Citation(
                citation_id=1,
                source_path="provider.py",
                start_line=40,
                end_line=50,
            )
        ],
        confidence=0.31,
        refusal_reason=None,
    )
    generator = RecordingGroundedAnswerGenerator(
        generated_answer
    )
    chunks = [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="Supporting context.",
            score=0.95,
        ),
    ]

    result = build_grounded_answer(
        query="Question",
        scored_chunks=chunks,
        generator=generator,
    )

    assert result is generated_answer
    assert result.confidence == 0.31
    assert result.citations[0].source_path == "provider.py"


def test_build_grounded_answer_uses_default_deterministic_generator():
    chunks = [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="Lower-scoring context.",
            score=0.4,
            source_path="first.py",
            chunk_index=0,
        ),
        make_scored_chunk(
            chunk_id="chunk-2",
            content="The endpoint is POST /predict.",
            score=0.9,
            source_path="api.py",
            start_line=10,
            end_line=20,
            chunk_index=1,
        ),
    ]

    result = build_grounded_answer(
        query="Where is the endpoint?",
        scored_chunks=chunks,
        min_score=0.0,
    )

    assert result.answer == (
        "The retrieved context says: "
        "The endpoint is POST /predict. [2]"
    )
    assert result.confidence == pytest.approx(0.9)
    assert result.refusal_reason is None
    assert len(result.citations) == 1

    citation = result.citations[0]

    assert citation.citation_id == 2
    assert citation.source_path == "api.py"
    assert citation.start_line == 10
    assert citation.end_line == 20