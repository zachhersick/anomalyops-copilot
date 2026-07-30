import pytest

from copilot.providers.deterministic_answers import (
    DeterministicGroundedAnswerGenerator,
)
from copilot.providers.errors import (
    GroundedAnswerConfigurationError,
)
from copilot.providers.interfaces import GroundedAnswerGenerator
from copilot.schemas.chunk import SourceChunk
from copilot.schemas.retrieval import ScoredChunk


def make_scored_chunk(
    chunk_id: str,
    content: str,
    score: float,
    source_path: str = "source.py",
    start_line: int = 1,
    end_line: int = 5,
) -> ScoredChunk:
    return ScoredChunk(
        chunk=SourceChunk(
            chunk_id=chunk_id,
            source_id=source_path,
            project_name="test-project",
            source_type="python",
            source_path=source_path,
            chunk_index=0,
            content=content,
            start_line=start_line,
            end_line=end_line,
        ),
        score=score,
    )


def test_provider_satisfies_grounded_answer_generator_protocol():
    provider = DeterministicGroundedAnswerGenerator()

    assert isinstance(provider, GroundedAnswerGenerator)


def test_provider_exposes_expected_metadata():
    provider = DeterministicGroundedAnswerGenerator()

    assert provider.provider_name == "deterministic"
    assert provider.model_name == "retrieval-context-template-v1"


def test_generate_selects_highest_scoring_chunk():
    provider = DeterministicGroundedAnswerGenerator()
    context = [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="Lower-scoring content.",
            score=0.4,
        ),
        make_scored_chunk(
            chunk_id="chunk-2",
            content="Highest-scoring content.",
            score=0.9,
        ),
        make_scored_chunk(
            chunk_id="chunk-3",
            content="Other content.",
            score=0.6,
        ),
    ]

    result = provider.generate(
        query="What is the answer?",
        context=context,
    )

    assert result.answer == (
        "The retrieved context says: "
        "Highest-scoring content. [2]"
    )
    assert result.confidence == pytest.approx(0.9)
    assert result.refusal_reason is None


def test_generate_preserves_original_context_position_as_citation_id():
    provider = DeterministicGroundedAnswerGenerator()
    context = [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="First content.",
            score=0.5,
        ),
        make_scored_chunk(
            chunk_id="chunk-2",
            content="Second content.",
            score=0.8,
            source_path="api.py",
            start_line=10,
            end_line=20,
        ),
    ]

    result = provider.generate(
        query="Where is the endpoint?",
        context=context,
    )

    assert len(result.citations) == 1

    citation = result.citations[0]

    assert citation.citation_id == 2
    assert citation.source_path == "api.py"
    assert citation.start_line == 10
    assert citation.end_line == 20
    assert "[2]" in result.answer


def test_generate_clamps_confidence_above_one():
    provider = DeterministicGroundedAnswerGenerator()
    context = [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="Content.",
            score=1.4,
        ),
    ]

    result = provider.generate(
        query="Question",
        context=context,
    )

    assert result.confidence == 1.0


def test_generate_clamps_confidence_below_zero():
    provider = DeterministicGroundedAnswerGenerator()
    context = [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="Content.",
            score=-0.4,
        ),
    ]

    result = provider.generate(
        query="Question",
        context=context,
    )

    assert result.confidence == 0.0


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "\n\t",
    ],
)
def test_generate_rejects_blank_query(query):
    provider = DeterministicGroundedAnswerGenerator()
    context = [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="Content.",
            score=0.8,
        ),
    ]

    with pytest.raises(
        GroundedAnswerConfigurationError,
        match="query cannot be blank or whitespace-only.",
    ):
        provider.generate(
            query=query,
            context=context,
        )


def test_generate_rejects_empty_context():
    provider = DeterministicGroundedAnswerGenerator()

    with pytest.raises(
        GroundedAnswerConfigurationError,
        match="context cannot be empty.",
    ):
        provider.generate(
            query="Question",
            context=[],
        )