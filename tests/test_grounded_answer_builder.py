from copilot.answering.grounded import build_grounded_answer
from copilot.schemas.retrieval import ScoredChunk
from copilot.schemas.answer import GroundedAnswer, Citation
from copilot.schemas.chunk import SourceChunk


def test_build_grounded_answer_builds_answer_with_citations():
    scored_chunks  = [
        make_scored_chunk(chunk_id="chunk-1"),
        make_scored_chunk(chunk_id="chunk-2"),
    ]
    query = "text"
    grounded_answer = build_grounded_answer(query, scored_chunks)
    
    assert isinstance(grounded_answer, GroundedAnswer)
    assert len(grounded_answer.citations) == 2
    assert all(isinstance(citation, Citation) for citation in grounded_answer.citations)
    
    
def test_build_grounded_answer_mentions_retrieved_context_exists():
    scored_chunks  = [
        make_scored_chunk(chunk_id="chunk-1"),
        make_scored_chunk(chunk_id="chunk-2"),
    ]
    query = "text"
    grounded_answer = build_grounded_answer(query, scored_chunks)
    
    assert "The retrieved context says" in grounded_answer.answer
    
    
def test_build_grounded_answer_confidence_comes_from_top_score():
    scored_chunks  = [
        make_scored_chunk(
            chunk_id="chunk-1",
            source_path="source.py",
            start_line=1,
            end_line=2,
            score=0.75
        ),
        make_scored_chunk(
            chunk_id="chunk-2",
            source_path="source.py",
            start_line=1,
            end_line=2,
            score=0.90
        ),
    ]
    query = "text"
    grounded_answer = build_grounded_answer(query, scored_chunks)
    
    assert grounded_answer.confidence == 0.90
    
    
def test_build_grounded_answer_clamps_confidence_greater_than_one_to_one():
    scored_chunks  = [
        make_scored_chunk(score=1.10),
    ]
    query = "text"
    grounded_answer = build_grounded_answer(query, scored_chunks)
    
    assert grounded_answer.confidence == 1.0
    
    
def test_build_grounded_answer_clamps_negative_confidence_to_zero():
    scored_chunks = [
        make_scored_chunk(score=-0.25),
    ]

    grounded_answer = build_grounded_answer("text", scored_chunks)

    assert grounded_answer.confidence == 0.0
    
    
def test_build_grounded_answer_refuses_when_no_scored_chunks_exist():
    grounded_answer = build_grounded_answer("text", [])
    
    assert grounded_answer.answer == ""
    assert grounded_answer.citations == []
    assert grounded_answer.confidence == 0.0
    assert grounded_answer.refusal_reason == "No relevant context was retrieved."
    
    
def test_build_grounded_answer_refuses_top_score_less_than_min_score():
    grounded_answer = build_grounded_answer("text", [make_scored_chunk(score=0.25)], 0.5)
    
    assert grounded_answer.answer == ""
    assert grounded_answer.citations == []
    assert grounded_answer.confidence == 0.0
    assert (
        grounded_answer.refusal_reason
        == "Retrieved context was below the confidence threshold."
    )
    
    
def test_build_grounded_answer_accepts_top_score_greater_than_or_equal_to_min_score():
    grounded_answer = build_grounded_answer(
        "text",
        [make_scored_chunk(score=0.75)],
        min_score=0.5,
    )

    assert grounded_answer.answer == "The retrieved context says: chunk content"
    assert grounded_answer.refusal_reason is None
    
    
def test_build_grounded_answer_returns_deterministic_answer():
    scored_chunks  = [
        make_scored_chunk(
            chunk_id="chunk-1",
            source_path="source.py",
            start_line=1,
            end_line=2,
            score=0.90,
        ),
        make_scored_chunk(
            chunk_id="chunk-2",
            source_path="source.py",
            start_line=3,
            end_line=4,
            score=0.5,
        )
    ]
    query = "chunk"
    grounded_answer = build_grounded_answer(query, scored_chunks)
    
    assert grounded_answer.answer == "The retrieved context says: chunk content"
    
    
def make_scored_chunk(
    chunk_id: str = "chunk-1",
    source_path: str = "source.py",
    start_line: int = 1,
    end_line: int = 2,
    score: float = 0.75,
) -> ScoredChunk:
    source_chunk = SourceChunk(
        chunk_id=chunk_id,
        source_id=source_path,
        project_name="test-project",
        source_type="python",
        source_path=source_path,
        chunk_index=0,
        content="chunk content",
        start_line=start_line,
        end_line=end_line,
    )

    return ScoredChunk(
        chunk=source_chunk,
        score=score,
    )