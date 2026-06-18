from copilot.retrieval.context import format_retrieval_context
from copilot.schemas.chunk import SourceChunk
from copilot.schemas.retrieval import ScoredChunk


def test_format_retrieval_context_includes_numbered_citations():
    format_str = format_retrieval_context(make_scored_chunks())
    
    assert "[1]" in format_str
    assert "[2]" in format_str
    
    
def test_format_retrieval_context_includes_source_path_and_line_range():
    format_str = format_retrieval_context(make_scored_chunks())
    
    assert "source.py:1-2" in format_str
    
    
def test_format_retrieval_context_includes_score_4_decimals():
    format_str = format_retrieval_context(make_scored_chunks())
    
    assert "score=1.0000" in format_str
    assert "score=0.5000" in format_str
    
    
def test_format_retrieval_context_includes_chunk_content():
    format_str = format_retrieval_context(make_scored_chunks())
    
    assert "text" in format_str
    assert "other text" in format_str
    
    
def test_format_retrieval_context_returns_empty_str_when_no_chunks_provided():
    format_str = format_retrieval_context([])
    
    assert format_str == ""
    
    
def make_source_chunk(chunk_id: str, content: str) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        source_id="source.py",
        project_name="test-project",
        source_type="python",
        source_path="source.py",
        chunk_index=0,
        content=content,
        start_line=1,
        end_line=2,
    )
    

def make_scored_chunk(chunk: SourceChunk, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=chunk,
        score=score,
    )
    
    
def make_scored_chunks() -> list[ScoredChunk]:
    return [
        make_scored_chunk(
            chunk=make_source_chunk(chunk_id="chunk-1", content="text"),
            score=1.0
        ),
        make_scored_chunk(
            chunk=make_source_chunk(chunk_id="chunk-2", content="other text"),
            score=0.5
        ),
    ]