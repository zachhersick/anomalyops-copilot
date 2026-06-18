import pytest

from copilot.schemas.chunk import SourceChunk
from copilot.schemas.retrieval import ScoredChunk
from copilot.retrieval.search import retrieve_relevant_chunks


def test_retrieve_relevant_chunks_returns_scored_chunks():
    chunks = [
        make_chunk("chunk-1", "text"),
        make_chunk("chunk-2", "somethjing else"),
    ]
    
    scored_chunks = retrieve_relevant_chunks("text", chunks, top_k=3)
    
    assert len(scored_chunks) == 2
    assert all(isinstance(scored_chunk, ScoredChunk) for scored_chunk in scored_chunks)
    
    
def test_retrieve_relevant_chunks_sorts_highest_score_first():
    chunks = [
        make_chunk("chunk-1", "text"),
        make_chunk("chunk-2", "somethjing else"),
    ]
    
    scored_chunks = retrieve_relevant_chunks("text", chunks, top_k=3)
    
    assert scored_chunks[0].score >= scored_chunks[1].score
    
    
def test_retrieve_relevant_chunks_respects_top_k():
    chunks = [
        make_chunk("chunk-1", "text"),
        make_chunk("chunk-2", "somethjing else"),
    ]
    
    scored_chunks = retrieve_relevant_chunks("text", chunks, 1)
    
    assert len(scored_chunks) == 1
    

def test_retrieve_relevant_chunks_rejects_non_positive_top_k():
    chunks = [
        make_chunk("chunk-1", "text"),
        make_chunk("chunk-2", "somethjing else"),
    ]
    
    with pytest.raises(ValueError):
        retrieve_relevant_chunks("text", chunks, 0)
        
        
def test_retrieve_relevant_chunks_returns_empty_list_when_chunks_empty():
    score_chunks = retrieve_relevant_chunks("text", [], 3)
    
    assert score_chunks == []
    
    
def make_chunk(chunk_id: str, content: str) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        source_id="source.py",
        project_name="test-project",
        source_type="python",
        source_path="source.py",
        chunk_index=0,
        content=content,
        start_line=1,
        end_line=1,
    )