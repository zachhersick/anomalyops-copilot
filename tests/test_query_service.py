import pytest

from copilot.api.query_service import query_service
from copilot.ingestion.manifest import write_chunk_manifest
from copilot.schemas.chunk import SourceChunk
from copilot.schemas.query import QueryRequest, QueryResponse


def test_service_returns_grounded_answer_with_citations(tmp_path):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk(
            "chunk-1",
            "The prediction API exposes a POST /predict endpoint.",
            source_path="api.py",
            start_line=10,
            end_line=20,
        ),
    ]
    write_chunk_manifest(chunks, manifest_path)
    
    query_response = query_service(
        manifest_path,
        make_query_request(query="prediction api"),
    )
    
    assert isinstance(query_response, QueryResponse)
    assert query_response.answer == "I found relevant project context for this question."
    assert isinstance(query_response.confidence, float)
    assert query_response.refusal_reason is None
    assert len(query_response.citations) == 1
    assert query_response.citations[0].citation_id == 1
    assert query_response.citations[0].source_path == "api.py"
    assert query_response.citations[0].start_line == 10
    assert query_response.citations[0].end_line == 20
    
    
def test_service_respects_min_score_refusal(tmp_path):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk(
            "chunk-1",
            "The prediction API exposes a POST /predict endpoint.",
            source_path="api.py",
            start_line=10,
            end_line=20,
        ),
    ]
    write_chunk_manifest(chunks, manifest_path)
    
    query_response = query_service(
        manifest_path,
        make_query_request(query="prediction api", min_score=1.1),
    )
    
    assert query_response.answer == ""
    assert query_response.citations == []
    assert query_response.confidence == 0.0
    assert (
        query_response.refusal_reason
        == "Retrieved context was below the confidence threshold."
    )
    
    
def test_service_handles_missing_manifest_path_in_same_way():
    with pytest.raises(ValueError, match="Manifest path is not configured."):
        query_service(
            None,
            make_query_request(query="prediction api"),
        )
        
        
def test_service_show_context_equals_true(tmp_path):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk(
            "chunk-1",
            "The prediction API exposes a POST /predict endpoint.",
            source_path="api.py",
            start_line=10,
            end_line=20,
        ),
    ]
    write_chunk_manifest(chunks, manifest_path)
    
    query_response = query_service(
        manifest_path,
        make_query_request(query="prediction api", show_context=True),
    )
    
    assert query_response.context is not None
    assert "[1]" in query_response.context
    assert "api.py:10-20" in query_response.context
    assert "The prediction API exposes a POST /predict endpoint." in query_response.context
    
    
def test_service_show_context_equals_false(tmp_path):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk(
            "chunk-1",
            "The prediction API exposes a POST /predict endpoint.",
            source_path="api.py",
            start_line=10,
            end_line=20,
        ),
    ]
    write_chunk_manifest(chunks, manifest_path)
    
    query_response = query_service(
        manifest_path,
        make_query_request(query="prediction api", show_context=False),
    )
    
    assert query_response.context is None
    
    
def make_query_request(
    query: str,
    top_k: int = 3,
    min_score: float = 0.0,
    show_context: bool = False
) -> QueryRequest:
    return QueryRequest(
        query=query,
        top_k=top_k,
        min_score=min_score,
        show_context=show_context,
    )
    
    
def make_chunk(
    chunk_id: str,
    content: str,
    source_path: str = "source.py",
    start_line: int = 1,
    end_line: int = 2,
) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        source_id=source_path,
        project_name="test-project",
        source_type="python",
        source_path=source_path,
        chunk_index=0,
        content=content,
        start_line=start_line,
        end_line=end_line,
    )