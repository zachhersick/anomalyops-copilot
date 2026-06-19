from copilot.retrieval.citations import build_citations_from_scored_chunks
from copilot.schemas.retrieval import ScoredChunk
from copilot.schemas.chunk import SourceChunk
from copilot.schemas.answer import Citation


def test_build_citations_returns_two_citations_from_two_scored_chunks():
    scored_chunks = [
        make_scored_chunk(chunk_id="chunk-1", source_path="api.py"),
        make_scored_chunk(chunk_id="chunk-2", source_path="model.py"),
    ]
    citations = build_citations_from_scored_chunks(scored_chunks)
    
    assert len(citations) == 2
    assert all(isinstance(citation, Citation) for citation in citations)
    
    
def test_build_citations_starts_citation_ids_at_one():
    scored_chunks = [
        make_scored_chunk(chunk_id="chunk-1", source_path="api.py"),
        make_scored_chunk(chunk_id="chunk-2", source_path="model.py"),
    ]
    citations = build_citations_from_scored_chunks(scored_chunks)
    
    assert citations[0].citation_id == 1
    assert citations[1].citation_id == 2
    
    
def test_build_citations_copies_source_path_and_line_range():
    scored_chunk = make_scored_chunk(
        chunk_id="chunk-1",
        source_path="api.py",
        start_line=10,
        end_line=20,
    )
    
    citations = build_citations_from_scored_chunks([scored_chunk])
    
    assert citations[0].source_path == "api.py"
    assert citations[0].start_line == 10
    assert citations[0].end_line == 20
    
    
def test_build_citations_returns_empty_list_for_no_scored_chunks():
    citations = build_citations_from_scored_chunks([])
    
    assert citations == []
    
    
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