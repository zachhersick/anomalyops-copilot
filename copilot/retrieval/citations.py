from copilot.schemas.retrieval import ScoredChunk
from copilot.schemas.answer import Citation


def build_citations_from_scored_chunks(scored_chunks: list[ScoredChunk]) -> list[Citation]:
    citations = []
    
    for i, scored_chunk in enumerate(scored_chunks, start=1):
        chunk = scored_chunk.chunk
        citation = Citation(
            citation_id=i,
            source_path=chunk.source_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
        )
        citations.append(citation)
        
    return citations