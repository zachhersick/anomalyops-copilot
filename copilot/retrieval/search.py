from copilot.schemas.retrieval import ScoredChunk
from copilot.schemas.chunk import SourceChunk
from copilot.retrieval.embeddings import embed_text
from copilot.retrieval.similarity import cosine_similarity


def retrieve_relevant_chunks(
    query: str,
    chunks: list[SourceChunk],
    top_k: int = 3,
) -> list[ScoredChunk]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    
    if not chunks:
        return []
    
    scored_chunks = []
    
    query_vector = embed_text(query)
    
    for chunk in chunks:
        chunk_vector = embed_text(chunk.content)
        score = cosine_similarity(query_vector, chunk_vector)
        scored_chunks.append(ScoredChunk(chunk=chunk, score=score))
    
    scored_chunks.sort(key=lambda scored_chunk: scored_chunk.score, reverse=True)
    
    selected_chunks = scored_chunks[:top_k]
    
    return selected_chunks
    
    