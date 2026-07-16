from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from copilot.storage.models import SourceChunkRecord
from copilot.schemas.chunk import SourceChunk
from copilot.schemas.retrieval import ScoredChunk
from copilot.retrieval.embeddings import embed_text
from copilot.storage.models import EMBEDDING_DIMENSIONS


def source_chunk_record_to_chunk(
    record: SourceChunkRecord,
) -> SourceChunk:
    return SourceChunk(
        chunk_id=record.chunk_id,
        source_id=record.source_id,
        project_name=record.project_name,
        source_type=record.source_type,
        source_path=record.source_path,
        chunk_index=record.chunk_index,
        content=record.content,
        start_line=record.start_line,
        end_line=record.end_line,
    )
    
    
def build_pgvector_retrieval_statement(
    query_embedding: list[float],
    top_k: int,
) -> Select:
    distance = SourceChunkRecord.embedding.cosine_distance(
        query_embedding
    ).label("distance")
    
    statement = (
        select(SourceChunkRecord, distance)
        .order_by(distance)
        .limit(top_k)
    )
    
    return statement


def retrieve_relevant_chunks_from_pgvector(
    session: Session,
    query: str,
    top_k: int = 3,
) -> list[ScoredChunk]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    
    vector = embed_text(
        query,
        dimensions=EMBEDDING_DIMENSIONS
    )
    
    statement = build_pgvector_retrieval_statement(vector, top_k)
    
    records = session.execute(statement).all()
    
    scored_chunks = []
    
    for record in records:
        source_chunk_record = record[0]
        distance = record[1]
        
        source_chunk = source_chunk_record_to_chunk(source_chunk_record)
        
        scored_chunks.append(
            ScoredChunk(
                chunk=source_chunk,
                score=1.0-distance,
            )
        )
        
    return scored_chunks