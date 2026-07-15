from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.dialects.postgresql.dml import Insert

from copilot.schemas.chunk import SourceChunk
from copilot.storage.models import EMBEDDING_DIMENSIONS
from copilot.retrieval.embeddings import embed_text
from copilot.storage.models import SourceChunkRecord


def source_chunk_to_values(chunk: SourceChunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "project_name": chunk.project_name,
        "source_type": chunk.source_type,
        "source_path": chunk.source_path,
        "chunk_index": chunk.chunk_index,
        "content": chunk.content,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "embedding": embed_text(
            chunk.content,
            EMBEDDING_DIMENSIONS,
        ),
    }
    
    
def build_source_chunk_upsert_statement(chunks: list[SourceChunk]) -> Insert:
    values = [
        source_chunk_to_values(chunk)
        for chunk in chunks
    ]
    
    statement = insert(SourceChunkRecord).values(values)
    
    statement = statement.on_conflict_do_update(
        index_elements=[SourceChunkRecord.chunk_id],
        set_={
            "source_id": statement.excluded.source_id,
            "project_name": statement.excluded.project_name,
            "source_type": statement.excluded.source_type,
            "source_path": statement.excluded.source_path,
            "chunk_index": statement.excluded.chunk_index,
            "content": statement.excluded.content,
            "start_line": statement.excluded.start_line,
            "end_line": statement.excluded.end_line,
            "embedding": statement.excluded.embedding,
        }
    )
    
    return statement