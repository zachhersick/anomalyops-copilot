import pytest

from sqlalchemy.dialects import postgresql
from unittest.mock import MagicMock

from copilot.storage.chunks import (
    source_chunk_to_values,
    build_source_chunk_upsert_statement,
    store_source_chunks,
)
from copilot.schemas.chunk import SourceChunk
from copilot.retrieval.embeddings import embed_text
from copilot.storage.models import EMBEDDING_DIMENSIONS


def test_source_chunk_to_values_every_chunk_field_is_copied():
    values_dict = source_chunk_to_values(make_chunk("chunk-1", "text"))
    
    assert len(values_dict) == 10
    assert values_dict["chunk_id"] == "chunk-1"
    assert values_dict["source_id"] == "source.py"
    assert values_dict["project_name"] == "test-project"
    assert values_dict["source_type"] == "python"
    assert values_dict["source_path"] == "source.py"
    assert values_dict["chunk_index"] == 0
    assert values_dict["content"] == "text"
    assert values_dict["start_line"] == 1
    assert values_dict["end_line"] == 2
    assert values_dict["embedding"] == embed_text("text", EMBEDDING_DIMENSIONS)
    
    
def test_source_chunk_to_values_embedding_contains_16_values():
    values_dict = source_chunk_to_values(make_chunk("chunk-1", "text"))
    
    assert len(values_dict["embedding"]) == EMBEDDING_DIMENSIONS
    
    
def test_build_source_chunk_upsert_statement_uses_chunk_id_conflict():
    chunks = [
        make_chunk("chunk-1", "text"),
    ]
    
    statement = build_source_chunk_upsert_statement(chunks)
    
    compiled_sql = str(
        statement.compile(
            dialect=postgresql.dialect(),
        )
    )
    
    assert "ON CONFLICT (chunk_id) DO UPDATE" in compiled_sql
    
    
def test_store_source_chunks_executes_commits_and_returns_count():
    session = MagicMock()
    
    chunks = [
        make_chunk("chunk-1", "first"),
        make_chunk("chunk-2", "second"),
    ]
    
    stored_count = store_source_chunks(session, chunks)
    
    assert stored_count == 2
    session.execute.assert_called_once()
    session.commit.assert_called_once_with()
    session.rollback.assert_not_called()
    
    
def test_store_source_chunks_returns_zero_for_empty_chunks():
    session = MagicMock()
    
    stored_count = store_source_chunks(session, [])
    
    assert stored_count == 0
    session.execute.assert_not_called()
    session.commit.assert_not_called()
    session.rollback.assert_not_called()
    
    
def test_store_source_chunks_rolls_back_and_reraises_on_failure():
    session = MagicMock()
    session.execute.side_effect = RuntimeError("database failed")
    
    chunks = [
        make_chunk("chunk-1", "text"),
    ]

    with pytest.raises(RuntimeError, match="database failed"):
        store_source_chunks(session, chunks)
        
    session.execute.assert_called_once()
    session.commit.assert_not_called()
    session.rollback.assert_called_once_with()


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