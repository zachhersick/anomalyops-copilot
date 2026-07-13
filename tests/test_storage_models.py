from pgvector.sqlalchemy import VECTOR

from copilot.storage.models import(
    EMBEDDING_DIMENSIONS,
    SourceChunkRecord,
)


def test_source_chunk_record_uses_expected_table_name():
    table = SourceChunkRecord.__table__
    
    assert table.name == "source_chunks"
    
    
def test_source_chunk_record_has_expected_columns():
    table = SourceChunkRecord.__table__

    expected_columns = {
        "id",
        "chunk_id",
        "source_id",
        "project_name",
        "source_type",
        "source_path",
        "chunk_index",
        "content",
        "start_line",
        "end_line",
        "embedding",
    }

    assert set(table.columns.keys()) == expected_columns


def test_source_chunk_record_requires_chunk_fields():
    table = SourceChunkRecord.__table__

    required_columns = {
        "chunk_id",
        "source_id",
        "project_name",
        "source_type",
        "source_path",
        "chunk_index",
        "content",
        "start_line",
        "end_line",
        "embedding",
    }

    assert all(
        table.columns[column_name].nullable is False
        for column_name in required_columns
    )


def test_source_chunk_record_chunk_id_is_unique():
    table = SourceChunkRecord.__table__

    assert table.columns["chunk_id"].unique is True


def test_source_chunk_record_embedding_is_vector_16():
    embedding_type = SourceChunkRecord.__table__.columns["embedding"].type

    assert isinstance(embedding_type, VECTOR)
    assert embedding_type.dim == EMBEDDING_DIMENSIONS
    assert EMBEDDING_DIMENSIONS == 16