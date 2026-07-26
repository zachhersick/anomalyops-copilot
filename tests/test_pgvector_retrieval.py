from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.dialects import postgresql

from copilot.providers.errors import EmbeddingConfigurationError
from copilot.retrieval.pgvector import (
    build_pgvector_retrieval_statement,
    retrieve_relevant_chunks_from_pgvector,
    source_chunk_record_to_chunk,
)
from copilot.storage.models import (
    EMBEDDING_DIMENSIONS,
    SourceChunkRecord,
)


class FakeEmbeddingProvider:
    provider_name = "fake"
    model_name = "fake-embedding"

    def __init__(
        self,
        query_embedding: list[float],
    ) -> None:
        self.query_embedding = query_embedding
        self.dimensions = len(query_embedding)
        self.queries: list[str] = []

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        self.queries.append(text)
        return list(self.query_embedding)

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        raise NotImplementedError


def make_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(
        query_embedding=[0.25] * EMBEDDING_DIMENSIONS,
    )


def make_record(
    chunk_id: str = "chunk-1",
    source_path: str = "source.py",
    content: str = "example content",
) -> SourceChunkRecord:
    return SourceChunkRecord(
        chunk_id=chunk_id,
        source_id=source_path,
        project_name="test-project",
        source_type="python",
        source_path=source_path,
        chunk_index=0,
        content=content,
        start_line=1,
        end_line=5,
        embedding=[0.0] * EMBEDDING_DIMENSIONS,
    )


def test_source_chunk_record_to_chunk_copies_expected_fields():
    record = make_record()

    chunk = source_chunk_record_to_chunk(record)

    assert chunk.chunk_id == "chunk-1"
    assert chunk.source_id == "source.py"
    assert chunk.project_name == "test-project"
    assert chunk.source_type == "python"
    assert chunk.source_path == "source.py"
    assert chunk.chunk_index == 0
    assert chunk.content == "example content"
    assert chunk.start_line == 1
    assert chunk.end_line == 5


def test_build_pgvector_retrieval_statement_uses_cosine_distance():
    query_embedding = [0.0] * EMBEDDING_DIMENSIONS

    statement = build_pgvector_retrieval_statement(
        query_embedding,
        top_k=3,
    )

    compiled_sql = str(
        statement.compile(
            dialect=postgresql.dialect(),
        )
    )

    assert "<=>" in compiled_sql
    assert "ORDER BY" in compiled_sql
    assert "LIMIT" in compiled_sql


@pytest.mark.parametrize("top_k", [0, -1])
def test_retrieve_relevant_chunks_rejects_nonpositive_top_k(
    top_k,
):
    session = MagicMock()
    provider = make_provider()

    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        retrieve_relevant_chunks_from_pgvector(
            session=session,
            query="query",
            embedding_provider=provider,
            top_k=top_k,
        )

    assert provider.queries == []
    session.execute.assert_not_called()


def test_retrieve_relevant_chunks_rejects_provider_dimension_mismatch():
    session = MagicMock()
    provider = FakeEmbeddingProvider(
        query_embedding=[0.1, 0.2, 0.3],
    )

    with pytest.raises(
        EmbeddingConfigurationError,
        match=(
            "Embedding provider dimensions do not match "
            "storage dimensions"
        ),
    ):
        retrieve_relevant_chunks_from_pgvector(
            session=session,
            query="query",
            embedding_provider=provider,
            top_k=3,
        )

    assert provider.queries == []
    session.execute.assert_not_called()


def test_retrieve_relevant_chunks_embeds_query_and_executes_once():
    session = MagicMock()
    statement = MagicMock()
    query_embedding = [0.25] * EMBEDDING_DIMENSIONS
    provider = FakeEmbeddingProvider(query_embedding)

    session.execute.return_value.all.return_value = []

    with patch(
        "copilot.retrieval.pgvector.build_pgvector_retrieval_statement",
        return_value=statement,
    ) as build_statement:
        results = retrieve_relevant_chunks_from_pgvector(
            session=session,
            query="anomaly types",
            embedding_provider=provider,
            top_k=4,
        )

    assert results == []
    assert provider.queries == ["anomaly types"]

    build_statement.assert_called_once_with(
        query_embedding,
        4,
    )
    session.execute.assert_called_once_with(statement)


def test_retrieve_relevant_chunks_converts_rows_to_scored_chunks():
    session = MagicMock()
    provider = make_provider()

    first_record = make_record(
        chunk_id="chunk-1",
        source_path="first.py",
        content="first",
    )
    second_record = make_record(
        chunk_id="chunk-2",
        source_path="second.py",
        content="second",
    )

    session.execute.return_value.all.return_value = [
        (first_record, 0.2),
        (second_record, 0.5),
    ]

    results = retrieve_relevant_chunks_from_pgvector(
        session=session,
        query="query",
        embedding_provider=provider,
        top_k=2,
    )

    assert len(results) == 2

    assert results[0].chunk.chunk_id == "chunk-1"
    assert results[0].chunk.source_path == "first.py"
    assert results[0].score == pytest.approx(0.8)

    assert results[1].chunk.chunk_id == "chunk-2"
    assert results[1].chunk.source_path == "second.py"
    assert results[1].score == pytest.approx(0.5)

    assert provider.queries == ["query"]
    session.execute.assert_called_once()


def test_retrieve_relevant_chunks_returns_empty_list_for_no_rows():
    session = MagicMock()
    provider = make_provider()
    session.execute.return_value.all.return_value = []

    results = retrieve_relevant_chunks_from_pgvector(
        session=session,
        query="query",
        embedding_provider=provider,
        top_k=3,
    )

    assert results == []
    assert provider.queries == ["query"]
    session.execute.assert_called_once()