from collections.abc import Sequence
from unittest.mock import (
    MagicMock,
    patch,
    call,
)

import pytest
from sqlalchemy.dialects import postgresql

from copilot.providers.errors import (
    EmbeddingConfigurationError,
    InvalidEmbeddingResponseError,
)
from copilot.schemas.chunk import SourceChunk
from copilot.storage.chunks import (
    build_source_chunk_upsert_statement,
    source_chunk_to_values,
    store_source_chunks,
    embed_source_chunks,
    replace_source_chunks,
)
from copilot.storage.models import EMBEDDING_DIMENSIONS


class FakeEmbeddingProvider:
    provider_name = "fake"
    model_name = "fake-embedding"

    def __init__(
        self,
        embeddings: list[list[float]],
        dimensions: int = EMBEDDING_DIMENSIONS,
    ) -> None:
        self.dimensions = dimensions
        self.embeddings = embeddings
        self.document_calls: list[list[str]] = []

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        raise NotImplementedError

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [
            list(embedding)
            for embedding in self.embeddings
        ]


def make_embedding(
    value: float = 0.25,
) -> list[float]:
    return [value] * EMBEDDING_DIMENSIONS


def test_source_chunk_to_values_copies_fields_and_given_embedding():
    embedding = make_embedding()
    provider = FakeEmbeddingProvider([embedding])

    values = source_chunk_to_values(
        make_chunk("chunk-1", "text"),
        embedding,
        provider,
    )

    assert len(values) == 13
    assert values["chunk_id"] == "chunk-1"
    assert values["source_id"] == "source.py"
    assert values["project_name"] == "test-project"
    assert values["source_type"] == "python"
    assert values["source_path"] == "source.py"
    assert values["chunk_index"] == 0
    assert values["content"] == "text"
    assert values["start_line"] == 1
    assert values["end_line"] == 2
    assert values["embedding_provider"] == "fake"
    assert values["embedding_model"] == "fake-embedding"
    assert values["embedding_dimensions"] == EMBEDDING_DIMENSIONS
    assert values["embedding"] == embedding


def test_build_source_chunk_upsert_statement_uses_chunk_id_conflict():
    chunks = [
        make_chunk("chunk-1", "text"),
    ]
    embeddings = [
        make_embedding(),
    ]

    provider = FakeEmbeddingProvider(embeddings)

    statement = build_source_chunk_upsert_statement(
        chunks,
        embeddings,
        provider,
    )

    compiled_sql = str(
        statement.compile(
            dialect=postgresql.dialect(),
        )
    )

    assert "ON CONFLICT (chunk_id) DO UPDATE" in compiled_sql


def test_build_source_chunk_upsert_statement_rejects_count_mismatch():
    chunks = [
        make_chunk("chunk-1", "first"),
        make_chunk("chunk-2", "second"),
    ]
    embeddings = [
        make_embedding(),
    ]
    
    provider = FakeEmbeddingProvider(embeddings)

    with pytest.raises(
        InvalidEmbeddingResponseError,
        match="Embedding count does not match chunk count.",
    ):
        build_source_chunk_upsert_statement(
            chunks,
            embeddings,
            provider,
        )


def test_store_source_chunks_batches_documents_and_commits():
    session = MagicMock()
    statement = MagicMock()

    chunks = [
        make_chunk("chunk-1", "first"),
        make_chunk("chunk-2", "second"),
    ]
    embeddings = [
        make_embedding(0.25),
        make_embedding(0.5),
    ]
    provider = FakeEmbeddingProvider(embeddings)

    with patch(
        "copilot.storage.chunks.build_source_chunk_upsert_statement",
        return_value=statement,
    ) as build_statement:
        stored_count = store_source_chunks(
            session=session,
            chunks=chunks,
            embedding_provider=provider,
        )

    assert stored_count == 2
    assert provider.document_calls == [["first", "second"]]

    build_statement.assert_called_once_with(
        chunks,
        embeddings,
        provider,
    )
    session.execute.assert_called_once_with(statement)
    session.commit.assert_called_once_with()
    session.rollback.assert_not_called()


def test_store_source_chunks_returns_zero_for_empty_chunks():
    session = MagicMock()
    provider = FakeEmbeddingProvider([])

    stored_count = store_source_chunks(
        session=session,
        chunks=[],
        embedding_provider=provider,
    )

    assert stored_count == 0
    assert provider.document_calls == []
    session.execute.assert_not_called()
    session.commit.assert_not_called()
    session.rollback.assert_not_called()


def test_store_source_chunks_rejects_provider_dimension_mismatch():
    session = MagicMock()
    provider = FakeEmbeddingProvider(
        embeddings=[],
        dimensions=3,
    )

    with pytest.raises(
        EmbeddingConfigurationError,
        match=(
            "Embedding provider dimensions do not match "
            "storage dimensions."
        ),
    ):
        store_source_chunks(
            session=session,
            chunks=[make_chunk("chunk-1", "text")],
            embedding_provider=provider,
        )

    assert provider.document_calls == []
    session.execute.assert_not_called()
    session.commit.assert_not_called()
    session.rollback.assert_not_called()


def test_store_source_chunks_rejects_embedding_count_mismatch():
    session = MagicMock()
    provider = FakeEmbeddingProvider(
        embeddings=[
            make_embedding(),
        ],
    )
    chunks = [
        make_chunk("chunk-1", "first"),
        make_chunk("chunk-2", "second"),
    ]

    with pytest.raises(
        InvalidEmbeddingResponseError,
        match="Embedding count does not match chunk count.",
    ):
        store_source_chunks(
            session=session,
            chunks=chunks,
            embedding_provider=provider,
        )

    assert provider.document_calls == [["first", "second"]]
    session.execute.assert_not_called()
    session.commit.assert_not_called()
    session.rollback.assert_not_called()


def test_store_source_chunks_rejects_invalid_vector_dimensions():
    session = MagicMock()
    provider = FakeEmbeddingProvider(
        embeddings=[
            [0.1, 0.2, 0.3],
        ],
        dimensions=EMBEDDING_DIMENSIONS,
    )

    with pytest.raises(
        InvalidEmbeddingResponseError,
        match=(
            "Document embedding dimensions do not match "
            "storage dimensions."
        ),
    ):
        store_source_chunks(
            session=session,
            chunks=[make_chunk("chunk-1", "text")],
            embedding_provider=provider,
        )

    session.execute.assert_not_called()
    session.commit.assert_not_called()
    session.rollback.assert_not_called()


def test_store_source_chunks_rolls_back_and_reraises_on_failure():
    session = MagicMock()
    session.execute.side_effect = RuntimeError(
        "database failed"
    )
    provider = FakeEmbeddingProvider(
        embeddings=[
            make_embedding(),
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="database failed",
    ):
        store_source_chunks(
            session=session,
            chunks=[make_chunk("chunk-1", "text")],
            embedding_provider=provider,
        )

    assert provider.document_calls == [["text"]]
    session.execute.assert_called_once()
    session.commit.assert_not_called()
    session.rollback.assert_called_once_with()
    
    
def test_embed_source_chunks_returns_validated_embeddings():
    embeddings = [
        make_embedding(0.25),
        make_embedding(0.5),
    ]
    provider = FakeEmbeddingProvider(
        embeddings
    )
    chunks = [
        make_chunk(
            "chunk-1",
            "first",
        ),
        make_chunk(
            "chunk-2",
            "second",
        ),
    ]

    result = embed_source_chunks(
        chunks,
        provider,
    )

    assert result == embeddings
    assert provider.document_calls == [
        [
            "first",
            "second",
        ]
    ]


def test_replace_source_chunks_embeds_before_deleting():
    session = MagicMock()
    delete_statement = MagicMock()
    upsert_statement = MagicMock()

    chunks = [
        make_chunk(
            "chunk-1",
            "first",
        ),
        make_chunk(
            "chunk-2",
            "second",
        ),
    ]
    embeddings = [
        make_embedding(0.25),
        make_embedding(0.5),
    ]
    provider = FakeEmbeddingProvider(
        embeddings
    )

    events: list[str] = []

    original_embed_documents = (
        provider.embed_documents
    )

    def recording_embed_documents(
        texts,
    ):
        events.append("embed")
        return original_embed_documents(
            texts
        )

    provider.embed_documents = (
        recording_embed_documents
    )

    def recording_execute(
        statement,
    ):
        if statement is delete_statement:
            events.append("delete")
        elif statement is upsert_statement:
            events.append("upsert")

    session.execute.side_effect = (
        recording_execute
    )

    with (
        patch(
            "copilot.storage.chunks.delete",
            return_value=delete_statement,
        ),
        patch(
            "copilot.storage.chunks."
            "build_source_chunk_upsert_statement",
            return_value=upsert_statement,
        ),
    ):
        stored_count = (
            replace_source_chunks(
                session=session,
                chunks=chunks,
                embedding_provider=provider,
            )
        )

    assert stored_count == 2
    assert events == [
        "embed",
        "delete",
        "upsert",
    ]

    session.execute.assert_has_calls(
        [
            call(delete_statement),
            call(upsert_statement),
        ]
    )
    session.commit.assert_called_once_with()
    session.rollback.assert_not_called()


def test_replace_source_chunks_provider_failure_preserves_index():
    session = MagicMock()
    provider = MagicMock()

    provider.dimensions = (
        EMBEDDING_DIMENSIONS
    )
    provider.embed_documents.side_effect = (
        RuntimeError(
            "provider failed"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="provider failed",
    ):
        replace_source_chunks(
            session=session,
            chunks=[
                make_chunk(
                    "chunk-1",
                    "text",
                )
            ],
            embedding_provider=provider,
        )

    session.execute.assert_not_called()
    session.commit.assert_not_called()
    session.rollback.assert_not_called()


def test_replace_source_chunks_rolls_back_failed_replacement():
    session = MagicMock()
    delete_statement = MagicMock()
    upsert_statement = MagicMock()

    provider = FakeEmbeddingProvider(
        [
            make_embedding(),
        ]
    )
    chunks = [
        make_chunk(
            "chunk-1",
            "text",
        )
    ]

    session.execute.side_effect = [
        None,
        RuntimeError(
            "insert failed"
        ),
    ]

    with (
        patch(
            "copilot.storage.chunks.delete",
            return_value=delete_statement,
        ),
        patch(
            "copilot.storage.chunks."
            "build_source_chunk_upsert_statement",
            return_value=upsert_statement,
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="insert failed",
        ):
            replace_source_chunks(
                session=session,
                chunks=chunks,
                embedding_provider=provider,
            )

    session.execute.assert_has_calls(
        [
            call(delete_statement),
            call(upsert_statement),
        ]
    )
    session.commit.assert_not_called()
    session.rollback.assert_called_once_with()


def test_replace_source_chunks_empty_manifest_clears_index():
    session = MagicMock()
    delete_statement = MagicMock()
    provider = FakeEmbeddingProvider([])

    with patch(
        "copilot.storage.chunks.delete",
        return_value=delete_statement,
    ):
        stored_count = replace_source_chunks(
            session=session,
            chunks=[],
            embedding_provider=provider,
        )

    assert stored_count == 0
    assert provider.document_calls == []

    session.execute.assert_called_once_with(
        delete_statement
    )
    session.commit.assert_called_once_with()
    session.rollback.assert_not_called()


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