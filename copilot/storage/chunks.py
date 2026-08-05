from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.dialects.postgresql.dml import Insert
from sqlalchemy.orm import Session

from copilot.providers.errors import (
    EmbeddingConfigurationError,
    InvalidEmbeddingResponseError,
)
from copilot.providers.interfaces import EmbeddingProvider
from copilot.schemas.chunk import SourceChunk
from copilot.storage.models import (
    EMBEDDING_DIMENSIONS,
    SourceChunkRecord,
)


def source_chunk_to_values(
    chunk: SourceChunk,
    embedding: list[float],
    embedding_provider: EmbeddingProvider,
) -> dict[str, object]:
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
        "embedding_provider": embedding_provider.provider_name,
        "embedding_model": embedding_provider.model_name,
        "embedding_dimensions": embedding_provider.dimensions,
        "embedding": list(embedding),
    }


def build_source_chunk_upsert_statement(
    chunks: list[SourceChunk],
    embeddings: list[list[float]],
    embedding_provider: EmbeddingProvider,
) -> Insert:
    if len(chunks) != len(embeddings):
        raise InvalidEmbeddingResponseError(
            "Embedding count does not match chunk count."
        )

    values = [
        source_chunk_to_values(
            chunk,
            embedding,
            embedding_provider,
        )
        for chunk, embedding in zip(
            chunks,
            embeddings,
        )
    ]

    statement = insert(
        SourceChunkRecord
    ).values(values)

    statement = statement.on_conflict_do_update(
        index_elements=[
            SourceChunkRecord.chunk_id
        ],
        set_={
            "source_id": (
                statement.excluded.source_id
            ),
            "project_name": (
                statement.excluded.project_name
            ),
            "source_type": (
                statement.excluded.source_type
            ),
            "source_path": (
                statement.excluded.source_path
            ),
            "chunk_index": (
                statement.excluded.chunk_index
            ),
            "content": (
                statement.excluded.content
            ),
            "start_line": (
                statement.excluded.start_line
            ),
            "end_line": (
                statement.excluded.end_line
            ),
            "embedding_provider": (
                statement.excluded.embedding_provider
            ),
            "embedding_model": (
                statement.excluded.embedding_model
            ),
            "embedding_dimensions": (
                statement.excluded.embedding_dimensions
            ),
            "embedding": (
                statement.excluded.embedding
            ),
        },
    )

    return statement


def embed_source_chunks(
    chunks: list[SourceChunk],
    embedding_provider: EmbeddingProvider,
) -> list[list[float]]:
    if not chunks:
        return []

    if (
        embedding_provider.dimensions
        != EMBEDDING_DIMENSIONS
    ):
        raise EmbeddingConfigurationError(
            "Embedding provider dimensions do not "
            "match storage dimensions."
        )

    embeddings = (
        embedding_provider.embed_documents(
            [
                chunk.content
                for chunk in chunks
            ]
        )
    )

    if len(embeddings) != len(chunks):
        raise InvalidEmbeddingResponseError(
            "Embedding count does not match "
            "chunk count."
        )

    if any(
        len(embedding)
        != EMBEDDING_DIMENSIONS
        for embedding in embeddings
    ):
        raise InvalidEmbeddingResponseError(
            "Document embedding dimensions do not "
            "match storage dimensions."
        )

    return [
        list(embedding)
        for embedding in embeddings
    ]


def _write_source_chunks(
    session: Session,
    chunks: list[SourceChunk],
    embeddings: list[list[float]],
    embedding_provider: EmbeddingProvider,
    *,
    replace_existing: bool,
) -> int:
    try:
        if replace_existing:
            session.execute(
                delete(SourceChunkRecord)
            )

        if chunks:
            statement = (
                build_source_chunk_upsert_statement(
                    chunks,
                    embeddings,
                    embedding_provider,
                )
            )
            session.execute(statement)

        session.commit()
    except Exception:
        session.rollback()
        raise

    return len(chunks)


def store_source_chunks(
    session: Session,
    chunks: list[SourceChunk],
    embedding_provider: EmbeddingProvider,
) -> int:
    if not chunks:
        return 0

    embeddings = embed_source_chunks(
        chunks,
        embedding_provider,
    )

    return _write_source_chunks(
        session,
        chunks,
        embeddings,
        embedding_provider,
        replace_existing=False,
    )


def replace_source_chunks(
    session: Session,
    chunks: list[SourceChunk],
    embedding_provider: EmbeddingProvider,
) -> int:
    """
    Replace the complete source-chunk index safely.

    All embeddings are generated and validated before the
    existing rows are deleted. The delete and insert then
    occur in one transaction, so a database failure rolls
    the replacement back.
    """
    embeddings = embed_source_chunks(
        chunks,
        embedding_provider,
    )

    return _write_source_chunks(
        session,
        chunks,
        embeddings,
        embedding_provider,
        replace_existing=True,
    )