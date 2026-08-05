from pathlib import Path
from unittest.mock import (
    MagicMock,
    patch,
)

import pytest

from copilot.api.settings import (
    ApiSettings,
)
from copilot.providers.errors import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
)
from scripts.reindex_embeddings import main


def test_main_safely_replaces_embeddings(
    capsys,
):
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url=(
            "postgresql+psycopg://test"
        ),
    )
    chunks = [
        MagicMock(),
        MagicMock(),
    ]
    embedding_provider = MagicMock()
    engine = MagicMock()
    session = MagicMock()
    session_factory = MagicMock()

    session_factory.return_value.__enter__.return_value = (
        session
    )

    with (
        patch(
            "scripts.reindex_embeddings."
            "load_dotenv",
        ) as load_dotenv,
        patch(
            "scripts.reindex_embeddings."
            "load_api_settings",
            return_value=settings,
        ) as load_api_settings,
        patch(
            "scripts.reindex_embeddings."
            "create_embedding_provider",
            return_value=embedding_provider,
        ) as create_embedding_provider,
        patch(
            "scripts.reindex_embeddings."
            "load_chunk_manifest",
            return_value=chunks,
        ) as load_chunk_manifest,
        patch(
            "scripts.reindex_embeddings."
            "create_engine_from_url",
            return_value=engine,
        ) as create_engine_from_url,
        patch(
            "scripts.reindex_embeddings."
            "initialize_database",
        ) as initialize_database,
        patch(
            "scripts.reindex_embeddings."
            "create_session_factory",
            return_value=session_factory,
        ) as create_session_factory,
        patch(
            "scripts.reindex_embeddings."
            "replace_source_chunks",
            return_value=2,
        ) as replace_source_chunks,
    ):
        result = main(
            [
                "outputs/chunks.json",
            ]
        )

    assert result == 0

    load_dotenv.assert_called_once_with()
    load_api_settings.assert_called_once_with()

    create_embedding_provider.assert_called_once_with(
        settings
    )
    load_chunk_manifest.assert_called_once_with(
        Path(
            "outputs/chunks.json"
        )
    )
    create_engine_from_url.assert_called_once_with(
        "postgresql+psycopg://test"
    )
    initialize_database.assert_called_once_with(
        engine
    )
    create_session_factory.assert_called_once_with(
        engine
    )
    session_factory.assert_called_once_with()

    replace_source_chunks.assert_called_once_with(
        session=session,
        chunks=chunks,
        embedding_provider=(
            embedding_provider
        ),
    )

    engine.dispose.assert_called_once_with()

    assert capsys.readouterr().out == (
        "Reindexed 2 source chunks.\n"
    )


def test_main_rejects_missing_database_url():
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url=None,
    )

    with (
        patch(
            "scripts.reindex_embeddings."
            "load_dotenv",
        ),
        patch(
            "scripts.reindex_embeddings."
            "load_api_settings",
            return_value=settings,
        ),
        patch(
            "scripts.reindex_embeddings."
            "create_embedding_provider",
        ) as create_embedding_provider,
        patch(
            "scripts.reindex_embeddings."
            "load_chunk_manifest",
        ) as load_chunk_manifest,
        patch(
            "scripts.reindex_embeddings."
            "create_engine_from_url",
        ) as create_engine_from_url,
    ):
        with pytest.raises(
            RuntimeError,
            match=(
                "ANOMALYOPS_DATABASE_URL "
                "is not configured"
            ),
        ):
            main(
                [
                    "outputs/chunks.json",
                ]
            )

    create_embedding_provider.assert_not_called()
    load_chunk_manifest.assert_not_called()
    create_engine_from_url.assert_not_called()


def test_main_propagates_embedding_configuration_error():
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url=(
            "postgresql+psycopg://test"
        ),
    )

    with (
        patch(
            "scripts.reindex_embeddings."
            "load_dotenv",
        ),
        patch(
            "scripts.reindex_embeddings."
            "load_api_settings",
            return_value=settings,
        ),
        patch(
            "scripts.reindex_embeddings."
            "create_embedding_provider",
            side_effect=(
                EmbeddingConfigurationError(
                    "Embedding configuration "
                    "is invalid."
                )
            ),
        ),
        patch(
            "scripts.reindex_embeddings."
            "load_chunk_manifest",
        ) as load_chunk_manifest,
        patch(
            "scripts.reindex_embeddings."
            "create_engine_from_url",
        ) as create_engine_from_url,
    ):
        with pytest.raises(
            EmbeddingConfigurationError,
            match=(
                "Embedding configuration "
                "is invalid."
            ),
        ):
            main(
                [
                    "outputs/chunks.json",
                ]
            )

    load_chunk_manifest.assert_not_called()
    create_engine_from_url.assert_not_called()


def test_main_explains_provider_failure_preserves_index():
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url=(
            "postgresql+psycopg://test"
        ),
    )
    engine = MagicMock()
    session = MagicMock()
    session_factory = MagicMock()

    session_factory.return_value.__enter__.return_value = (
        session
    )

    with (
        patch(
            "scripts.reindex_embeddings."
            "load_dotenv",
        ),
        patch(
            "scripts.reindex_embeddings."
            "load_api_settings",
            return_value=settings,
        ),
        patch(
            "scripts.reindex_embeddings."
            "create_embedding_provider",
            return_value=MagicMock(),
        ),
        patch(
            "scripts.reindex_embeddings."
            "load_chunk_manifest",
            return_value=[
                MagicMock(),
            ],
        ),
        patch(
            "scripts.reindex_embeddings."
            "create_engine_from_url",
            return_value=engine,
        ),
        patch(
            "scripts.reindex_embeddings."
            "initialize_database",
        ),
        patch(
            "scripts.reindex_embeddings."
            "create_session_factory",
            return_value=session_factory,
        ),
        patch(
            "scripts.reindex_embeddings."
            "replace_source_chunks",
            side_effect=EmbeddingProviderError(
                "quota unavailable"
            ),
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match=(
                "Embedding generation failed; "
                "the existing source chunk "
                "index was not modified."
            ),
        ):
            main(
                [
                    "outputs/chunks.json",
                ]
            )

    engine.dispose.assert_called_once_with()