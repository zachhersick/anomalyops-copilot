from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from copilot.api.settings import ApiSettings
from copilot.providers.errors import EmbeddingConfigurationError
from scripts.store_chunk_manifest import main


def test_main_stores_chunk_manifest_with_configured_provider(capsys):
    chunks = [
        MagicMock(),
        MagicMock(),
    ]
    engine = MagicMock()
    session = MagicMock()
    session_factory = MagicMock()
    embedding_provider = MagicMock()

    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url="postgresql+psycopg://test",
    )

    session_factory.return_value.__enter__.return_value = session

    with (
        patch(
            "scripts.store_chunk_manifest.load_dotenv",
        ) as load_dotenv,
        patch(
            "scripts.store_chunk_manifest.load_api_settings",
            return_value=settings,
        ) as load_api_settings,
        patch(
            "scripts.store_chunk_manifest.create_embedding_provider",
            return_value=embedding_provider,
        ) as create_embedding_provider,
        patch(
            "scripts.store_chunk_manifest.load_chunk_manifest",
            return_value=chunks,
        ) as load_chunk_manifest,
        patch(
            "scripts.store_chunk_manifest.create_engine_from_url",
            return_value=engine,
        ) as create_engine_from_url,
        patch(
            "scripts.store_chunk_manifest.initialize_database",
        ) as initialize_database,
        patch(
            "scripts.store_chunk_manifest.create_session_factory",
            return_value=session_factory,
        ) as create_session_factory,
        patch(
            "scripts.store_chunk_manifest.store_source_chunks",
            return_value=2,
        ) as store_source_chunks,
    ):
        result = main(["outputs/chunks.json"])

    assert result == 0

    load_dotenv.assert_called_once_with()
    load_api_settings.assert_called_once_with()

    create_embedding_provider.assert_called_once_with(settings)

    load_chunk_manifest.assert_called_once_with(
        Path("outputs/chunks.json")
    )

    create_engine_from_url.assert_called_once_with(
        "postgresql+psycopg://test"
    )
    initialize_database.assert_called_once_with(engine)
    create_session_factory.assert_called_once_with(engine)

    session_factory.assert_called_once_with()

    store_source_chunks.assert_called_once_with(
        session=session,
        chunks=chunks,
        embedding_provider=embedding_provider,
    )

    assert capsys.readouterr().out == (
        "Stored 2 source chunks.\n"
    )


def test_main_raises_when_database_url_is_missing():
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url=None,
    )

    with (
        patch(
            "scripts.store_chunk_manifest.load_dotenv",
        ) as load_dotenv,
        patch(
            "scripts.store_chunk_manifest.load_api_settings",
            return_value=settings,
        ) as load_api_settings,
        patch(
            "scripts.store_chunk_manifest.create_embedding_provider",
        ) as create_embedding_provider,
        patch(
            "scripts.store_chunk_manifest.load_chunk_manifest",
        ) as load_chunk_manifest,
        patch(
            "scripts.store_chunk_manifest.create_engine_from_url",
        ) as create_engine_from_url,
    ):
        with pytest.raises(
            RuntimeError,
            match=(
                "ANOMALYOPS_DATABASE_URL "
                "is not configured"
            ),
        ):
            main(["outputs/chunks.json"])

    load_dotenv.assert_called_once_with()
    load_api_settings.assert_called_once_with()

    create_embedding_provider.assert_not_called()
    load_chunk_manifest.assert_not_called()
    create_engine_from_url.assert_not_called()


def test_main_propagates_embedding_configuration_error():
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url="postgresql+psycopg://test",
    )

    configuration_error = EmbeddingConfigurationError(
        "Embedding provider is invalid."
    )

    with (
        patch(
            "scripts.store_chunk_manifest.load_dotenv",
        ),
        patch(
            "scripts.store_chunk_manifest.load_api_settings",
            return_value=settings,
        ),
        patch(
            "scripts.store_chunk_manifest.create_embedding_provider",
            side_effect=configuration_error,
        ) as create_embedding_provider,
        patch(
            "scripts.store_chunk_manifest.load_chunk_manifest",
        ) as load_chunk_manifest,
        patch(
            "scripts.store_chunk_manifest.create_engine_from_url",
        ) as create_engine_from_url,
    ):
        with pytest.raises(
            EmbeddingConfigurationError,
            match="Embedding provider is invalid.",
        ):
            main(["outputs/chunks.json"])

    create_embedding_provider.assert_called_once_with(settings)
    load_chunk_manifest.assert_not_called()
    create_engine_from_url.assert_not_called()