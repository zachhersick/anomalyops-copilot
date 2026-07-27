from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from copilot.api.settings import ApiSettings
from copilot.providers.errors import EmbeddingConfigurationError
from scripts.reindex_embeddings import main


def test_main_rebuilds_and_reindexes_embeddings(capsys):
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url="postgresql+psycopg://test",
    )
    chunks = [MagicMock(), MagicMock()]
    embedding_provider = MagicMock()
    engine = MagicMock()
    session = MagicMock()
    session_factory = MagicMock()

    session_factory.return_value.__enter__.return_value = session

    with (
        patch(
            "scripts.reindex_embeddings.load_dotenv",
        ) as load_dotenv,
        patch(
            "scripts.reindex_embeddings.load_api_settings",
            return_value=settings,
        ) as load_api_settings,
        patch(
            "scripts.reindex_embeddings.create_embedding_provider",
            return_value=embedding_provider,
        ) as create_embedding_provider,
        patch(
            "scripts.reindex_embeddings.load_chunk_manifest",
            return_value=chunks,
        ) as load_chunk_manifest,
        patch(
            "scripts.reindex_embeddings.create_engine_from_url",
            return_value=engine,
        ) as create_engine_from_url,
        patch(
            "scripts.reindex_embeddings.initialize_database",
        ) as initialize_database,
        patch(
            "scripts.reindex_embeddings.rebuild_source_chunks_table",
        ) as rebuild_source_chunks_table,
        patch(
            "scripts.reindex_embeddings.create_session_factory",
            return_value=session_factory,
        ) as create_session_factory,
        patch(
            "scripts.reindex_embeddings.store_source_chunks",
            return_value=2,
        ) as store_source_chunks,
    ):
        database_steps = MagicMock()
        database_steps.attach_mock(
            initialize_database,
            "initialize_database",
        )
        database_steps.attach_mock(
            rebuild_source_chunks_table,
            "rebuild_source_chunks_table",
        )
        database_steps.attach_mock(
            create_session_factory,
            "create_session_factory",
        )

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

    assert database_steps.mock_calls == [
        call.initialize_database(engine),
        call.rebuild_source_chunks_table(engine),
        call.create_session_factory(engine),
    ]

    session_factory.assert_called_once_with()

    store_source_chunks.assert_called_once_with(
        session=session,
        chunks=chunks,
        embedding_provider=embedding_provider,
    )

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
            "scripts.reindex_embeddings.load_dotenv",
        ),
        patch(
            "scripts.reindex_embeddings.load_api_settings",
            return_value=settings,
        ),
        patch(
            "scripts.reindex_embeddings.create_embedding_provider",
        ) as create_embedding_provider,
        patch(
            "scripts.reindex_embeddings.load_chunk_manifest",
        ) as load_chunk_manifest,
        patch(
            "scripts.reindex_embeddings.create_engine_from_url",
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

    create_embedding_provider.assert_not_called()
    load_chunk_manifest.assert_not_called()
    create_engine_from_url.assert_not_called()


def test_main_propagates_embedding_configuration_error():
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url="postgresql+psycopg://test",
    )

    with (
        patch(
            "scripts.reindex_embeddings.load_dotenv",
        ),
        patch(
            "scripts.reindex_embeddings.load_api_settings",
            return_value=settings,
        ),
        patch(
            "scripts.reindex_embeddings.create_embedding_provider",
            side_effect=EmbeddingConfigurationError(
                "Embedding configuration is invalid."
            ),
        ),
        patch(
            "scripts.reindex_embeddings.load_chunk_manifest",
        ) as load_chunk_manifest,
        patch(
            "scripts.reindex_embeddings.create_engine_from_url",
        ) as create_engine_from_url,
        patch(
            "scripts.reindex_embeddings.rebuild_source_chunks_table",
        ) as rebuild_source_chunks_table,
    ):
        with pytest.raises(
            EmbeddingConfigurationError,
            match="Embedding configuration is invalid.",
        ):
            main(["outputs/chunks.json"])

    load_chunk_manifest.assert_not_called()
    create_engine_from_url.assert_not_called()
    rebuild_source_chunks_table.assert_not_called()