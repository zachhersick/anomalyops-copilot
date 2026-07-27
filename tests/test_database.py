from unittest.mock import MagicMock, patch

import pytest

from copilot.storage.database import (
    create_engine_from_url,
    create_session_factory,
    initialize_database,
    rebuild_source_chunks_table,
)
from copilot.storage.models import Base, SourceChunkRecord


def test_create_engine_from_url_uses_postgresql_psycopg():
    engine = create_engine_from_url(
        "postgresql+psycopg://user:password@localhost/test_database"
    )

    assert engine.url.drivername == "postgresql+psycopg"
    assert engine.url.username == "user"
    assert engine.url.host == "localhost"
    assert engine.url.database == "test_database"


def test_create_session_factory_binds_engine():
    engine = create_engine_from_url(
        "postgresql+psycopg://user:password@localhost/test_database"
    )

    session_factory = create_session_factory(engine)

    assert session_factory.kw["bind"] is engine


def test_initialize_database_enables_vector_and_creates_tables():
    engine = MagicMock()
    connection = MagicMock()

    engine.begin.return_value.__enter__.return_value = connection

    with patch.object(Base.metadata, "create_all") as create_all:
        initialize_database(engine)

    engine.begin.assert_called_once_with()
    connection.execute.assert_called_once()

    statement = connection.execute.call_args.args[0]

    assert str(statement) == "CREATE EXTENSION IF NOT EXISTS vector"
    create_all.assert_called_once_with(bind=connection)


def test_rebuild_source_chunks_table_drops_and_recreates_only_that_table():
    engine = MagicMock()
    table = SourceChunkRecord.__table__

    with (
        patch.object(table, "drop") as drop,
        patch.object(table, "create") as create,
    ):
        rebuild_source_chunks_table(engine)

    drop.assert_called_once_with(
        bind=engine,
        checkfirst=True,
    )
    create.assert_called_once_with(
        bind=engine,
        checkfirst=False,
    )


def test_rebuild_source_chunks_table_propagates_drop_failure():
    engine = MagicMock()
    table = SourceChunkRecord.__table__

    with (
        patch.object(
            table,
            "drop",
            side_effect=RuntimeError("drop failed"),
        ),
        patch.object(table, "create") as create,
    ):
        with pytest.raises(RuntimeError, match="drop failed"):
            rebuild_source_chunks_table(engine)

    create.assert_not_called()