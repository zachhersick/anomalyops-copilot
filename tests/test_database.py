from unittest.mock import MagicMock, patch

from copilot.storage.database import (
    create_engine_from_url,
    create_session_factory,
    initialize_database,
)
from copilot.storage.models import Base


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