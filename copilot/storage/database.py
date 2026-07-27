from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from copilot.storage.models import Base, SourceChunkRecord


def create_engine_from_url(database_url: str) -> Engine:
    return create_engine(database_url)


def create_session_factory(
    engine: Engine,
) -> sessionmaker[Session]:
    return sessionmaker(bind=engine)


def initialize_database(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text("CREATE EXTENSION IF NOT EXISTS vector")
        )
        Base.metadata.create_all(bind=connection)
        

def rebuild_source_chunks_table(
    engine: Engine,
) -> None:
    table = SourceChunkRecord.__table__
    
    table.drop(
        bind=engine,
        checkfirst=True,
    )
    table.create(
        bind=engine,
        checkfirst=False,
    )