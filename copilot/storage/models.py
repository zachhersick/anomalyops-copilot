from pgvector.sqlalchemy import VECTOR
from sqlalchemy import String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


EMBEDDING_DIMENSIONS = 16


class Base(DeclarativeBase):
    pass


class SourceChunkRecord(Base):
    __tablename__ = "source_chunks"
    
    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )
    chunk_id: Mapped[str] = mapped_column(
        String,
        unique=True,
        nullable=False,
    )
    source_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    project_name: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    source_type: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    source_path: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    start_line: Mapped[int] = mapped_column(
        nullable=False,
    )
    end_line: Mapped[int] = mapped_column(
        nullable=False,
    )
    embedding: Mapped[list[float]] = mapped_column(
        VECTOR(EMBEDDING_DIMENSIONS),
        nullable=False,
    )