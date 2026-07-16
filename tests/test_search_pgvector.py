from unittest.mock import MagicMock, patch

import pytest

from copilot.schemas.chunk import SourceChunk
from copilot.schemas.retrieval import ScoredChunk
from scripts.search_pgvector import main


def make_scored_chunk(
    chunk_id: str,
    source_path: str,
    content: str,
    score: float,
) -> ScoredChunk:
    return ScoredChunk(
        chunk=SourceChunk(
            chunk_id=chunk_id,
            source_id=source_path,
            project_name="test-project",
            source_type="python",
            source_path=source_path,
            chunk_index=0,
            content=content,
            start_line=1,
            end_line=5,
        ),
        score=score,
    )


def test_main_searches_pgvector_and_prints_results(capsys):
    engine = MagicMock()
    session = MagicMock()
    session_factory = MagicMock()

    session_factory.return_value.__enter__.return_value = session

    results = [
        make_scored_chunk(
            chunk_id="chunk-1",
            source_path="source_code/generator.py",
            content="The platform generates spike and drift anomalies.",
            score=0.9,
        ),
        make_scored_chunk(
            chunk_id="chunk-2",
            source_path="README.md",
            content="Supported anomaly types are documented here.",
            score=0.7,
        ),
    ]

    with (
        patch(
            "scripts.search_pgvector.load_dotenv",
        ) as load_dotenv,
        patch(
            "scripts.search_pgvector.os.getenv",
            return_value="postgresql+psycopg://test",
        ),
        patch(
            "scripts.search_pgvector.create_engine_from_url",
            return_value=engine,
        ) as create_engine_from_url,
        patch(
            "scripts.search_pgvector.create_session_factory",
            return_value=session_factory,
        ) as create_session_factory,
        patch(
            "scripts.search_pgvector.retrieve_relevant_chunks_from_pgvector",
            return_value=results,
        ) as retrieve_chunks,
    ):
        result = main(
            [
                "What anomaly types are supported?",
                "--top-k",
                "2",
            ]
        )

    assert result == 0

    load_dotenv.assert_called_once_with()
    create_engine_from_url.assert_called_once_with(
        "postgresql+psycopg://test"
    )
    create_session_factory.assert_called_once_with(engine)
    retrieve_chunks.assert_called_once_with(
        session,
        "What anomaly types are supported?",
        top_k=2,
    )

    output = capsys.readouterr().out

    assert "0.9000" in output
    assert "source_code/generator.py:1-5" in output
    assert "spike and drift anomalies" in output

    assert "0.7000" in output
    assert "README.md:1-5" in output


def test_main_prints_nothing_when_no_chunks_are_found(capsys):
    engine = MagicMock()
    session = MagicMock()
    session_factory = MagicMock()

    session_factory.return_value.__enter__.return_value = session

    with (
        patch(
            "scripts.search_pgvector.load_dotenv",
        ),
        patch(
            "scripts.search_pgvector.os.getenv",
            return_value="postgresql+psycopg://test",
        ),
        patch(
            "scripts.search_pgvector.create_engine_from_url",
            return_value=engine,
        ),
        patch(
            "scripts.search_pgvector.create_session_factory",
            return_value=session_factory,
        ),
        patch(
            "scripts.search_pgvector.retrieve_relevant_chunks_from_pgvector",
            return_value=[],
        ),
    ):
        result = main(["query"])

    assert result == 0
    assert capsys.readouterr().out == ""


def test_main_raises_when_database_url_is_missing():
    with (
        patch(
            "scripts.search_pgvector.load_dotenv",
        ),
        patch(
            "scripts.search_pgvector.os.getenv",
            return_value=None,
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="ANOMALYOPS_DATABASE_URL is not configured",
        ):
            main(["query"])