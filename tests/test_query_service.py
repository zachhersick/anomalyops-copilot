import pytest

from unittest.mock import MagicMock, patch

from copilot.api.query_service import query_service, retrieve_chunks_for_query
from copilot.ingestion.manifest import write_chunk_manifest
from copilot.schemas.chunk import SourceChunk
from copilot.schemas.query import QueryRequest, QueryResponse
from copilot.api.settings import ApiSettings
from copilot.schemas.retrieval import ScoredChunk
from copilot.api.errors import (
    ManifestNotConfiguredError,
    ManifestFileNotFoundError,
    InvalidManifestError,
    DatabaseNotConfiguredError,
)


def test_service_returns_grounded_answer_with_citations(tmp_path):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk(
            "chunk-1",
            "The prediction API exposes a POST /predict endpoint.",
            source_path="api.py",
            start_line=10,
            end_line=20,
        ),
    ]
    write_chunk_manifest(chunks, manifest_path)
    
    query_response = query_service(
        ApiSettings(
            retrieval_backend="manifest",
            manifest_path=manifest_path,
        ),
        make_query_request(query="prediction api"),
    )
    
    assert isinstance(query_response, QueryResponse)
    assert query_response.answer == "The retrieved context says: The prediction API exposes a POST /predict endpoint."
    assert isinstance(query_response.confidence, float)
    assert query_response.refusal_reason is None
    assert len(query_response.citations) == 1
    assert query_response.citations[0].citation_id == 1
    assert query_response.citations[0].source_path == "api.py"
    assert query_response.citations[0].start_line == 10
    assert query_response.citations[0].end_line == 20
    
    
def test_service_respects_min_score_refusal(tmp_path):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk(
            "chunk-1",
            "The prediction API exposes a POST /predict endpoint.",
            source_path="api.py",
            start_line=10,
            end_line=20,
        ),
    ]
    write_chunk_manifest(chunks, manifest_path)
    
    query_response = query_service(
        ApiSettings(
            retrieval_backend="manifest",
            manifest_path=manifest_path,
        ),
        make_query_request(query="prediction api", min_score=1.1),
    )
    
    assert query_response.answer == ""
    assert query_response.citations == []
    assert query_response.confidence == 0.0
    assert (
        query_response.refusal_reason
        == "Retrieved context was below the confidence threshold."
    )
        
        
def test_service_show_context_equals_true(tmp_path):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk(
            "chunk-1",
            "The prediction API exposes a POST /predict endpoint.",
            source_path="api.py",
            start_line=10,
            end_line=20,
        ),
    ]
    write_chunk_manifest(chunks, manifest_path)
    
    query_response = query_service(
        ApiSettings(
            retrieval_backend="manifest",
            manifest_path=manifest_path,
        ),
        make_query_request(query="prediction api", show_context=True),
    )
    
    assert query_response.context is not None
    assert "[1]" in query_response.context
    assert "api.py:10-20" in query_response.context
    assert "The prediction API exposes a POST /predict endpoint." in query_response.context
    
    snippet = query_response.context_snippets[0]
    
    assert snippet.citation_id == 1
    assert snippet.source_path == "api.py"
    assert snippet.start_line == 10
    assert snippet.end_line == 20
    assert snippet.content == "The prediction API exposes a POST /predict endpoint."
    assert isinstance(snippet.score, float)
    
    
def test_service_show_context_equals_false(tmp_path):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk(
            "chunk-1",
            "The prediction API exposes a POST /predict endpoint.",
            source_path="api.py",
            start_line=10,
            end_line=20,
        ),
    ]
    write_chunk_manifest(chunks, manifest_path)
    
    query_response = query_service(
        ApiSettings(
            retrieval_backend="manifest",
            manifest_path=manifest_path,
        ),
        make_query_request(query="prediction api", show_context=False),
    )
    
    assert query_response.context is None
    assert query_response.context_snippets == []
    
    
def test_query_service_raises_manifest_not_configured_when_manifest_path_is_None():
    settings = ApiSettings(
        retrieval_backend="manifest",
        manifest_path=None,
    )
    
    with pytest.raises(
        ManifestNotConfiguredError,
        match="Manifest path is not configured.",
    ):
        query_service(
            settings,
            make_query_request(query="prediction api"),
        )
        
        
def test_query_service_missing_manifest_file_returns_manifest_file_not_found_error(tmp_path):
    settings = ApiSettings(
        retrieval_backend="manifest",
        manifest_path=tmp_path / "missing.json",
    )
    
    with pytest.raises(ManifestFileNotFoundError):
        query_service(
            settings,
            make_query_request(query="prediction api"),
        )
        
        
def test_query_service_invalid_manifest(tmp_path):
    manifest_path = tmp_path / "chunks.json"
    manifest_path.write_text("{not valid json", encoding="utf-8")
    
    settings = ApiSettings(
        retrieval_backend="manifest",
        manifest_path=manifest_path,
    )
    
    with pytest.raises(InvalidManifestError):
        query_service(
            settings,
            make_query_request(query="prediction api"),
        )
        
        
def test_retrieve_chunks_for_query_uses_manifest_backend(tmp_path):
    manifest_path = tmp_path / "chunks.json"
    manifest_path.write_text("[]", encoding="utf-8")

    chunks = [
        make_chunk(
            "chunk-1",
            "The prediction API exposes a POST /predict endpoint.",
            source_path="api.py",
        ),
    ]
    expected_results = [
        ScoredChunk(
            chunk=chunks[0],
            score=0.9,
        ),
    ]

    settings = ApiSettings(
        retrieval_backend="manifest",
        manifest_path=manifest_path,
    )
    query_request = make_query_request(
        query="prediction api",
        top_k=2,
    )

    with (
        patch(
            "copilot.api.query_service.load_chunk_manifest",
            return_value=chunks,
        ) as load_chunk_manifest,
        patch(
            "copilot.api.query_service.retrieve_relevant_chunks",
            return_value=expected_results,
        ) as retrieve_relevant_chunks,
    ):
        results = retrieve_chunks_for_query(
            settings,
            query_request,
        )

    assert results == expected_results

    load_chunk_manifest.assert_called_once_with(manifest_path)
    retrieve_relevant_chunks.assert_called_once_with(
        query="prediction api",
        chunks=chunks,
        top_k=2,
    )
    
    
def test_retrieve_chunks_for_query_uses_shared_pgvector_session_factory():
    session = MagicMock()
    session_factory = MagicMock()
    session_factory.return_value.__enter__.return_value = session

    expected_results = [
        ScoredChunk(
            chunk=make_chunk(
                "chunk-1",
                "The prediction API exposes a POST /predict endpoint.",
                source_path="api.py",
            ),
            score=0.9,
        ),
    ]

    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url="postgresql+psycopg://test",
    )
    query_request = make_query_request(
        query="prediction api",
        top_k=4,
    )

    with patch(
        "copilot.api.query_service.retrieve_relevant_chunks_from_pgvector",
        return_value=expected_results,
    ) as retrieve_pgvector:
        results = retrieve_chunks_for_query(
            settings,
            query_request,
            session_factory=session_factory,
        )

    assert results == expected_results

    session_factory.assert_called_once_with()
    retrieve_pgvector.assert_called_once_with(
        session=session,
        query="prediction api",
        top_k=4,
    )
    
    
def test_pgvector_backend_requires_database_url():
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url=None,
    )

    with pytest.raises(
        DatabaseNotConfiguredError,
        match="Database URL is not configured.",
    ):
        retrieve_chunks_for_query(
            settings,
            make_query_request(query="prediction api"),
            session_factory=None,
        )
        
        
def test_query_service_returns_same_response_for_both_backends():
    scored_chunks = [
        ScoredChunk(
            chunk=make_chunk(
                "chunk-1",
                "The prediction API exposes a POST /predict endpoint.",
                source_path="api.py",
                start_line=10,
                end_line=20,
            ),
            score=0.9,
        ),
    ]

    query_request = make_query_request(
        query="prediction api",
        show_context=True,
    )

    manifest_settings = ApiSettings(
        retrieval_backend="manifest",
        manifest_path="outputs/chunks.json",
    )
    pgvector_settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url="postgresql+psycopg://test",
    )

    with patch(
        "copilot.api.query_service.retrieve_chunks_for_query",
        return_value=scored_chunks,
    ):
        manifest_response = query_service(
            manifest_settings,
            query_request,
        )
        pgvector_response = query_service(
            pgvector_settings,
            query_request,
        )

    assert manifest_response.model_dump() == pgvector_response.model_dump()

    assert manifest_response.answer == (
        "The retrieved context says: "
        "The prediction API exposes a POST /predict endpoint."
    )
    assert manifest_response.confidence == pytest.approx(0.9)
    assert len(manifest_response.citations) == 1
    assert manifest_response.context is not None
    assert len(manifest_response.context_snippets) == 1
    
    
def test_pgvector_backend_requires_session_factory():
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url="postgresql+psycopg://test",
    )

    with pytest.raises(
        RuntimeError,
        match="Database session factory is not configured.",
    ):
        retrieve_chunks_for_query(
            settings,
            make_query_request(query="prediction api"),
            session_factory=None,
        )
        
        
def test_query_service_passes_session_factory_to_retrieval():
    session_factory = MagicMock()
    scored_chunks = [
        ScoredChunk(
            chunk=make_chunk(
                "chunk-1",
                "The prediction API exposes a POST /predict endpoint.",
                source_path="api.py",
            ),
            score=0.9,
        ),
    ]

    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url="postgresql+psycopg://test",
    )
    query_request = make_query_request(query="prediction api")

    with patch(
        "copilot.api.query_service.retrieve_chunks_for_query",
        return_value=scored_chunks,
    ) as retrieve_chunks:
        query_service(
            settings,
            query_request,
            session_factory=session_factory,
        )

    retrieve_chunks.assert_called_once_with(
        settings,
        query_request,
        session_factory=session_factory,
    )
    
    
def make_query_request(
    query: str,
    top_k: int = 3,
    min_score: float = 0.0,
    show_context: bool = False,
) -> QueryRequest:
    return QueryRequest(
        query=query,
        top_k=top_k,
        min_score=min_score,
        show_context=show_context,
    )
    
    
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