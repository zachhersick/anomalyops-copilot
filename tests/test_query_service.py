from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import pytest

import copilot.api.query_service as query_service_module
from copilot.api.errors import (
    DatabaseNotConfiguredError,
    InvalidManifestError,
    ManifestFileNotFoundError,
    ManifestNotConfiguredError,
)
from copilot.api.query_service import (
    query_service,
    retrieve_chunks_for_query,
)
from copilot.api.settings import ApiSettings
from copilot.ingestion.manifest import write_chunk_manifest
from copilot.schemas.answer import Citation, GroundedAnswer
from copilot.schemas.chunk import SourceChunk
from copilot.schemas.query import QueryRequest, QueryResponse
from copilot.schemas.retrieval import ScoredChunk


class FakeEmbeddingProvider:
    provider_name = "fake"
    model_name = "fake-embedding"
    dimensions = 16

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return [0.25] * self.dimensions

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        raise NotImplementedError
    
    
class RecordingGroundedAnswerGenerator:
    provider_name = "recording"
    model_name = "recording-test"

    def __init__(
        self,
        result: GroundedAnswer,
    ) -> None:
        self.result = result
        self.calls: list[dict] = []

    def generate(
        self,
        query: str,
        context: Sequence[ScoredChunk],
    ) -> GroundedAnswer:
        self.calls.append(
            {
                "query": query,
                "context": context,
            }
        )
        return self.result


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
    assert query_response.answer == (
        "The retrieved context says: "
        "The prediction API exposes a POST /predict endpoint. [1]"
    )
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
    assert 0.0 <= query_response.confidence <= 1.0
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
    embedding_provider = FakeEmbeddingProvider()

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
            embedding_provider=embedding_provider,
        )

    assert results == expected_results

    session_factory.assert_called_once_with()
    retrieve_pgvector.assert_called_once_with(
        session=session,
        query="prediction api",
        embedding_provider=embedding_provider,
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
        "The prediction API exposes a POST /predict endpoint. [1]"
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
        
        
def test_query_service_passes_session_factory_and_provider_to_retrieval():
    session_factory = MagicMock()
    embedding_provider = FakeEmbeddingProvider()

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
            embedding_provider=embedding_provider,
        )

    retrieve_chunks.assert_called_once_with(
        settings,
        query_request,
        session_factory=session_factory,
        embedding_provider=embedding_provider,
    )
    
    
def test_pgvector_backend_requires_embedding_provider():
    session_factory = MagicMock()

    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url="postgresql+psycopg://test",
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding provider is not configured.",
    ):
        retrieve_chunks_for_query(
            settings,
            make_query_request(query="prediction api"),
            session_factory=session_factory,
            embedding_provider=None,
        )

    session_factory.assert_not_called()
    
    
def test_query_service_injects_grounded_answer_generator(
    monkeypatch,
):
    selected_chunks = [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="First context.",
            score=0.7,
            source_path="first.py",
            chunk_index=0,
        ),
        make_scored_chunk(
            chunk_id="chunk-2",
            content="Second context.",
            score=0.9,
            source_path="second.py",
            chunk_index=1,
        ),
    ]
    generated_answer = GroundedAnswer(
        answer="The generated answer uses the second chunk. [2]",
        citations=[
            Citation(
                citation_id=2,
                source_path="second.py",
                start_line=1,
                end_line=5,
            )
        ],
        confidence=0.83,
        refusal_reason=None,
    )
    generator = RecordingGroundedAnswerGenerator(
        generated_answer
    )

    monkeypatch.setattr(
        query_service_module,
        "retrieve_chunks_for_query",
        lambda *args, **kwargs: selected_chunks,
    )

    response = query_service(
        settings=ApiSettings(),
        query_request=QueryRequest(
            query="Exact query",
            top_k=3,
            min_score=0.0,
            show_context=False,
        ),
        grounded_answer_generator=generator,
    )

    assert generator.calls == [
        {
            "query": "Exact query",
            "context": selected_chunks,
        }
    ]
    assert generator.calls[0]["context"] is selected_chunks

    assert response.answer == generated_answer.answer
    assert response.confidence == generated_answer.confidence
    assert response.citations == generated_answer.citations
    assert response.refusal_reason is None
    assert response.context is None
    assert response.context_snippets == []


def test_query_service_refuses_before_calling_generator_below_min_score(
    monkeypatch,
):
    selected_chunks = [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="Weak context.",
            score=0.4,
        )
    ]
    generator = RecordingGroundedAnswerGenerator(
        GroundedAnswer(
            answer="This should not be returned. [1]",
            citations=[
                Citation(
                    citation_id=1,
                    source_path="source.py",
                    start_line=1,
                    end_line=5,
                )
            ],
            confidence=0.4,
            refusal_reason=None,
        )
    )

    monkeypatch.setattr(
        query_service_module,
        "retrieve_chunks_for_query",
        lambda *args, **kwargs: selected_chunks,
    )

    response = query_service(
        settings=ApiSettings(),
        query_request=QueryRequest(
            query="Question",
            top_k=3,
            min_score=0.5,
            show_context=False,
        ),
        grounded_answer_generator=generator,
    )

    assert generator.calls == []
    assert response.answer == ""
    assert response.citations == []
    assert response.confidence == pytest.approx(0.4)
    assert response.refusal_reason == (
        "Retrieved context was below the confidence threshold."
    )


def test_query_service_displays_all_context_independent_of_citations(
    monkeypatch,
):
    selected_chunks = [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="First retrieved context.",
            score=0.7,
            source_path="first.py",
            start_line=10,
            end_line=20,
            chunk_index=0,
        ),
        make_scored_chunk(
            chunk_id="chunk-2",
            content="Second retrieved context.",
            score=0.9,
            source_path="second.py",
            start_line=30,
            end_line=40,
            chunk_index=1,
        ),
    ]
    generator = RecordingGroundedAnswerGenerator(
        GroundedAnswer(
            answer="Only the second chunk supports this answer. [2]",
            citations=[
                Citation(
                    citation_id=2,
                    source_path="second.py",
                    start_line=30,
                    end_line=40,
                )
            ],
            confidence=0.9,
            refusal_reason=None,
        )
    )

    monkeypatch.setattr(
        query_service_module,
        "retrieve_chunks_for_query",
        lambda *args, **kwargs: selected_chunks,
    )

    response = query_service(
        settings=ApiSettings(),
        query_request=QueryRequest(
            query="Question",
            top_k=3,
            min_score=0.0,
            show_context=True,
        ),
        grounded_answer_generator=generator,
    )

    assert [
        citation.citation_id
        for citation in response.citations
    ] == [2]

    assert len(response.context_snippets) == 2

    assert response.context_snippets[0].citation_id == 1
    assert response.context_snippets[0].source_path == "first.py"
    assert response.context_snippets[0].content == (
        "First retrieved context."
    )

    assert response.context_snippets[1].citation_id == 2
    assert response.context_snippets[1].source_path == "second.py"
    assert response.context_snippets[1].content == (
        "Second retrieved context."
    )

    assert response.context is not None
    assert "First retrieved context." in response.context
    assert "Second retrieved context." in response.context


def test_query_service_maps_provider_refusal_exactly(
    monkeypatch,
):
    selected_chunks = [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="Retrieved context.",
            score=0.8,
        )
    ]
    provider_result = GroundedAnswer(
        answer="",
        citations=[],
        confidence=0.0,
        refusal_reason="The context does not support the answer.",
    )
    generator = RecordingGroundedAnswerGenerator(
        provider_result
    )

    monkeypatch.setattr(
        query_service_module,
        "retrieve_chunks_for_query",
        lambda *args, **kwargs: selected_chunks,
    )

    response = query_service(
        settings=ApiSettings(),
        query_request=QueryRequest(
            query="Unsupported question",
            top_k=3,
            min_score=0.0,
            show_context=False,
        ),
        grounded_answer_generator=generator,
    )

    assert response.answer == provider_result.answer
    assert response.citations == provider_result.citations
    assert response.confidence == provider_result.confidence
    assert response.refusal_reason == (
        provider_result.refusal_reason
    )


def test_query_service_passes_embedding_dependency_to_retrieval(
    monkeypatch,
):
    selected_chunks = [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="Retrieved context.",
            score=0.8,
        )
    ]
    embedding_provider = object()
    captured = {}

    def fake_retrieve_chunks_for_query(
        settings,
        query_request,
        session_factory=None,
        embedding_provider=None,
    ):
        captured["settings"] = settings
        captured["query_request"] = query_request
        captured["session_factory"] = session_factory
        captured["embedding_provider"] = embedding_provider
        return selected_chunks

    monkeypatch.setattr(
        query_service_module,
        "retrieve_chunks_for_query",
        fake_retrieve_chunks_for_query,
    )

    settings = ApiSettings()
    request = QueryRequest(
        query="Question",
        top_k=3,
        min_score=0.0,
        show_context=False,
    )
    session_factory = object()

    query_service(
        settings=settings,
        query_request=request,
        session_factory=session_factory,
        embedding_provider=embedding_provider,
    )

    assert captured == {
        "settings": settings,
        "query_request": request,
        "session_factory": session_factory,
        "embedding_provider": embedding_provider,
    }
    
    
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
    

def make_scored_chunk(
    chunk_id: str,
    content: str,
    score: float,
    *,
    source_path: str = "source.py",
    start_line: int = 1,
    end_line: int = 5,
    chunk_index: int = 0,
) -> ScoredChunk:
    return ScoredChunk(
        chunk=SourceChunk(
            chunk_id=chunk_id,
            source_id=source_path,
            project_name="test-project",
            source_type="python",
            source_path=source_path,
            chunk_index=chunk_index,
            content=content,
            start_line=start_line,
            end_line=end_line,
        ),
        score=score,
    )