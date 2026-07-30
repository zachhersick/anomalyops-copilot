from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import copilot.api.app as app_module
from copilot.api.app import create_app
from copilot.api.errors import DatabaseNotConfiguredError
from copilot.api.settings import ApiSettings
import copilot.api.query_service as query_service_module
from copilot.ingestion.manifest import write_chunk_manifest
from copilot.providers.errors import (
    GroundedAnswerProviderError,
    InvalidGroundedAnswerResponseError,
)
from copilot.schemas.answer import Citation, GroundedAnswer
from copilot.schemas.chunk import SourceChunk
from copilot.schemas.query import QueryResponse
from copilot.schemas.retrieval import ScoredChunk


class StubGroundedAnswerGenerator:
    provider_name = "stub"
    model_name = "stub-model"

    def __init__(
        self,
        *,
        result: GroundedAnswer | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
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

        if self.error is not None:
            raise self.error

        assert self.result is not None
        return self.result
    
    
def create_test_manifest(tmp_path):
    manifest_path = tmp_path / "chunks.json"

    write_chunk_manifest(
        [
            SourceChunk(
                chunk_id="chunk-1",
                source_id="api.py",
                project_name="test-project",
                source_type="python",
                source_path="api.py",
                chunk_index=0,
                content=(
                    "The prediction API exposes POST /predict."
                ),
                start_line=10,
                end_line=20,
            )
        ],
        manifest_path,
    )

    return manifest_path


def patch_successful_retrieval(monkeypatch) -> None:
    chunk = SourceChunk(
        chunk_id="chunk-1",
        source_id="api.py",
        project_name="test-project",
        source_type="python",
        source_path="api.py",
        chunk_index=0,
        content="The prediction API exposes POST /predict.",
        start_line=10,
        end_line=20,
    )

    monkeypatch.setattr(
        query_service_module,
        "retrieve_chunks_for_query",
        lambda *args, **kwargs: [
            ScoredChunk(
                chunk=chunk,
                score=0.9,
            )
        ],
    )


def test_query_endpoint_returns_grounded_answer(tmp_path):
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

    response = post_query_with_manifest(
        manifest_path,
        {
            "query": "prediction api",
            "top_k": 3,
            "min_score": 0.0,
            "show_context": False,
        },
    )

    data = response.json()

    assert response.status_code == 200
    assert data["answer"] == (
        "The retrieved context says: "
        "The prediction API exposes a POST /predict endpoint. [1]"
    )
    assert isinstance(data["confidence"], float)
    assert data["refusal_reason"] is None
    assert data["citations"] == [
        {
            "citation_id": 1,
            "source_path": "api.py",
            "start_line": 10,
            "end_line": 20,
        }
    ]


def test_query_endpoint_respects_top_k(tmp_path):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk("chunk-1", "prediction api endpoint", source_path="api.py"),
        make_chunk("chunk-2", "dashboard summary view", source_path="dashboard.py"),
    ]
    write_chunk_manifest(chunks, manifest_path)

    response = post_query_with_manifest(
        manifest_path,
        {
            "query": "prediction api",
            "top_k": 1,
            "min_score": 0.0,
            "show_context": True,
        },
    )

    data = response.json()

    assert response.status_code == 200
    assert len(data["context_snippets"]) == 1


def test_query_endpoint_respects_min_score_refusal(tmp_path):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk("chunk-1", "prediction api endpoint", source_path="api.py"),
    ]
    write_chunk_manifest(chunks, manifest_path)

    response = post_query_with_manifest(
        manifest_path,
        {
            "query": "prediction api",
            "top_k": 3,
            "min_score": 1.1,
            "show_context": False,
        },
    )

    data = response.json()

    assert response.status_code == 200
    assert data["answer"] == ""
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["citations"] == []
    assert (
        data["refusal_reason"]
        == "Retrieved context was below the confidence threshold."
    )


def test_query_endpoint_returns_error_when_manifest_path_not_configured():
    test_app = create_app(settings=ApiSettings())

    with TestClient(test_app) as client:
        response = client.post(
            "/query",
            json={
                "query": "prediction api",
                "top_k": 3,
                "min_score": 0.0,
                "show_context": False,
            },
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Manifest path is not configured."}
    
    
def test_query_api_returns_context_when_show_context_is_true(tmp_path):
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
    
    test_app = create_app(settings=ApiSettings(manifest_path=manifest_path))

    with TestClient(test_app) as client:
        response = client.post(
            "/query",
            json={
                "query": "prediction api",
                "top_k": 3,
                "min_score": 0.0,
                "show_context": True,
            },
        )
        
    assert response.status_code == 200    
    
    payload = response.json()
    context = payload["context"]
    context_snippets = payload["context_snippets"]
    
    assert context is not None
    assert "[1]" in context
    assert "api.py:10-20" in context
    assert "The prediction API exposes a POST /predict endpoint." in context
    
    snippet = context_snippets[0]
    
    assert snippet["citation_id"] == 1
    assert snippet["source_path"] == "api.py"
    assert snippet["start_line"] == 10
    assert snippet["end_line"] == 20
    assert snippet["content"] == "The prediction API exposes a POST /predict endpoint."
    assert isinstance(snippet["score"], float)
    
    
def test_query_api_omits_context_when_show_context_is_false(tmp_path):
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
    
    test_app = create_app(settings=ApiSettings(manifest_path=manifest_path))

    with TestClient(test_app) as client:
        response = client.post(
            "/query",
            json={
                "query": "prediction api",
                "top_k": 3,
                "min_score": 0.0,
                "show_context": False,
            },
        )
        
    assert response.status_code == 200
        
    payload = response.json()
    context = payload["context"]
    context_snippets = payload["context_snippets"]

    assert context is None
    assert context_snippets == []
    
    
def test_query_api_context_none_when_show_context_default(tmp_path):
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
    
    test_app = create_app(settings=ApiSettings(manifest_path=manifest_path))

    with TestClient(test_app) as client:
        response = client.post(
            "/query",
            json={
                "query": "prediction api",
                "top_k": 3,
                "min_score": 0.0,
            },
        )
        
    assert response.status_code == 200
        
    payload = response.json()
    context = payload["context"]
    context_snippets = payload["context_snippets"]

    assert context is None
    assert context_snippets == []
    
    
@pytest.mark.parametrize(
    "payload",
    [
        {"top_k": 3, "min_score": 0.0, "show_context": False},
        {"query": "", "top_k": 3, "min_score": 0.0, "show_context": False},
        {"query": "prediction api", "top_k": 0, "min_score": 0.0, "show_context": False},
        {"query": "prediction api", "top_k": -1, "min_score": 0.0, "show_context": False},
        {"query": "prediction api", "top_k": 3, "min_score": -0.1, "show_context": False},
    ],
)
def test_invalid_query_request_returns_422(tmp_path, payload):
    response = post_query(payload, tmp_path)
    
    assert response.status_code == 422
    assert "detail" in response.json()
    
    
def test_api_unconfigured_manifest_path_returns_500(tmp_path):
    manifest_path = tmp_path / "missing.json"
    test_app = create_app(settings=ApiSettings(manifest_path=manifest_path))

    with TestClient(test_app) as client:
        response = client.post(
            "/query",
            json={
                "query": "prediction api",
            },
        )
        
    assert response.status_code == 500
    assert response.json() == {"detail": "Manifest file was not found."}
    
    
def test_api_manifest_file_invalid(tmp_path):
    manifest_path = tmp_path / "chunks.json"
    manifest_path.write_text("{not valid json", encoding="utf-8")
    
    test_app = create_app(settings=ApiSettings(manifest_path=manifest_path))
    
    with TestClient(test_app) as client:
        response = client.post(
            "/query",
            json={
                "query": "prediction api",
            },
        )
        
    assert response.status_code == 500
    assert response.json() == {"detail": "Manifest file is invalid."}
    
    
def test_query_endpoint_passes_full_settings_to_query_service():
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url="postgresql+psycopg://test",
    )
    expected_response = QueryResponse(
        answer="Grounded answer",
        confidence=0.9,
        citations=[],
        refusal_reason=None,
        context=None,
        context_snippets=[],
    )

    test_app = create_app(settings=settings)

    with patch(
        "copilot.api.app.query_service",
        return_value=expected_response,
    ) as query_service:
        with TestClient(test_app) as client:
            response = client.post(
                "/query",
                json={
                    "query": "prediction api",
                    "top_k": 3,
                    "min_score": 0.0,
                    "show_context": False,
                },
            )

    assert response.status_code == 200

    query_service.assert_called_once()

    called_settings, called_request = query_service.call_args.args

    assert called_settings is settings
    assert called_request.query == "prediction api"
    assert called_request.top_k == 3
    assert (
        query_service.call_args.kwargs["session_factory"]
        is test_app.state.session_factory
    )
    assert (
        query_service.call_args.kwargs["embedding_provider"]
        is test_app.state.embedding_provider
    )
    assert (
        query_service.call_args.kwargs["grounded_answer_generator"]
        is test_app.state.grounded_answer_generator
    )
    
    
def test_query_endpoint_maps_missing_database_url_to_500():
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url=None,
    )
    test_app = create_app(settings=settings)

    with patch(
        "copilot.api.app.query_service",
        side_effect=DatabaseNotConfiguredError(
            "Database URL is not configured."
        ),
    ):
        with TestClient(test_app) as client:
            response = client.post(
                "/query",
                json={
                    "query": "prediction api",
                    "top_k": 3,
                    "min_score": 0.0,
                    "show_context": False,
                },
            )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Database URL is not configured."
    }
    
    
def test_pgvector_app_creates_engine_once_and_reuses_dependencies():
    engine = MagicMock()
    session_factory = MagicMock()
    embedding_provider = MagicMock()

    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url="postgresql+psycopg://test",
    )
    expected_response = QueryResponse(
        answer="Grounded answer",
        confidence=0.9,
        citations=[],
        refusal_reason=None,
        context=None,
        context_snippets=[],
    )

    payload = {
        "query": "prediction api",
        "top_k": 3,
        "min_score": 0.0,
        "show_context": False,
    }

    with (
        patch(
            "copilot.api.app.create_engine_from_url",
            return_value=engine,
        ) as create_engine_from_url,
        patch(
            "copilot.api.app.create_session_factory",
            return_value=session_factory,
        ) as create_session_factory,
        patch(
            "copilot.api.app.create_embedding_provider",
            return_value=embedding_provider,
        ) as create_embedding_provider,
        patch(
            "copilot.api.app.query_service",
            return_value=expected_response,
        ) as query_service,
    ):
        test_app = create_app(settings=settings)

        with TestClient(test_app) as client:
            first_response = client.post(
                "/query",
                json=payload,
            )
            second_response = client.post(
                "/query",
                json=payload,
            )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    create_engine_from_url.assert_called_once_with(
        "postgresql+psycopg://test"
    )
    create_session_factory.assert_called_once_with(engine)
    create_embedding_provider.assert_called_once_with(
        settings,
        openai_client=None,
    )

    assert test_app.state.embedding_provider is embedding_provider
    assert query_service.call_count == 2

    for query_call in query_service.call_args_list:
        called_settings, called_request = query_call.args

        assert called_settings is settings
        assert called_request.query == "prediction api"
        assert (
            query_call.kwargs["session_factory"]
            is session_factory
        )
        assert (
            query_call.kwargs["embedding_provider"]
            is embedding_provider
        )
        assert (
            query_call.kwargs["grounded_answer_generator"]
            is test_app.state.grounded_answer_generator
        )
        
        
def test_pgvector_app_disposes_engine_on_shutdown():
    engine = MagicMock()
    session_factory = MagicMock()
    embedding_provider = MagicMock()

    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url="postgresql+psycopg://test",
    )

    with (
        patch(
            "copilot.api.app.create_engine_from_url",
            return_value=engine,
        ),
        patch(
            "copilot.api.app.create_session_factory",
            return_value=session_factory,
        ),
        patch(
            "copilot.api.app.create_embedding_provider",
            return_value=embedding_provider,
        ),
    ):
        test_app = create_app(settings=settings)

        with TestClient(test_app):
            pass

    engine.dispose.assert_called_once_with()
    
    
def test_manifest_app_does_not_create_embedding_provider():
    settings = ApiSettings(
        retrieval_backend="manifest",
    )

    with patch(
        "copilot.api.app.create_embedding_provider",
    ) as create_embedding_provider:
        test_app = create_app(settings=settings)

    assert test_app.state.embedding_provider is None
    create_embedding_provider.assert_not_called()


def test_pgvector_app_creates_deterministic_embedding_provider():
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url="postgresql+psycopg://test",
        ai_provider="deterministic",
        embedding_dimensions=16,
    )

    with (
        patch(
            "copilot.api.app.create_engine_from_url",
        ),
        patch(
            "copilot.api.app.create_session_factory",
        ),
    ):
        test_app = create_app(settings=settings)

    provider = test_app.state.embedding_provider

    assert provider is not None
    assert provider.provider_name == "deterministic"
    assert provider.dimensions == 16
    
    
def post_query(payload: dict, tmp_path):
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
    
    test_app = create_app(settings=ApiSettings(manifest_path=manifest_path))

    with TestClient(test_app) as client:
        return client.post("/query", json=payload)
    

def post_query_with_manifest(manifest_path, payload):
    settings = ApiSettings(manifest_path=manifest_path)
    test_app = create_app(settings=settings)

    with TestClient(test_app) as client:
        return client.post("/query", json=payload)
    
    
def test_create_app_stores_grounded_answer_generator(
    monkeypatch,
    tmp_path,
):
    manifest_path = create_test_manifest(tmp_path)
    generator = StubGroundedAnswerGenerator(
        result=GroundedAnswer(
            answer="Generated answer. [1]",
            citations=[
                Citation(
                    citation_id=1,
                    source_path="api.py",
                    start_line=10,
                    end_line=20,
                )
            ],
            confidence=0.8,
            refusal_reason=None,
        )
    )
    captured = {}

    def fake_create_generator(
        settings,
        openai_client=None,
    ):
        captured["settings"] = settings
        captured["openai_client"] = openai_client
        return generator

    monkeypatch.setattr(
        app_module,
        "create_grounded_answer_generator",
        fake_create_generator,
    )

    settings = ApiSettings(
        retrieval_backend="manifest",
        manifest_path=manifest_path,
    )
    fake_openai_client = object()

    app = create_app(
        settings=settings,
        openai_client=fake_openai_client,
    )

    assert app.state.grounded_answer_generator is generator
    assert captured == {
        "settings": settings,
        "openai_client": fake_openai_client,
    }


def test_query_route_uses_grounded_answer_generator(
    monkeypatch,
    tmp_path,
):
    patch_successful_retrieval(monkeypatch)
    manifest_path = create_test_manifest(tmp_path)
    generator = StubGroundedAnswerGenerator(
        result=GroundedAnswer(
            answer="The endpoint is POST /predict. [1]",
            citations=[
                Citation(
                    citation_id=1,
                    source_path="api.py",
                    start_line=10,
                    end_line=20,
                )
            ],
            confidence=0.88,
            refusal_reason=None,
        )
    )

    monkeypatch.setattr(
        app_module,
        "create_grounded_answer_generator",
        lambda *args, **kwargs: generator,
    )

    app = create_app(
        settings=ApiSettings(
            retrieval_backend="manifest",
            manifest_path=manifest_path,
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={
                "query": "Where is the prediction endpoint?",
                "top_k": 3,
                "min_score": 0.0,
                "show_context": True,
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["answer"] == (
        "The endpoint is POST /predict. [1]"
    )
    assert payload["confidence"] == 0.88
    assert payload["refusal_reason"] is None
    assert payload["citations"] == [
        {
            "citation_id": 1,
            "source_path": "api.py",
            "start_line": 10,
            "end_line": 20,
        }
    ]

    assert len(generator.calls) == 1
    assert generator.calls[0]["query"] == (
        "Where is the prediction endpoint?"
    )

    assert len(payload["context_snippets"]) == 1
    assert payload["context_snippets"][0]["citation_id"] == 1


def test_query_route_maps_invalid_provider_response_to_502(
    monkeypatch,
    tmp_path,
):
    patch_successful_retrieval(monkeypatch)
    manifest_path = create_test_manifest(tmp_path)
    generator = StubGroundedAnswerGenerator(
        error=InvalidGroundedAnswerResponseError(
            "Invalid provider output."
        )
    )

    monkeypatch.setattr(
        app_module,
        "create_grounded_answer_generator",
        lambda *args, **kwargs: generator,
    )

    app = create_app(
        settings=ApiSettings(
            retrieval_backend="manifest",
            manifest_path=manifest_path,
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={
                "query": "Where is the prediction endpoint?",
                "top_k": 3,
                "min_score": 0.0,
                "show_context": False,
            },
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": (
            "Grounded answer provider returned an invalid response."
        )
    }


def test_query_route_maps_provider_request_failure_to_502(
    monkeypatch,
    tmp_path,
):
    patch_successful_retrieval(monkeypatch)
    manifest_path = create_test_manifest(tmp_path)
    generator = StubGroundedAnswerGenerator(
        error=GroundedAnswerProviderError(
            "Provider request failed."
        )
    )

    monkeypatch.setattr(
        app_module,
        "create_grounded_answer_generator",
        lambda *args, **kwargs: generator,
    )

    app = create_app(
        settings=ApiSettings(
            retrieval_backend="manifest",
            manifest_path=manifest_path,
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={
                "query": "Where is the prediction endpoint?",
                "top_k": 3,
                "min_score": 0.0,
                "show_context": False,
            },
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Grounded answer provider request failed."
    }


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