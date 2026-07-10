import pytest

from fastapi.testclient import TestClient

from copilot.api.app import create_app
from copilot.api.settings import ApiSettings
from copilot.ingestion.manifest import write_chunk_manifest
from copilot.schemas.chunk import SourceChunk


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
    assert data["answer"] == "The retrieved context says: The prediction API exposes a POST /predict endpoint."
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
            "show_context": False,
        },
    )

    data = response.json()

    assert response.status_code == 200
    assert len(data["citations"]) == 1


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
    assert data["confidence"] == 0.0
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