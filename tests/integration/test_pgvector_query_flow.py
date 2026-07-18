import pytest

from sqlalchemy import delete
from fastapi.testclient import TestClient

from copilot.schemas.chunk import SourceChunk
from copilot.storage.chunks import store_source_chunks
from copilot.storage.database import (
    create_engine_from_url,
    create_session_factory,
    initialize_database,
)
from copilot.storage.models import SourceChunkRecord
from copilot.retrieval.pgvector import (
    retrieve_relevant_chunks_from_pgvector,
)
from copilot.api.app import create_app
from copilot.api.settings import ApiSettings


@pytest.mark.integration
def test_pgvector_storage_and_retrieval_flow(
    test_database_url: str,
):
    engine = create_engine_from_url(test_database_url)
    initialize_database(engine)
    
    SessionFactory = create_session_factory(engine)
    
    with SessionFactory() as session:
        session.execute(delete(SourceChunkRecord))
        session.commit()
        
    chunks = [
        SourceChunk(
            chunk_id="chunk-1",
            source_id="api.py",
            project_name="anomaly-detection",
            source_type="python",
            source_path="api.py",
            chunk_index=0,
            content="The prediction API exposes a /predict endpoint.",
            start_line=1,
            end_line=9,
        ),
        SourceChunk(
            chunk_id="chunk-2",
            source_id="dashboard.py",
            project_name="anomaly-detection",
            source_type="python",
            source_path="dashboard.py",
            chunk_index=0,
            content="The Streamlit dashboard shows the outcomes of the API requests.",
            start_line=10,
            end_line=20,
        ),
    ]
    
    with SessionFactory() as session:
        stored_chunks = store_source_chunks(session, chunks)
        selected_chunks = retrieve_relevant_chunks_from_pgvector(session, "Where is the prediction API endpoint?", 2)
        
    assert stored_chunks == 2
    assert len(selected_chunks) == 2
    assert selected_chunks[0].score >= selected_chunks[1].score
    assert selected_chunks[0].chunk.chunk_id == "chunk-1"
    assert selected_chunks[0].chunk.source_path == "api.py"
    assert selected_chunks[0].chunk.start_line == 1
    assert selected_chunks[0].chunk.end_line == 9
    
    
@pytest.mark.integration
def test_pgvector_query_api_flow(
    test_database_url: str,
):
    engine = create_engine_from_url(test_database_url)
    initialize_database(engine)
    
    SessionFactory = create_session_factory(engine)
    
    with SessionFactory() as session:
        session.execute(delete(SourceChunkRecord))
        session.commit()
        
    chunks = [
        SourceChunk(
            chunk_id="chunk-1",
            source_id="api.py",
            project_name="anomaly-detection",
            source_type="python",
            source_path="api.py",
            chunk_index=0,
            content="The prediction API exposes a /predict endpoint.",
            start_line=1,
            end_line=9,
        ),
        SourceChunk(
            chunk_id="chunk-2",
            source_id="dashboard.py",
            project_name="anomaly-detection",
            source_type="python",
            source_path="dashboard.py",
            chunk_index=0,
            content="The Streamlit dashboard shows the outcomes of the API requests.",
            start_line=10,
            end_line=20,
        ),
    ]
    
    with SessionFactory() as session:
        stored_chunks = store_source_chunks(session, chunks)
        
    assert stored_chunks == 2
    
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url=test_database_url,
    )
    
    test_app = create_app(settings)
    
    with TestClient(test_app) as client:
        response = client.post(
            "/query",
            json={
                "query": "Where is the prediction API endpoint?",
                "top_k": 1,
                "min_score": 0.0,
                "show_context": True,
            },
        )
    
    assert response.status_code == 200
    
    payload = response.json()
    
    assert payload["answer"]
    assert payload["citations"]
    assert payload["context"] is not None
    assert payload["context_snippets"]
    
    citation = payload["citations"][0]
    snippet = payload["context_snippets"][0]
    
    assert citation["source_path"] == "api.py"
    assert citation["start_line"] == 1
    assert citation["end_line"] == 9
    
    assert snippet["source_path"] == "api.py"
    assert snippet["content"] == (
        "The prediction API exposes a /predict endpoint."
    )