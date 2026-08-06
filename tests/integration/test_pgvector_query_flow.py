import pytest

from fastapi.testclient import TestClient

from copilot.schemas.chunk import SourceChunk
from copilot.storage.chunks import store_source_chunks
from copilot.storage.database import (
    create_engine_from_url,
    create_session_factory,
    initialize_database,
    rebuild_source_chunks_table,
)
from copilot.storage.models import EMBEDDING_DIMENSIONS
from copilot.retrieval.pgvector import (
    retrieve_relevant_chunks_from_pgvector,
)
from copilot.api.app import create_app
from copilot.api.settings import ApiSettings
from copilot.providers.deterministic_embeddings import (
    DeterministicEmbeddingProvider
)
from copilot.api.query_service import query_service

from copilot.providers.deterministic_answers import (
    DeterministicGroundedAnswerGenerator,
)
from copilot.schemas.query import QueryRequest
from scripts.run_rag_evals import validate_semantic_index


@pytest.mark.integration
def test_pgvector_storage_and_retrieval_flow(
    test_database_url: str,
):
    engine = create_engine_from_url(test_database_url)
    initialize_database(engine)
    
    rebuild_source_chunks_table(engine)
    
    provider = DeterministicEmbeddingProvider(EMBEDDING_DIMENSIONS)
    
    SessionFactory = create_session_factory(engine)
        
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
        stored_chunks = store_source_chunks(
            session=session,
            chunks=chunks,
            embedding_provider=provider,
        )
        selected_chunks = retrieve_relevant_chunks_from_pgvector(
            session=session,
            query="The prediction API exposes a /predict endpoint.",
            embedding_provider=provider,
            top_k=2,
        )
        
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
    
    rebuild_source_chunks_table(engine)
    
    provider = DeterministicEmbeddingProvider(EMBEDDING_DIMENSIONS)
    
    SessionFactory = create_session_factory(engine)
        
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
        stored_chunks = store_source_chunks(
            session=session,
            chunks=chunks,
            embedding_provider=provider,
        )
        
    assert stored_chunks == 2
    
    settings = ApiSettings(
        retrieval_backend="pgvector",
        database_url=test_database_url,
        ai_provider="deterministic",
        embedding_dimensions=EMBEDDING_DIMENSIONS,
    )
    
    test_app = create_app(settings)
    
    with TestClient(test_app) as client:
        response = client.post(
            "/query",
            json={
                "query": "The prediction API exposes a /predict endpoint.",
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
    
    
@pytest.mark.integration
def test_pgvector_query_flow_returns_deterministic_grounded_answer(
    test_database_url: str,
):
    engine = create_engine_from_url(test_database_url)
    initialize_database(engine)
    rebuild_source_chunks_table(engine)

    embedding_provider = DeterministicEmbeddingProvider(
        EMBEDDING_DIMENSIONS
    )
    answer_generator = DeterministicGroundedAnswerGenerator()
    session_factory = create_session_factory(engine)

    chunk = SourceChunk(
        chunk_id="chunk-1",
        source_id="api.py",
        project_name="anomaly-detection",
        source_type="python",
        source_path="api.py",
        chunk_index=0,
        content="The prediction API exposes POST /predict.",
        start_line=10,
        end_line=20,
    )

    with session_factory() as session:
        stored_chunks = store_source_chunks(
            session=session,
            chunks=[chunk],
            embedding_provider=embedding_provider,
        )

    assert stored_chunks == 1

    validate_semantic_index(
        session_factory,
        [chunk],
        embedding_provider,
    )

    response = query_service(
        settings=ApiSettings(
            retrieval_backend="pgvector",
            database_url=test_database_url,
            ai_provider="deterministic",
            embedding_dimensions=EMBEDDING_DIMENSIONS,
        ),
        query_request=QueryRequest(
            query="The prediction API exposes POST /predict.",
            top_k=1,
            min_score=0.0,
            show_context=True,
        ),
        session_factory=session_factory,
        embedding_provider=embedding_provider,
        grounded_answer_generator=answer_generator,
    )

    assert response.refusal_reason is None
    assert response.answer == (
        "The retrieved context says: "
        "The prediction API exposes POST /predict. [1]"
    )
    assert response.confidence == pytest.approx(1.0)

    assert len(response.citations) == 1

    citation = response.citations[0]

    assert citation.citation_id == 1
    assert citation.source_path == "api.py"
    assert citation.start_line == 10
    assert citation.end_line == 20

    assert response.context is not None
    assert len(response.context_snippets) == 1

    snippet = response.context_snippets[0]

    assert snippet.citation_id == 1
    assert snippet.source_path == "api.py"
    assert snippet.start_line == 10
    assert snippet.end_line == 20
    assert snippet.content == (
        "The prediction API exposes POST /predict."
    )

    engine.dispose()
