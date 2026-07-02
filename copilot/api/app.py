from fastapi import FastAPI, HTTPException, Request

from copilot.schemas.query import QueryRequest, QueryResponse
from copilot.answering.grounded import build_grounded_answer
from copilot.ingestion.manifest import load_chunk_manifest
from copilot.retrieval.search import retrieve_relevant_chunks
from copilot.api.settings import ApiSettings


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    app = FastAPI(
        title="AnomalyOps-Copilot API",
        description="API RAG answers.",
        version="0.1.0",
    )
    
    app.state.settings = settings or ApiSettings()
        

    @app.get("/health")
    def health_check():
        return {"status": "ok"}


    @app.post("/query", response_model=QueryResponse)
    def query(request: Request, query_request: QueryRequest) -> QueryResponse:
        settings = request.app.state.settings
        manifest_path = settings.manifest_path
        
        if manifest_path is None:
            raise HTTPException(
                status_code=500,
                detail="Manifest path is not configured.",
            )
            
        source_chunks = load_chunk_manifest(manifest_path)
        selected_chunks = retrieve_relevant_chunks(
            query=query_request.query,
            chunks=source_chunks,
            top_k=query_request.top_k
        )
        grounded_answer = build_grounded_answer(
            query=query_request.query,
            scored_chunks=selected_chunks,
            min_score=query_request.min_score
        )
        
        return QueryResponse(
            answer=grounded_answer.answer,
            confidence=grounded_answer.confidence,
            citations=grounded_answer.citations,
            refusal_reason=grounded_answer.refusal_reason,
        )
        
    return app


app = create_app()