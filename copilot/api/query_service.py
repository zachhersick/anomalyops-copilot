from pathlib import Path

from copilot.schemas.query import QueryRequest, QueryResponse
from copilot.answering.grounded import build_grounded_answer
from copilot.ingestion.manifest import load_chunk_manifest
from copilot.retrieval.search import retrieve_relevant_chunks


def query_service(manifest_path: Path | None, query_request: QueryRequest) -> QueryResponse:
    if manifest_path is None:
        raise ValueError("Manifest path is not configured.")
        
    source_chunks = load_chunk_manifest(manifest_path)
    selected_chunks = retrieve_relevant_chunks(
        query=query_request.query,
        chunks=source_chunks,
        top_k=query_request.top_k,
    )
    grounded_answer = build_grounded_answer(
        query=query_request.query,
        scored_chunks=selected_chunks,
        min_score=query_request.min_score,
    )
    
    return QueryResponse(
        answer=grounded_answer.answer,
        confidence=grounded_answer.confidence,
        citations=grounded_answer.citations,
        refusal_reason=grounded_answer.refusal_reason,
    )