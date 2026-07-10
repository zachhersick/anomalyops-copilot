from pathlib import Path
from json import JSONDecodeError
from pydantic import ValidationError

from copilot.schemas.query import QueryRequest, QueryResponse, ContextSnippet
from copilot.answering.grounded import build_grounded_answer
from copilot.ingestion.manifest import load_chunk_manifest
from copilot.retrieval.search import retrieve_relevant_chunks
from copilot.retrieval.context import format_retrieval_context
from copilot.api.errors import (
    ManifestNotConfiguredError,
    ManifestFileNotFoundError,
    InvalidManifestError
)


def query_service(manifest_path: Path | None, query_request: QueryRequest) -> QueryResponse:
    if manifest_path is None:
        raise ManifestNotConfiguredError("Manifest path is not configured.")
    
    if not manifest_path.is_file():
        raise ManifestFileNotFoundError(f"{manifest_path} does not exist.")
    
    try:
        source_chunks = load_chunk_manifest(manifest_path)
    except (JSONDecodeError, ValidationError) as exc:
        raise InvalidManifestError("Manifest file is invalid.") from exc
        
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
    
    context_snippets = []
    
    if query_request.show_context:
        context = format_retrieval_context(selected_chunks)
        for i, selected_chunk in enumerate(selected_chunks, start=1):
            chunk = selected_chunk.chunk
            context_snippets.append(
                ContextSnippet(
                    citation_id=i,
                    source_path=chunk.source_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    content=chunk.content,
                    score=selected_chunk.score,
                )
            )
    else:
        context = None
    
    return QueryResponse(
        answer=grounded_answer.answer,
        confidence=grounded_answer.confidence,
        citations=grounded_answer.citations,
        refusal_reason=grounded_answer.refusal_reason,
        context=context,
        context_snippets=context_snippets,
    )