from json import JSONDecodeError
from pydantic import ValidationError

from copilot.schemas.query import QueryRequest, QueryResponse, ContextSnippet
from copilot.answering.grounded import build_grounded_answer
from copilot.ingestion.manifest import load_chunk_manifest
from copilot.retrieval.search import retrieve_relevant_chunks
from copilot.retrieval.context import format_retrieval_context
from copilot.retrieval.pgvector import retrieve_relevant_chunks_from_pgvector
from copilot.api.settings import ApiSettings
from copilot.schemas.retrieval import ScoredChunk
from copilot.api.errors import DatabaseNotConfiguredError
from copilot.api.errors import (
    ManifestNotConfiguredError,
    ManifestFileNotFoundError,
    InvalidManifestError
)
from copilot.storage.database import (
    create_engine_from_url,
    create_session_factory,
)


def query_service(settings: ApiSettings, query_request: QueryRequest) -> QueryResponse:
    selected_chunks = retrieve_chunks_for_query(settings, query_request)
    
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
    
    
def retrieve_chunks_for_query(
    settings: ApiSettings,
    query_request: QueryRequest,
) -> list[ScoredChunk]:
    retrieval_backend = settings.retrieval_backend
    
    if retrieval_backend == "manifest":
        if settings.manifest_path is None:
            raise ManifestNotConfiguredError("Manifest path is not configured.")
        
        if not settings.manifest_path.is_file():
            raise ManifestFileNotFoundError(f"{settings.manifest_path} does not exist.")
        
        try:
            source_chunks = load_chunk_manifest(settings.manifest_path)
        except (JSONDecodeError, ValidationError) as exc:
            raise InvalidManifestError("Manifest file is invalid.") from exc
            
        return retrieve_relevant_chunks(
            query=query_request.query,
            chunks=source_chunks,
            top_k=query_request.top_k,
        )
    
    if settings.database_url is None:
        raise DatabaseNotConfiguredError(
            "Database URL is not configured."
        )
        
    engine = create_engine_from_url(
        settings.database_url
    )
    SessionFactory = create_session_factory(engine)
        
    with SessionFactory() as session:
        return retrieve_relevant_chunks_from_pgvector(
            session=session,
            query=query_request.query,
            top_k=query_request.top_k
        )