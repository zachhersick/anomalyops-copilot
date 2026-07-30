from collections.abc import Sequence

from copilot.schemas.retrieval import ScoredChunk
from copilot.schemas.answer import Citation, GroundedAnswer
from copilot.providers.errors import (
    GroundedAnswerConfigurationError,
)


class DeterministicGroundedAnswerGenerator:
    provider_name = "deterministic"
    model_name = "retrieval-context-template-v1"
    
    def generate(
        self,
        query: str,
        context: Sequence[ScoredChunk],
    ) -> GroundedAnswer:
        if not query.strip():
            raise GroundedAnswerConfigurationError(
                "query cannot be blank or whitespace-only."
            )
        if not context:
            raise GroundedAnswerConfigurationError(
                "context cannot be empty."
            )
            
        pairs = [(citation_id, scored_chunk) for citation_id, scored_chunk in enumerate(context, start=1)]
        
        citation_id, selected_chunk = max(
            pairs,
            key=lambda item: item[1].score,
        )
        
        citation = Citation(
            citation_id=citation_id,
            source_path=selected_chunk.chunk.source_path,
            start_line=selected_chunk.chunk.start_line,
            end_line=selected_chunk.chunk.end_line,
        )
        
        return GroundedAnswer(
            answer=f"The retrieved context says: {selected_chunk.chunk.content} [{citation_id}]",
            citations=[citation],
            confidence=max(
                0.0,
                min(selected_chunk.score, 1.0),
            ),
            refusal_reason=None,
        )