from copilot.retrieval.citations import build_citations_from_scored_chunks
from copilot.schemas.retrieval import ScoredChunk
from copilot.schemas.answer import GroundedAnswer


def build_grounded_answer(query: str, scored_chunks: list[ScoredChunk]) -> GroundedAnswer:
    if not scored_chunks:
        return GroundedAnswer(
            answer="",
            citations=[],
            confidence=0.0,
            refusal_reason="No relevant context was retrieved."
        )
    citations = build_citations_from_scored_chunks(scored_chunks)
    top_score = max(scored_chunk.score for scored_chunk in scored_chunks)
    clamped_score = max(0.0, min(top_score, 1.0))
    
    return GroundedAnswer(
        answer="I found relevant project context for this question.",
        citations=citations,
        confidence=clamped_score,
        refusal_reason=None,
    )