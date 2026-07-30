from copilot.schemas.retrieval import ScoredChunk
from copilot.schemas.answer import GroundedAnswer
from copilot.providers.deterministic_answers import DeterministicGroundedAnswerGenerator
from copilot.providers.interfaces import GroundedAnswerGenerator


def build_grounded_answer(
    query: str,
    scored_chunks: list[ScoredChunk],
    min_score: float = 0.0,
    *,
    generator: GroundedAnswerGenerator | None = None,
) -> GroundedAnswer:
    if not scored_chunks:
        return GroundedAnswer(
            answer="",
            citations=[],
            confidence=0.0,
            refusal_reason="No relevant context was retrieved.",
        )
    
    top_score = max(scored_chunk.score for scored_chunk in scored_chunks)
    clamped_score = max(0.0, min(top_score, 1.0))
    
    if top_score < min_score:
        return GroundedAnswer(
            answer="",
            citations=[],
            confidence=clamped_score,
            refusal_reason="Retrieved context was below the confidence threshold.",
        )
    
    generator = (
        generator
        if generator is not None
        else DeterministicGroundedAnswerGenerator()
    )
    
    return generator.generate(
        query=query,
        context=scored_chunks,
    )