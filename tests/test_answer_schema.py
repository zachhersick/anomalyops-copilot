import pytest
from pydantic import ValidationError

from copilot.schemas.answer import Citation, GroundedAnswer


def test_create_grounded_answer_with_citation(tmp_path):
    citation = make_citation()
    grounded_answer = make_grounded_answer(
        answer="api exposes a prediction endpoint",
        citations=[citation],
        confidence=0.9,
    )
    
    assert grounded_answer.answer == "api exposes a prediction endpoint"
    assert len(grounded_answer.citations) == 1
    assert grounded_answer.citations[0].source_path == "source.py"
    assert grounded_answer.citations[0].start_line == 1
    assert grounded_answer.citations[0].end_line == 2
    assert grounded_answer.confidence == 0.9
    
    
def test_grounded_answer_refusal_reason_defaults_to_none(tmp_path):
    citation = make_citation()
    grounded_answer = make_grounded_answer(
        answer="api exposes a prediction endpoint",
        citations=[citation],
        confidence=0.9,
    )
    
    assert grounded_answer.refusal_reason is None
    
    
def test_grounded_answer_rejects_confidence_below_zero(tmp_path):
    citation = make_citation()
    
    with pytest.raises(ValidationError):
        make_grounded_answer(
            answer="api exposes a prediction endpoint",
            citations=[citation],
            confidence=-0.1,
        )
        
        
def test_grounded_answer_rejects_confidence_above_one(tmp_path):
    citation = make_citation()
    
    with pytest.raises(ValidationError):
        make_grounded_answer(
            answer="api exposes a prediction endpoint",
            citations=[citation],
            confidence=1.1,
        )
        
        
def test_citation_stores_source_path_and_line_range(tmp_path):
    citation = make_citation(
        source_path="source.py",
        start_line=10,
        end_line=20,
    )
    
    assert citation.source_path == "source.py"
    assert citation.start_line == 10
    assert citation.end_line == 20
    
    
def make_citation(
    citation_id: int = 1,
    source_path: str = "source.py",
    start_line: int = 1,
    end_line: int = 2,
) -> Citation:
    return Citation(
        citation_id=citation_id,
        source_path=source_path,
        start_line=start_line,
        end_line=end_line,
    )
    
    
def make_grounded_answer(
    answer: str,
    citations: list[Citation],
    confidence: float,
    refusal_reason: str | None = None,
) -> GroundedAnswer:
    return GroundedAnswer(
        answer=answer,
        citations=citations,
        confidence=confidence,
        refusal_reason=refusal_reason,
    )