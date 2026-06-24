import pytest
from copilot.schemas.query import QueryRequest, QueryResponse
from copilot.schemas.answer import Citation


def test_query_request_defaults_correctly():
    query_request = make_query_request("text")
    
    assert query_request.top_k == 3
    assert query_request.min_score == 0.0
    assert query_request.show_context is False

def test_query_request_rejects_non_positive_top_k():
    with pytest.raises(ValueError):
        make_query_request("text", top_k=0)
        

def test_query_request_rejects_negative_min_score():
    with pytest.raises(ValueError):
        make_query_request("text", min_score=-0.1)
        
        
def test_query_response_accepts_answer_confidence_citations_refusal_reason():
    query_response = make_query_response(
        answer="Context found",
        confidence=0.6,
        citations=[make_citation()],
        refusal_reason=None,
    )
    
    assert query_response.answer == "Context found"
    assert query_response.confidence == 0.6
    assert query_response.citations[0].source_path == "source.py"
    assert query_response.refusal_reason is None
    

def test_query_response_rejects_confidence_above_one():
    with pytest.raises(ValueError):
        make_query_response(answer="text", confidence=1.1, citations=[])


def test_query_response_rejects_confidence_below_zero():
    with pytest.raises(ValueError):
        make_query_response(answer="text", confidence=-0.1, citations=[])
        
        
def test_query_request_rejects_empty_query():
    with pytest.raises(ValueError):
        make_query_request("")
    
    
def make_query_request(
    query: str,
    top_k: int = 3,
    min_score: float = 0.0,
    show_context: bool = False
) -> QueryRequest:
    return QueryRequest(
        query=query,
        top_k=top_k,
        min_score=min_score,
        show_context=show_context,
    )
    
    
def make_query_response(
    answer: str,
    confidence: float,
    citations: list[Citation],
    refusal_reason: str | None = None
) -> QueryResponse:
    return QueryResponse(
        answer=answer,
        confidence=confidence,
        citations=citations,
        refusal_reason=refusal_reason,
    )
    
    
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