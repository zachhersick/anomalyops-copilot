import pytest

from copilot.retrieval.embeddings import embed_text


def test_embed_text_is_deterministic():
    first_vector = embed_text("same text")
    second_vector = embed_text("same text")
    
    assert first_vector == second_vector
    
    
def test_embed_text_returns_requested_dimensions():
    vector = embed_text("text", dimensions=6)
    
    assert len(vector) == 6
    
    
def test_embed_text_different_text_returns_different_vector():
    first_vector = embed_text("same text")
    second_vector = embed_text("different vector")
    
    assert first_vector != second_vector
    
    
def test_embed_text_returns_float_values():
    vector = embed_text("text")
    
    assert all(isinstance(value, float) for value in vector)
    
    
def test_embed_text_rejects_non_positive_dimensions():
    with pytest.raises(ValueError):
        embed_text("text", dimensions=0)
