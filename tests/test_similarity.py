import pytest

from copilot.retrieval.similarity import cosine_similarity


def test_cosine_similarity_same_vector_is_one():
    similarity = cosine_similarity(left=[1.0, 2.0], right=[1.0, 2.0])
    
    assert similarity == pytest.approx(1.0)
    
    
def test_cosine_similarity_orthogonal_vectors_is_zero():
    similarity = cosine_similarity(left=[1.0, 0.0], right=[0.0, 1.0])
    
    assert similarity == pytest.approx(0.0)
    
    
def test_cosine_similarity_opposite_vectors_is_negative_one():
    similarity = cosine_similarity(left=[-1.0, 0.0], right=[1.0, 0.0])
    
    assert similarity == pytest.approx(-1.0)
    
    
def test_cosine_similarity_rejects_different_length_vectors():
    with pytest.raises(ValueError):
        cosine_similarity(left=[0.0, 1.0], right=[1.0])
        
        
def test_cosine_similarity_rejects_zero_left_vector():
    with pytest.raises(ValueError):
        cosine_similarity(left=[0.0, 0.0], right=[1.0, 0.0])
        
        
def test_cosine_similarity_rejects_zero_right_vector():
    with pytest.raises(ValueError):
        cosine_similarity(left=[1.0, 0.0], right=[0.0, 0.0])