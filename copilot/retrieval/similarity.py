def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vector lengths must be equal.")
    
    if len(left) == 0 or len(right) == 0:
        raise ValueError("Vector lengths must be positive.")
    
    left_magnitude = sum(value ** 2 for value in left) ** 0.5
    right_magnitude = sum(value ** 2 for value in right) ** 0.5
    
    if left_magnitude == 0 or right_magnitude == 0:
        raise ValueError("Magnitudes may not be 0")
    
    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right)
    )
    
    return dot_product / (left_magnitude * right_magnitude)