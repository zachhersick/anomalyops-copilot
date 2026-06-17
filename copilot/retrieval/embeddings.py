import hashlib


def embed_text(text: str, dimensions: int = 16) -> list[float]:
    if dimensions <= 0:
        raise ValueError("Dimensions must be greater than 0")
    
    all_bytes = b""
    counter = 0
    
    while len(all_bytes) < dimensions:
        hash_input = f"{text}:{counter}".encode("utf-8")
        digest = hashlib.sha256(hash_input).digest()
        all_bytes += digest
        counter += 1
        
    selected_bytes = all_bytes[:dimensions]
    
    vector = []
    
    for byte in selected_bytes:
        value = (byte / 127.5) - 1.0
        vector.append(value)
    
    return vector