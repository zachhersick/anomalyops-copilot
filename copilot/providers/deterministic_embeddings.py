from collections.abc import Sequence

from copilot.retrieval.embeddings import embed_text


class DeterministicEmbeddingProvider:
    provider_name = "deterministic"
    model_name = "sha256-hashing"
    
    def __init__(
        self,
        dimensions: int
    ) -> None:
        if dimensions <= 0:
            raise ValueError("Dimensions must be positive.")
        
        self.dimensions = dimensions
            
        
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return embed_text(
            text,
            dimensions=self.dimensions,
        )
    
    
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [
            self.embed_query(text)
            for text in texts
        ]