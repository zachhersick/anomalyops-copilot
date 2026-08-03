from collections.abc import Sequence
from openai import OpenAI, OpenAIError

from copilot.providers.errors import (
    EmbeddingProviderError,
    EmbeddingConfigurationError,
    InvalidEmbeddingResponseError,
)
from copilot.observability import trace_span


class OpenAIEmbeddingProvider:
    provider_name = "openai"
    
    def __init__(
        self,
        model_name: str,
        dimensions: int,
        client: OpenAI,
    ) -> None:
        if not model_name.strip():
            raise EmbeddingConfigurationError("model_name cannot be empty or whitespace-only.")
        
        if dimensions <= 0:
            raise EmbeddingConfigurationError("Dimensions must be positive.")
        
        self.model_name = model_name
        self.dimensions = dimensions
        self._client = client
        
        
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        if not text.strip():
            raise EmbeddingConfigurationError("text cannot be empty or whitespace-only.")
        
        try:
            with trace_span(
                "provider.request",
                provider=self.provider_name,
                model=self.model_name,
                operation="embed_query",
                input_count=1,
                dimensions=self.dimensions,
            ):
                response = self._client.embeddings.create(
                    model=self.model_name,
                    input=text,
                    dimensions=self.dimensions,
                )
        except OpenAIError as exc:
            raise EmbeddingProviderError(
                "Embedding provider returned an error."
            ) from exc
            
        if len(response.data) != 1:
            raise InvalidEmbeddingResponseError("Embedding response must contain exactly one vector.")
        
        vector = list(response.data[0].embedding)
        
        if len(vector) != self.dimensions:
            raise InvalidEmbeddingResponseError("Embedding vector has incorrect dimensions.")

        return vector
    
    
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        if not texts:
            return []
        for text in texts:
            if not text.strip():
                raise EmbeddingConfigurationError("Documents cannot be empty or whitespace-only.")
         
        try:
            with trace_span(
                "provider.request",
                provider=self.provider_name,
                model=self.model_name,
                operation="embed_documents",
                input_count=len(texts),
                dimensions=self.dimensions,
            ):
                response = self._client.embeddings.create(
                    model=self.model_name,
                    input=list(texts),
                    dimensions=self.dimensions,
                )
        except OpenAIError as exc:
            raise EmbeddingProviderError(
                "Embedding provider returned an error."
            ) from exc
        
        if len(response.data) != len(texts):
            raise InvalidEmbeddingResponseError("Embedding response count does not match input count.")
        
        index_sorted = sorted(response.data, key=lambda item: item.index)
        
        expected_indices = list(range(len(texts)))
        actual_indices = [
            item.index
            for item in index_sorted
        ]
        
        if actual_indices != expected_indices:
            raise InvalidEmbeddingResponseError(
                "Embedding response contains invalid indices."
            )
        
        embeddings: list[list[float]] = []
        
        for item in index_sorted:
            vector = list(item.embedding)
            
            if len(vector) != self.dimensions:
                raise InvalidEmbeddingResponseError("Embedding vector has incorrect dimensions.")
            
            embeddings.append(vector)
            
        return embeddings
