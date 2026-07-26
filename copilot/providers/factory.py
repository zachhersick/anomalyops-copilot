from openai import OpenAI

from copilot.api.settings import ApiSettings
from copilot.providers.interfaces import EmbeddingProvider
from copilot.providers.deterministic_embeddings import DeterministicEmbeddingProvider
from copilot.providers.openai_embeddings import OpenAIEmbeddingProvider
from copilot.providers.errors import (
    EmbeddingConfigurationError,
)


def create_embedding_provider(
    settings: ApiSettings,
    openai_client: OpenAI | None = None,
) -> EmbeddingProvider:
    if settings.ai_provider == "deterministic":
        return DeterministicEmbeddingProvider(
            dimensions=settings.embedding_dimensions,
        )
    if settings.ai_provider == "openai":
        if (
            settings.openai_api_key is None
            or not settings.openai_api_key.strip()
        ):
            raise EmbeddingConfigurationError("OPENAI_API_KEY is required for the OpenAI embedding provider.")
        if (
            settings.embedding_model is None
            or not settings.embedding_model.strip()
        ):
            raise EmbeddingConfigurationError("ANOMALYOPS_EMBEDDING_MODEL is required for the OpenAI embedding provider.")
        
        client = (
            openai_client
            if openai_client is not None
            else OpenAI(api_key=settings.openai_api_key)
        )
        
        return OpenAIEmbeddingProvider(
            client=client,
            model_name=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    
    raise EmbeddingConfigurationError(f"Unsupported embedding provider: {settings.ai_provider}")