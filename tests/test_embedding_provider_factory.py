import pytest

from copilot.api.settings import ApiSettings
from copilot.providers.deterministic_embeddings import (
    DeterministicEmbeddingProvider,
)
from copilot.providers.errors import EmbeddingConfigurationError
from copilot.providers.factory import create_embedding_provider
from copilot.providers.interfaces import EmbeddingProvider
from copilot.providers.openai_embeddings import (
    OpenAIEmbeddingProvider,
)


def test_create_embedding_provider_returns_deterministic_provider():
    settings = ApiSettings(
        ai_provider="deterministic",
        embedding_dimensions=24,
    )

    provider = create_embedding_provider(settings)

    assert isinstance(provider, DeterministicEmbeddingProvider)
    assert isinstance(provider, EmbeddingProvider)
    assert provider.dimensions == 24


def test_create_embedding_provider_returns_openai_provider_with_injected_client():
    injected_client = object()
    settings = ApiSettings(
        ai_provider="openai",
        openai_api_key="test-api-key",
        embedding_model="text-embedding-test",
        embedding_dimensions=1536,
    )

    provider = create_embedding_provider(
        settings,
        openai_client=injected_client,
    )

    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert isinstance(provider, EmbeddingProvider)
    assert provider._client is injected_client
    assert provider.model_name == "text-embedding-test"
    assert provider.dimensions == 1536


@pytest.mark.parametrize(
    "api_key",
    [None, "", "   "],
)
def test_create_embedding_provider_rejects_missing_openai_api_key(
    api_key,
):
    settings = ApiSettings(
        ai_provider="openai",
        openai_api_key=api_key,
        embedding_model="text-embedding-test",
        embedding_dimensions=1536,
    )

    with pytest.raises(
        EmbeddingConfigurationError,
        match=(
            "OPENAI_API_KEY is required for the "
            "OpenAI embedding provider"
        ),
    ):
        create_embedding_provider(
            settings,
            openai_client=object(),
        )


@pytest.mark.parametrize(
    "embedding_model",
    [None, "", "   "],
)
def test_create_embedding_provider_rejects_missing_embedding_model(
    embedding_model,
):
    settings = ApiSettings(
        ai_provider="openai",
        openai_api_key="test-api-key",
        embedding_model=embedding_model,
        embedding_dimensions=1536,
    )

    with pytest.raises(
        EmbeddingConfigurationError,
        match=(
            "ANOMALYOPS_EMBEDDING_MODEL is required for the "
            "OpenAI embedding provider"
        ),
    ):
        create_embedding_provider(
            settings,
            openai_client=object(),
        )


def test_create_embedding_provider_rejects_unsupported_provider():
    settings = ApiSettings.model_construct(
        ai_provider="unsupported",
        embedding_dimensions=16,
        openai_api_key=None,
        embedding_model=None,
    )

    with pytest.raises(
        EmbeddingConfigurationError,
        match="Unsupported embedding provider: unsupported",
    ):
        create_embedding_provider(settings)