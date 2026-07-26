import pytest

from copilot.providers.deterministic_embeddings import (
    DeterministicEmbeddingProvider,
)
from copilot.providers.interfaces import EmbeddingProvider


def test_deterministic_embedding_provider_satisfies_protocol():
    provider = DeterministicEmbeddingProvider(
        dimensions=16,
    )

    assert isinstance(provider, EmbeddingProvider)


def test_deterministic_embedding_provider_exposes_metadata():
    provider = DeterministicEmbeddingProvider(
        dimensions=24,
    )

    assert provider.provider_name == "deterministic"
    assert provider.model_name == "sha256-hashing"
    assert provider.dimensions == 24


@pytest.mark.parametrize(
    "dimensions",
    [0, -1, -16],
)
def test_deterministic_embedding_provider_rejects_invalid_dimensions(
    dimensions,
):
    with pytest.raises(
        ValueError,
        match="Dimensions must be positive",
    ):
        DeterministicEmbeddingProvider(
            dimensions=dimensions,
        )


def test_embed_query_is_repeatable():
    provider = DeterministicEmbeddingProvider(
        dimensions=16,
    )

    first_result = provider.embed_query(
        "temperature alert",
    )
    second_result = provider.embed_query(
        "temperature alert",
    )

    assert first_result == second_result


def test_embed_query_uses_configured_dimensions():
    provider = DeterministicEmbeddingProvider(
        dimensions=24,
    )

    result = provider.embed_query(
        "temperature alert",
    )

    assert len(result) == 24


def test_embed_documents_returns_one_vector_per_document():
    provider = DeterministicEmbeddingProvider(
        dimensions=16,
    )
    texts = [
        "temperature alert",
        "pressure warning",
        "vibration anomaly",
    ]

    result = provider.embed_documents(texts)

    assert len(result) == len(texts)


def test_embed_documents_preserves_document_order():
    provider = DeterministicEmbeddingProvider(
        dimensions=16,
    )
    texts = [
        "temperature alert",
        "pressure warning",
        "vibration anomaly",
    ]

    result = provider.embed_documents(texts)

    assert result == [
        provider.embed_query(text)
        for text in texts
    ]


def test_embed_documents_uses_configured_dimensions():
    provider = DeterministicEmbeddingProvider(
        dimensions=24,
    )

    result = provider.embed_documents(
        [
            "temperature alert",
            "pressure warning",
        ]
    )

    assert all(
        len(vector) == 24
        for vector in result
    )


def test_embed_documents_returns_empty_list_for_empty_input():
    provider = DeterministicEmbeddingProvider(
        dimensions=16,
    )

    result = provider.embed_documents([])

    assert result == []