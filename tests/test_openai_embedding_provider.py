from dataclasses import dataclass

import pytest
from openai import OpenAIError

from copilot.providers.errors import (
    EmbeddingConfigurationError,
    InvalidEmbeddingResponseError,
    EmbeddingProviderError,
)
from copilot.providers.interfaces import EmbeddingProvider
from copilot.providers.openai_embeddings import (
    OpenAIEmbeddingProvider,
)


@dataclass
class FakeEmbeddingItem:
    index: int
    embedding: list[float] | tuple[float, ...]


@dataclass
class FakeEmbeddingResponse:
    data: list[FakeEmbeddingItem]


class FakeEmbeddingsResource:
    def __init__(
        self,
        response: FakeEmbeddingResponse,
        error: OpenAIError | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(
        self,
        **kwargs: object,
    ) -> FakeEmbeddingResponse:
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        return self.response


class FakeOpenAIClient:
    def __init__(
        self,
        response: FakeEmbeddingResponse,
        error: OpenAIError | None = None,
    ) -> None:
        self.embeddings = FakeEmbeddingsResource(
            response=response,
            error=error,
        )


def test_openai_embedding_provider_satisfies_protocol():
    provider = OpenAIEmbeddingProvider(
        client=object(),
        model_name="text-embedding-test",
        dimensions=3,
    )

    assert isinstance(provider, EmbeddingProvider)


def test_openai_embedding_provider_exposes_metadata():
    provider = OpenAIEmbeddingProvider(
        client=object(),
        model_name="text-embedding-test",
        dimensions=3,
    )

    assert provider.provider_name == "openai"
    assert provider.model_name == "text-embedding-test"
    assert provider.dimensions == 3


@pytest.mark.parametrize(
    "model_name",
    ["", "   "],
)
def test_openai_embedding_provider_rejects_invalid_model_name(
    model_name,
):
    with pytest.raises(
        EmbeddingConfigurationError,
        match="model_name cannot be empty or whitespace-only",
    ):
        OpenAIEmbeddingProvider(
            client=object(),
            model_name=model_name,
            dimensions=3,
        )


@pytest.mark.parametrize(
    "dimensions",
    [0, -1, -10],
)
def test_openai_embedding_provider_rejects_invalid_dimensions(
    dimensions,
):
    with pytest.raises(
        EmbeddingConfigurationError,
        match="Dimensions must be positive",
    ):
        OpenAIEmbeddingProvider(
            client=object(),
            model_name="text-embedding-test",
            dimensions=dimensions,
        )


def test_embed_query_sends_expected_request_arguments():
    client = FakeOpenAIClient(
        FakeEmbeddingResponse(
            data=[
                FakeEmbeddingItem(
                    index=0,
                    embedding=[0.1, 0.2, 0.3],
                )
            ]
        )
    )
    provider = OpenAIEmbeddingProvider(
        client=client,
        model_name="text-embedding-test",
        dimensions=3,
    )

    provider.embed_query("temperature alert")

    assert client.embeddings.calls == [
        {
            "model": "text-embedding-test",
            "input": "temperature alert",
            "dimensions": 3,
        }
    ]


def test_embed_query_returns_embedding_as_plain_list():
    client = FakeOpenAIClient(
        FakeEmbeddingResponse(
            data=[
                FakeEmbeddingItem(
                    index=0,
                    embedding=(0.1, 0.2, 0.3),
                )
            ]
        )
    )
    provider = OpenAIEmbeddingProvider(
        client=client,
        model_name="text-embedding-test",
        dimensions=3,
    )

    result = provider.embed_query("temperature alert")

    assert result == [0.1, 0.2, 0.3]
    assert isinstance(result, list)
    assert len(result) == 3


@pytest.mark.parametrize(
    "text",
    ["", "   "],
)
def test_embed_query_rejects_empty_text_without_calling_provider(
    text,
):
    client = FakeOpenAIClient(
        FakeEmbeddingResponse(data=[])
    )
    provider = OpenAIEmbeddingProvider(
        client=client,
        model_name="text-embedding-test",
        dimensions=3,
    )

    with pytest.raises(
        EmbeddingConfigurationError,
        match="text cannot be empty or whitespace-only",
    ):
        provider.embed_query(text)

    assert client.embeddings.calls == []


@pytest.mark.parametrize(
    "response_data",
    [
        [],
        [
            FakeEmbeddingItem(
                index=0,
                embedding=[0.1, 0.2, 0.3],
            ),
            FakeEmbeddingItem(
                index=1,
                embedding=[0.4, 0.5, 0.6],
            ),
        ],
    ],
)
def test_embed_query_rejects_incorrect_response_count(
    response_data,
):
    client = FakeOpenAIClient(
        FakeEmbeddingResponse(data=response_data)
    )
    provider = OpenAIEmbeddingProvider(
        client=client,
        model_name="text-embedding-test",
        dimensions=3,
    )

    with pytest.raises(
        InvalidEmbeddingResponseError,
        match="must contain exactly one vector",
    ):
        provider.embed_query("temperature alert")


def test_embed_query_rejects_incorrect_vector_dimensions():
    client = FakeOpenAIClient(
        FakeEmbeddingResponse(
            data=[
                FakeEmbeddingItem(
                    index=0,
                    embedding=[0.1, 0.2],
                )
            ]
        )
    )
    provider = OpenAIEmbeddingProvider(
        client=client,
        model_name="text-embedding-test",
        dimensions=3,
    )

    with pytest.raises(
        InvalidEmbeddingResponseError,
        match="incorrect dimensions",
    ):
        provider.embed_query("temperature alert")
        
        
def test_embed_documents_returns_empty_list_without_calling_provider():
    client = FakeOpenAIClient(
        FakeEmbeddingResponse(data=[])
    )
    provider = OpenAIEmbeddingProvider(
        client=client,
        model_name="text-embedding-test",
        dimensions=3,
    )

    result = provider.embed_documents([])

    assert result == []
    assert client.embeddings.calls == []


@pytest.mark.parametrize(
    "texts",
    [
        ["temperature alert", ""],
        ["temperature alert", "   "],
    ],
)
def test_embed_documents_rejects_invalid_text_without_calling_provider(
    texts,
):
    client = FakeOpenAIClient(
        FakeEmbeddingResponse(data=[])
    )
    provider = OpenAIEmbeddingProvider(
        client=client,
        model_name="text-embedding-test",
        dimensions=3,
    )

    with pytest.raises(
        EmbeddingConfigurationError,
        match="Documents cannot be empty or whitespace-only",
    ):
        provider.embed_documents(texts)

    assert client.embeddings.calls == []


def test_embed_documents_sends_all_documents_in_one_request():
    client = FakeOpenAIClient(
        FakeEmbeddingResponse(
            data=[
                FakeEmbeddingItem(
                    index=0,
                    embedding=[0.1, 0.2, 0.3],
                ),
                FakeEmbeddingItem(
                    index=1,
                    embedding=[0.4, 0.5, 0.6],
                ),
            ]
        )
    )
    provider = OpenAIEmbeddingProvider(
        client=client,
        model_name="text-embedding-test",
        dimensions=3,
    )
    texts = [
        "temperature alert",
        "pressure warning",
    ]

    provider.embed_documents(texts)

    assert client.embeddings.calls == [
        {
            "model": "text-embedding-test",
            "input": texts,
            "dimensions": 3,
        }
    ]


def test_embed_documents_preserves_input_order_using_response_indices():
    client = FakeOpenAIClient(
        FakeEmbeddingResponse(
            data=[
                FakeEmbeddingItem(
                    index=2,
                    embedding=[0.7, 0.8, 0.9],
                ),
                FakeEmbeddingItem(
                    index=0,
                    embedding=[0.1, 0.2, 0.3],
                ),
                FakeEmbeddingItem(
                    index=1,
                    embedding=[0.4, 0.5, 0.6],
                ),
            ]
        )
    )
    provider = OpenAIEmbeddingProvider(
        client=client,
        model_name="text-embedding-test",
        dimensions=3,
    )

    result = provider.embed_documents(
        [
            "temperature alert",
            "pressure warning",
            "vibration anomaly",
        ]
    )

    assert result == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]


def test_embed_documents_returns_vectors_as_plain_lists():
    client = FakeOpenAIClient(
        FakeEmbeddingResponse(
            data=[
                FakeEmbeddingItem(
                    index=0,
                    embedding=(0.1, 0.2, 0.3),
                ),
                FakeEmbeddingItem(
                    index=1,
                    embedding=(0.4, 0.5, 0.6),
                ),
            ]
        )
    )
    provider = OpenAIEmbeddingProvider(
        client=client,
        model_name="text-embedding-test",
        dimensions=3,
    )

    result = provider.embed_documents(
        [
            "temperature alert",
            "pressure warning",
        ]
    )

    assert all(
        isinstance(vector, list)
        for vector in result
    )


def test_embed_documents_rejects_incorrect_response_count():
    client = FakeOpenAIClient(
        FakeEmbeddingResponse(
            data=[
                FakeEmbeddingItem(
                    index=0,
                    embedding=[0.1, 0.2, 0.3],
                ),
                FakeEmbeddingItem(
                    index=1,
                    embedding=[0.4, 0.5, 0.6],
                ),
            ]
        )
    )
    provider = OpenAIEmbeddingProvider(
        client=client,
        model_name="text-embedding-test",
        dimensions=3,
    )

    with pytest.raises(
        InvalidEmbeddingResponseError,
        match="response count does not match input count",
    ):
        provider.embed_documents(
            [
                "temperature alert",
                "pressure warning",
                "vibration anomaly",
            ]
        )


@pytest.mark.parametrize(
    "response_indices",
    [
        [0, 0],
        [0, 2],
    ],
)
def test_embed_documents_rejects_invalid_response_indices(
    response_indices,
):
    client = FakeOpenAIClient(
        FakeEmbeddingResponse(
            data=[
                FakeEmbeddingItem(
                    index=response_indices[0],
                    embedding=[0.1, 0.2, 0.3],
                ),
                FakeEmbeddingItem(
                    index=response_indices[1],
                    embedding=[0.4, 0.5, 0.6],
                ),
            ]
        )
    )
    provider = OpenAIEmbeddingProvider(
        client=client,
        model_name="text-embedding-test",
        dimensions=3,
    )

    with pytest.raises(
        InvalidEmbeddingResponseError,
        match="response contains invalid indices",
    ):
        provider.embed_documents(
            [
                "temperature alert",
                "pressure warning",
            ]
        )


def test_embed_documents_rejects_incorrect_vector_dimensions():
    client = FakeOpenAIClient(
        FakeEmbeddingResponse(
            data=[
                FakeEmbeddingItem(
                    index=0,
                    embedding=[0.1, 0.2, 0.3],
                ),
                FakeEmbeddingItem(
                    index=1,
                    embedding=[0.4, 0.5],
                ),
            ]
        )
    )
    provider = OpenAIEmbeddingProvider(
        client=client,
        model_name="text-embedding-test",
        dimensions=3,
    )

    with pytest.raises(
        InvalidEmbeddingResponseError,
        match="incorrect dimensions",
    ):
        provider.embed_documents(
            [
                "temperature alert",
                "pressure warning",
            ]
        )
        
        
def test_embed_query_converts_openai_error_to_provider_error():
    original_error = OpenAIError("test failure")
    client = FakeOpenAIClient(
        response=FakeEmbeddingResponse(data=[]),
        error=original_error,
    )
    provider = OpenAIEmbeddingProvider(
        client=client,
        model_name="text-embedding-test",
        dimensions=3,
    )

    with pytest.raises(
        EmbeddingProviderError,
        match="Embedding provider returned an error",
    ) as exc_info:
        provider.embed_query("temperature alert")

    assert exc_info.value.__cause__ is original_error


def test_embed_documents_converts_openai_error_to_provider_error():
    original_error = OpenAIError("test failure")
    client = FakeOpenAIClient(
        response=FakeEmbeddingResponse(data=[]),
        error=original_error,
    )
    provider = OpenAIEmbeddingProvider(
        client=client,
        model_name="text-embedding-test",
        dimensions=3,
    )

    with pytest.raises(
        EmbeddingProviderError,
        match="Embedding provider returned an error",
    ) as exc_info:
        provider.embed_documents(
            [
                "temperature alert",
                "pressure warning",
            ]
        )

    assert exc_info.value.__cause__ is original_error