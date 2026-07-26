class EmbeddingProviderError(RuntimeError):
    """Base error for failures raised by an embedding provider."""


class EmbeddingConfigurationError(EmbeddingProviderError):
    """Raised when an embedding provider is configured incorrectly."""


class InvalidEmbeddingResponseError(EmbeddingProviderError):
    """Raised when a provider returns malformed embedding data."""