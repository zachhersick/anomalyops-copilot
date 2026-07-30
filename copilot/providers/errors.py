class EmbeddingProviderError(RuntimeError):
    """Base error for failures raised by an embedding provider."""


class EmbeddingConfigurationError(EmbeddingProviderError):
    """Raised when an embedding provider is configured incorrectly."""


class InvalidEmbeddingResponseError(EmbeddingProviderError):
    """Raised when a provider returns malformed embedding data."""
    
    
class GroundedAnswerProviderError(RuntimeError):
    """Base error for failures raised by a grouded answer provider."""
    
class GroundedAnswerConfigurationError(GroundedAnswerProviderError):
    """Raised when a grounded answer provider is configured incorrectly."""
    
    
class InvalidGroundedAnswerResponseError(GroundedAnswerProviderError):
    """Raised when a grounded answer provider returns malformed embedding data."""