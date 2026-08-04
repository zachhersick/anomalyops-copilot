from openai import OpenAI

from copilot.api.settings import ApiSettings
from copilot.providers.deterministic_answers import (
    DeterministicGroundedAnswerGenerator,
)
from copilot.providers.deterministic_embeddings import (
    DeterministicEmbeddingProvider,
)
from copilot.providers.deterministic_triage import (
    DeterministicTriageAgent,
)
from copilot.providers.errors import (
    EmbeddingConfigurationError,
    GroundedAnswerConfigurationError,
    TriageAgentConfigurationError,
)
from copilot.providers.interfaces import (
    EmbeddingProvider,
    GroundedAnswerGenerator,
    ToolCallingTriageAgent,
)
from copilot.providers.openai_answers import (
    OpenAIGroundedAnswerGenerator,
)
from copilot.providers.openai_embeddings import (
    OpenAIEmbeddingProvider,
)
from copilot.providers.openai_triage import (
    OpenAITriageAgent,
)
from copilot.tools.anomaly import AnomalyOperationalTools


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
            raise EmbeddingConfigurationError(
                "OPENAI_API_KEY is required for the OpenAI embedding provider."
            )

        if (
            settings.embedding_model is None
            or not settings.embedding_model.strip()
        ):
            raise EmbeddingConfigurationError(
                "ANOMALYOPS_EMBEDDING_MODEL is required for the "
                "OpenAI embedding provider."
            )

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

    raise EmbeddingConfigurationError(
        f"Unsupported embedding provider: {settings.ai_provider}"
    )


def create_grounded_answer_generator(
    settings: ApiSettings,
    openai_client: OpenAI | None = None,
) -> GroundedAnswerGenerator:
    if settings.ai_provider == "deterministic":
        return DeterministicGroundedAnswerGenerator()

    if settings.ai_provider == "openai":
        if (
            settings.openai_api_key is None
            or not settings.openai_api_key.strip()
        ):
            raise GroundedAnswerConfigurationError(
                "OPENAI_API_KEY is required for the "
                "OpenAI grounded answer provider."
            )

        if (
            settings.grounded_answer_model is None
            or not settings.grounded_answer_model.strip()
        ):
            raise GroundedAnswerConfigurationError(
                "ANOMALYOPS_GROUNDED_ANSWER_MODEL is required for the "
                "OpenAI grounded answer provider."
            )

        client = (
            openai_client
            if openai_client is not None
            else OpenAI(api_key=settings.openai_api_key)
        )

        return OpenAIGroundedAnswerGenerator(
            model_name=settings.grounded_answer_model,
            client=client,
        )

    raise GroundedAnswerConfigurationError(
        f"Unsupported grounded answer provider: {settings.ai_provider}"
    )


def create_triage_agent(
    settings: ApiSettings,
    tools: AnomalyOperationalTools,
    openai_client: OpenAI | None = None,
) -> ToolCallingTriageAgent:
    if settings.ai_provider == "deterministic":
        return DeterministicTriageAgent(tools)

    if settings.ai_provider == "openai":
        if (
            settings.openai_api_key is None
            or not settings.openai_api_key.strip()
        ):
            raise TriageAgentConfigurationError(
                "OPENAI_API_KEY is required for the OpenAI triage agent."
            )

        if (
            settings.triage_model is None
            or not settings.triage_model.strip()
        ):
            raise TriageAgentConfigurationError(
                "ANOMALYOPS_TRIAGE_MODEL is required for the "
                "OpenAI triage agent."
            )

        client = (
            openai_client
            if openai_client is not None
            else OpenAI(api_key=settings.openai_api_key)
        )

        return OpenAITriageAgent(
            model_name=settings.triage_model,
            client=client,
            tools=tools,
        )

    raise TriageAgentConfigurationError(
        f"Unsupported triage agent provider: {settings.ai_provider}"
    )