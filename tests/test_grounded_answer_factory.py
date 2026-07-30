import pytest

import copilot.providers.factory as factory_module
from copilot.api.settings import ApiSettings
from copilot.providers.deterministic_answers import (
    DeterministicGroundedAnswerGenerator,
)
from copilot.providers.errors import (
    GroundedAnswerConfigurationError,
)
from copilot.providers.factory import (
    create_grounded_answer_generator,
)
from copilot.providers.openai_answers import (
    OpenAIGroundedAnswerGenerator,
)


class FakeOpenAIClient:
    pass


def make_settings(
    **overrides,
) -> ApiSettings:
    values = {
        "ai_provider": "deterministic",
        "openai_api_key": None,
        "embedding_model": None,
        "grounded_answer_model": None,
    }
    values.update(overrides)

    return ApiSettings(**values)


def test_factory_creates_deterministic_generator():
    settings = make_settings(
        ai_provider="deterministic",
    )

    generator = create_grounded_answer_generator(settings)

    assert isinstance(
        generator,
        DeterministicGroundedAnswerGenerator,
    )
    assert generator.provider_name == "deterministic"
    assert generator.model_name == (
        "retrieval-context-template-v1"
    )


def test_deterministic_generator_does_not_require_openai_settings():
    settings = make_settings(
        ai_provider="deterministic",
        openai_api_key=None,
        grounded_answer_model=None,
    )

    generator = create_grounded_answer_generator(settings)

    assert isinstance(
        generator,
        DeterministicGroundedAnswerGenerator,
    )


def test_deterministic_generator_does_not_construct_openai_client(
    monkeypatch,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "OpenAI client should not be constructed."
        )

    monkeypatch.setattr(
        factory_module,
        "OpenAI",
        fail_if_called,
    )

    settings = make_settings(
        ai_provider="deterministic",
    )

    generator = create_grounded_answer_generator(settings)

    assert isinstance(
        generator,
        DeterministicGroundedAnswerGenerator,
    )


def test_factory_creates_openai_generator_with_injected_client():
    client = FakeOpenAIClient()
    settings = make_settings(
        ai_provider="openai",
        openai_api_key="test-api-key",
        grounded_answer_model="gpt-test",
    )

    generator = create_grounded_answer_generator(
        settings,
        openai_client=client,
    )

    assert isinstance(
        generator,
        OpenAIGroundedAnswerGenerator,
    )
    assert generator.provider_name == "openai"
    assert generator.model_name == "gpt-test"
    assert generator._client is client


def test_factory_constructs_openai_client_when_not_injected(
    monkeypatch,
):
    created_clients = []
    fake_client = FakeOpenAIClient()

    def create_fake_client(*, api_key):
        created_clients.append(api_key)
        return fake_client

    monkeypatch.setattr(
        factory_module,
        "OpenAI",
        create_fake_client,
    )

    settings = make_settings(
        ai_provider="openai",
        openai_api_key="test-api-key",
        grounded_answer_model="gpt-test",
    )

    generator = create_grounded_answer_generator(settings)

    assert isinstance(
        generator,
        OpenAIGroundedAnswerGenerator,
    )
    assert generator._client is fake_client
    assert created_clients == ["test-api-key"]


@pytest.mark.parametrize(
    "api_key",
    [
        None,
        "",
        "   ",
        "\n\t",
    ],
)
def test_openai_generator_requires_api_key(api_key):
    settings = make_settings(
        ai_provider="openai",
        openai_api_key=api_key,
        grounded_answer_model="gpt-test",
    )

    with pytest.raises(
        GroundedAnswerConfigurationError,
        match=(
            "OPENAI_API_KEY is required for the OpenAI "
            "grounded answer provider."
        ),
    ):
        create_grounded_answer_generator(
            settings,
            openai_client=FakeOpenAIClient(),
        )


@pytest.mark.parametrize(
    "model_name",
    [
        None,
        "",
        "   ",
        "\n\t",
    ],
)
def test_openai_generator_requires_grounded_answer_model(
    model_name,
):
    settings = make_settings(
        ai_provider="openai",
        openai_api_key="test-api-key",
        grounded_answer_model=model_name,
    )

    with pytest.raises(
        GroundedAnswerConfigurationError,
        match=(
            "ANOMALYOPS_GROUNDED_ANSWER_MODEL is required "
            "for the OpenAI grounded answer provider."
        ),
    ):
        create_grounded_answer_generator(
            settings,
            openai_client=FakeOpenAIClient(),
        )


def test_openai_generator_does_not_require_embedding_model():
    client = FakeOpenAIClient()
    settings = make_settings(
        ai_provider="openai",
        openai_api_key="test-api-key",
        grounded_answer_model="gpt-test",
        embedding_model=None,
    )

    generator = create_grounded_answer_generator(
        settings,
        openai_client=client,
    )

    assert isinstance(
        generator,
        OpenAIGroundedAnswerGenerator,
    )
    assert generator.model_name == "gpt-test"
    assert generator._client is client