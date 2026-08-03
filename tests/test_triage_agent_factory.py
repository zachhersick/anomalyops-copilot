from unittest.mock import Mock

import pytest

import copilot.providers.factory as factory_module
from copilot.api.settings import ApiSettings
from copilot.providers.deterministic_triage import (
    DeterministicTriageAgent,
)
from copilot.providers.errors import (
    TriageAgentConfigurationError,
)
from copilot.providers.openai_triage import (
    OpenAITriageAgent,
)
from copilot.tools.anomaly import AnomalyOperationalTools


def make_tools() -> Mock:
    return Mock(spec=AnomalyOperationalTools)


def test_factory_creates_deterministic_triage_agent():
    tools = make_tools()

    agent = factory_module.create_triage_agent(
        ApiSettings(
            ai_provider="deterministic",
        ),
        tools,
    )

    assert isinstance(
        agent,
        DeterministicTriageAgent,
    )


def test_deterministic_triage_requires_no_openai_configuration():
    tools = make_tools()

    agent = factory_module.create_triage_agent(
        ApiSettings(
            ai_provider="deterministic",
            openai_api_key=None,
            triage_model=None,
        ),
        tools,
    )

    assert isinstance(
        agent,
        DeterministicTriageAgent,
    )


def test_deterministic_factory_does_not_construct_openai(
    monkeypatch,
):
    tools = make_tools()
    openai_constructor = Mock()

    monkeypatch.setattr(
        factory_module,
        "OpenAI",
        openai_constructor,
    )

    factory_module.create_triage_agent(
        ApiSettings(
            ai_provider="deterministic",
        ),
        tools,
    )

    openai_constructor.assert_not_called()


def test_factory_creates_openai_triage_agent_with_injected_client():
    tools = make_tools()
    client = Mock()

    agent = factory_module.create_triage_agent(
        ApiSettings(
            ai_provider="openai",
            openai_api_key="test-key",
            triage_model="gpt-test",
        ),
        tools,
        openai_client=client,
    )

    assert isinstance(
        agent,
        OpenAITriageAgent,
    )
    assert agent.model_name == "gpt-test"
    assert agent._client is client
    assert agent._tools is tools


def test_openai_factory_reuses_injected_client(
    monkeypatch,
):
    tools = make_tools()
    client = Mock()
    openai_constructor = Mock()

    monkeypatch.setattr(
        factory_module,
        "OpenAI",
        openai_constructor,
    )

    factory_module.create_triage_agent(
        ApiSettings(
            ai_provider="openai",
            openai_api_key="test-key",
            triage_model="gpt-test",
        ),
        tools,
        openai_client=client,
    )

    openai_constructor.assert_not_called()


def test_openai_factory_constructs_client_when_not_injected(
    monkeypatch,
):
    tools = make_tools()
    client = Mock()
    openai_constructor = Mock(
        return_value=client
    )

    monkeypatch.setattr(
        factory_module,
        "OpenAI",
        openai_constructor,
    )

    agent = factory_module.create_triage_agent(
        ApiSettings(
            ai_provider="openai",
            openai_api_key="test-key",
            triage_model="gpt-test",
        ),
        tools,
    )

    openai_constructor.assert_called_once_with(
        api_key="test-key"
    )
    assert agent._client is client


@pytest.mark.parametrize(
    "api_key",
    [
        None,
        "",
        "   ",
    ],
)
def test_openai_factory_requires_api_key(
    api_key,
):
    tools = make_tools()

    with pytest.raises(
        TriageAgentConfigurationError,
        match=(
            "OPENAI_API_KEY is required for "
            "the OpenAI triage agent."
        ),
    ):
        factory_module.create_triage_agent(
            ApiSettings(
                ai_provider="openai",
                openai_api_key=api_key,
                triage_model="gpt-test",
            ),
            tools,
        )


@pytest.mark.parametrize(
    "triage_model",
    [
        None,
        "",
        "   ",
    ],
)
def test_openai_factory_requires_triage_model(
    triage_model,
):
    tools = make_tools()

    with pytest.raises(
        TriageAgentConfigurationError,
        match=(
            "ANOMALYOPS_TRIAGE_MODEL is required "
            "for the OpenAI triage agent."
        ),
    ):
        factory_module.create_triage_agent(
            ApiSettings(
                ai_provider="openai",
                openai_api_key="test-key",
                triage_model=triage_model,
            ),
            tools,
        )


def test_triage_factory_does_not_require_embedding_model():
    tools = make_tools()

    agent = factory_module.create_triage_agent(
        ApiSettings(
            ai_provider="openai",
            openai_api_key="test-key",
            triage_model="gpt-test",
            embedding_model=None,
        ),
        tools,
        openai_client=Mock(),
    )

    assert isinstance(
        agent,
        OpenAITriageAgent,
    )


def test_triage_factory_does_not_require_grounded_answer_model():
    tools = make_tools()

    agent = factory_module.create_triage_agent(
        ApiSettings(
            ai_provider="openai",
            openai_api_key="test-key",
            triage_model="gpt-test",
            grounded_answer_model=None,
        ),
        tools,
        openai_client=Mock(),
    )

    assert isinstance(
        agent,
        OpenAITriageAgent,
    )