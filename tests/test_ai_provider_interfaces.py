from unittest.mock import Mock

from copilot.providers.deterministic_triage import (
    DeterministicTriageAgent,
)
from copilot.providers.openai_triage import (
    OpenAITriageAgent,
)
from copilot.tools.anomaly import (
    AnomalyOperationalTools,
)
from collections.abc import Sequence

from copilot.providers.interfaces import (
    EmbeddingProvider,
    GroundedAnswerGenerator,
    ToolCallingTriageAgent,
)
from copilot.schemas.answer import Citation, GroundedAnswer
from copilot.schemas.retrieval import ScoredChunk
from copilot.schemas.anomaly import RunSummary
from copilot.schemas.triage import TriageReport, TriageRequest
from copilot.providers.deterministic_answers import (
    DeterministicGroundedAnswerGenerator,
)
from copilot.providers.openai_answers import (
    OpenAIGroundedAnswerGenerator,
)


class FakeEmbeddingProvider:
    provider_name = "fake"
    model_name = "fake-embedding"
    dimensions = 3

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [
            [float(len(text)), 0.0, 1.0]
            for text in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return [float(len(text)), 0.0, 1.0]
    
    
class FakeOpenAIClient:
    pass


class MissingGenerateMethod:
    provider_name = "incomplete"
    model_name = "incomplete-model"


class MissingProviderName:
    model_name = "incomplete-model"

    def generate(self, query, context):
        raise NotImplementedError


class MissingModelName:
    provider_name = "incomplete"

    def generate(self, query, context):
        raise NotImplementedError
    
    
class FakeGroundedAnswerGenerator:
    provider_name = "fake"
    model_name = "fake-grounded-answer"

    def __init__(self, answer: GroundedAnswer) -> None:
        self._answer = answer

    def generate(
        self,
        query: str,
        context: Sequence[ScoredChunk],
    ) -> GroundedAnswer:
        return self._answer
    
    
class FakeToolCallingTriageAgent:
    provider_name = "fake"
    model_name = "fake-triage-agent"

    def __init__(self, report: TriageReport) -> None:
        self._report = report

    def triage(
        self,
        request: TriageRequest,
    ) -> TriageReport:
        return self._report
    
    
class MissingTriageMethod:
    provider_name = "incomplete"
    model_name = "incomplete-model"


class MissingTriageProviderName:
    model_name = "incomplete-model"

    def triage(self, request):
        raise NotImplementedError


class MissingTriageModelName:
    provider_name = "incomplete"

    def triage(self, request):
        raise NotImplementedError


def test_fake_embedding_provider_satisfies_protocol():
    provider = FakeEmbeddingProvider()

    assert isinstance(provider, EmbeddingProvider)
    
    
def test_fake_embedding_provider_returns_repeatable_document_embeddings():
    provider = FakeEmbeddingProvider()
    texts = ["temperature alert", "pressure warning"]

    first_result = provider.embed_documents(texts)
    second_result = provider.embed_documents(texts)

    assert first_result == second_result
    assert first_result == [
        [17.0, 0.0, 1.0],
        [16.0, 0.0, 1.0],
    ]


def test_fake_embedding_provider_returns_repeatable_query_embedding():
    provider = FakeEmbeddingProvider()
    query = "temperature alert"

    first_result = provider.embed_query(query)
    second_result = provider.embed_query(query)

    assert first_result == second_result
    assert first_result == [17.0, 0.0, 1.0]
    
    
def test_fake_grounded_answer_generator_satisfies_protocol():
    answer = GroundedAnswer(
        answer="The endpoint is defined in api.py.",
        citations=[
            Citation(
                citation_id=1,
                source_path="api.py",
                start_line=10,
                end_line=20,
            )
        ],
        confidence=0.9,
    )

    generator = FakeGroundedAnswerGenerator(answer)

    assert isinstance(generator, GroundedAnswerGenerator)
    
    
def test_fake_grounded_answer_generator_returns_configured_answer():
    expected_answer = GroundedAnswer(
        answer="The endpoint is defined in api.py.",
        citations=[
            Citation(
                citation_id=1,
                source_path="api.py",
                start_line=10,
                end_line=20,
            )
        ],
        confidence=0.9,
    )

    generator = FakeGroundedAnswerGenerator(expected_answer)

    result = generator.generate(
        query="Where is the endpoint defined?",
        context=[],
    )

    assert result == expected_answer
    
    
def test_fake_tool_calling_triage_agent_satisfies_protocol():
    report = TriageReport(
        run_id=42,
        status="no_alerts",
        run_summary=RunSummary(
            run_id=42,
            total_predictions=100,
            total_anomalies_predicted=0,
            total_row_alerts=0,
            total_alert_events=0,
            critical_alert_events=0,
            warning_alert_events=0,
            info_alert_events=0,
            machines_with_alerts=0,
            max_anomaly_score=None,
            mean_anomaly_score=None,
        ),
        findings=[],
        evidence=[],
    )

    agent = FakeToolCallingTriageAgent(report)

    assert isinstance(agent, ToolCallingTriageAgent)
    
    
def test_fake_tool_calling_triage_agent_returns_configured_report():
    expected_report = TriageReport(
        run_id=42,
        status="no_alerts",
        run_summary=RunSummary(
            run_id=42,
            total_predictions=100,
            total_anomalies_predicted=0,
            total_row_alerts=0,
            total_alert_events=0,
            critical_alert_events=0,
            warning_alert_events=0,
            info_alert_events=0,
            machines_with_alerts=0,
            max_anomaly_score=None,
            mean_anomaly_score=None,
        ),
        findings=[],
        evidence=[],
    )

    agent = FakeToolCallingTriageAgent(expected_report)

    result = agent.triage(
        TriageRequest(
            run_id=42,
            max_events=5,
        )
    )

    assert result == expected_report
    
    
def test_deterministic_answer_generator_satisfies_protocol():
    generator = DeterministicGroundedAnswerGenerator()

    assert isinstance(
        generator,
        GroundedAnswerGenerator,
    )


def test_openai_answer_generator_satisfies_protocol():
    generator = OpenAIGroundedAnswerGenerator(
        model_name="gpt-test",
        client=FakeOpenAIClient(),
    )

    assert isinstance(
        generator,
        GroundedAnswerGenerator,
    )


def test_object_without_generate_does_not_satisfy_answer_protocol():
    provider = MissingGenerateMethod()

    assert not isinstance(
        provider,
        GroundedAnswerGenerator,
    )


def test_object_without_provider_name_does_not_satisfy_answer_protocol():
    provider = MissingProviderName()

    assert not isinstance(
        provider,
        GroundedAnswerGenerator,
    )


def test_object_without_model_name_does_not_satisfy_answer_protocol():
    provider = MissingModelName()

    assert not isinstance(
        provider,
        GroundedAnswerGenerator,
    )
    
    
def test_deterministic_triage_agent_satisfies_protocol():
    tools = Mock(
        spec=AnomalyOperationalTools
    )

    agent = DeterministicTriageAgent(
        tools
    )

    assert isinstance(
        agent,
        ToolCallingTriageAgent,
    )


def test_openai_triage_agent_satisfies_protocol():
    tools = Mock(
        spec=AnomalyOperationalTools
    )

    agent = OpenAITriageAgent(
        model_name="gpt-test",
        client=FakeOpenAIClient(),
        tools=tools,
    )

    assert isinstance(
        agent,
        ToolCallingTriageAgent,
    )


def test_object_without_triage_does_not_satisfy_triage_protocol():
    agent = MissingTriageMethod()

    assert not isinstance(
        agent,
        ToolCallingTriageAgent,
    )


def test_triage_agent_without_provider_name_does_not_satisfy_protocol():
    agent = MissingTriageProviderName()

    assert not isinstance(
        agent,
        ToolCallingTriageAgent,
    )


def test_triage_agent_without_model_name_does_not_satisfy_protocol():
    agent = MissingTriageModelName()

    assert not isinstance(
        agent,
        ToolCallingTriageAgent,
    )