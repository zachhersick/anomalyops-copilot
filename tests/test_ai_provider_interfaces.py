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