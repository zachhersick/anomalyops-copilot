from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from copilot.schemas.answer import GroundedAnswer
from copilot.schemas.retrieval import ScoredChunk
from copilot.schemas.triage import TriageReport, TriageRequest


@runtime_checkable
class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str
    dimensions: int
    
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        ...
    
    
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        ...
    
    
@runtime_checkable
class GroundedAnswerGenerator(Protocol):
    provider_name: str
    model_name: str
    
    def generate(
        self,
        query: str,
        context: Sequence[ScoredChunk],
    ) -> GroundedAnswer:
        ...
    
    
@runtime_checkable
class ToolCallingTriageAgent(Protocol):
    provider_name: str
    model_name: str
    
    def triage(
        self,
        request: TriageRequest,
    ) -> TriageReport:
        ...