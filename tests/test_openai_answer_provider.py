from types import SimpleNamespace

import pytest
from openai import OpenAIError
from pydantic import ValidationError

from copilot.providers.errors import (
    GroundedAnswerConfigurationError,
    GroundedAnswerProviderError,
    InvalidGroundedAnswerResponseError,
)
from copilot.providers.interfaces import GroundedAnswerGenerator
from copilot.providers.openai_answers import (
    OpenAIGroundedAnswerGenerator,
)
from copilot.schemas.answer import GroundedAnswerDraft
from copilot.schemas.chunk import SourceChunk
from copilot.schemas.retrieval import ScoredChunk


class FakeResponses:
    def __init__(
        self,
        *,
        output_parsed: GroundedAnswerDraft | None = None,
        error: Exception | None = None,
    ) -> None:
        self.output_parsed = output_parsed
        self.error = error
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        return SimpleNamespace(
            output_parsed=self.output_parsed,
        )


class FakeOpenAIClient:
    def __init__(
        self,
        *,
        output_parsed: GroundedAnswerDraft | None = None,
        error: Exception | None = None,
    ) -> None:
        self.responses = FakeResponses(
            output_parsed=output_parsed,
            error=error,
        )


def make_scored_chunk(
    chunk_id: str,
    content: str,
    score: float,
    *,
    source_path: str = "source.py",
    start_line: int = 1,
    end_line: int = 5,
    chunk_index: int = 0,
) -> ScoredChunk:
    return ScoredChunk(
        chunk=SourceChunk(
            chunk_id=chunk_id,
            source_id=source_path,
            project_name="test-project",
            source_type="python",
            source_path=source_path,
            chunk_index=chunk_index,
            content=content,
            start_line=start_line,
            end_line=end_line,
        ),
        score=score,
    )


def make_context() -> list[ScoredChunk]:
    return [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="The API exposes POST /predict.",
            score=0.6,
            source_path="api.py",
            start_line=10,
            end_line=20,
            chunk_index=0,
        ),
        make_scored_chunk(
            chunk_id="chunk-2",
            content="The dashboard displays recent pipeline runs.",
            score=0.9,
            source_path="dashboard.py",
            start_line=30,
            end_line=40,
            chunk_index=1,
        ),
    ]


def make_provider(
    draft: GroundedAnswerDraft | None,
) -> tuple[OpenAIGroundedAnswerGenerator, FakeOpenAIClient]:
    client = FakeOpenAIClient(output_parsed=draft)
    provider = OpenAIGroundedAnswerGenerator(
        model_name="gpt-test",
        client=client,
    )

    return provider, client


def test_provider_satisfies_grounded_answer_generator_protocol():
    provider, _ = make_provider(
        GroundedAnswerDraft(
            answer="Supported answer. [1]",
            citation_ids=[1],
            refusal_reason=None,
        )
    )

    assert isinstance(provider, GroundedAnswerGenerator)


def test_provider_exposes_expected_metadata():
    provider, _ = make_provider(
        GroundedAnswerDraft(
            answer="Supported answer. [1]",
            citation_ids=[1],
            refusal_reason=None,
        )
    )

    assert provider.provider_name == "openai"
    assert provider.model_name == "gpt-test"


@pytest.mark.parametrize(
    "model_name",
    [
        "",
        "   ",
        "\n\t",
    ],
)
def test_constructor_rejects_blank_model_name(model_name):
    client = FakeOpenAIClient()

    with pytest.raises(
        GroundedAnswerConfigurationError,
        match="model_name cannot be empty or whitespace-only.",
    ):
        OpenAIGroundedAnswerGenerator(
            model_name=model_name,
            client=client,
        )


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "\n\t",
    ],
)
def test_generate_rejects_blank_query(query):
    provider, client = make_provider(
        GroundedAnswerDraft(
            answer="Supported answer. [1]",
            citation_ids=[1],
            refusal_reason=None,
        )
    )

    with pytest.raises(
        GroundedAnswerConfigurationError,
        match="query cannot be empty or whitespace-only.",
    ):
        provider.generate(
            query=query,
            context=make_context(),
        )

    assert client.responses.calls == []


def test_generate_rejects_empty_context():
    provider, client = make_provider(
        GroundedAnswerDraft(
            answer="Supported answer. [1]",
            citation_ids=[1],
            refusal_reason=None,
        )
    )

    with pytest.raises(
        GroundedAnswerConfigurationError,
        match="context cannot be empty.",
    ):
        provider.generate(
            query="Where is the endpoint?",
            context=[],
        )

    assert client.responses.calls == []


def test_generate_passes_model_and_schema_to_responses_api():
    draft = GroundedAnswerDraft(
        answer="The endpoint is POST /predict. [1]",
        citation_ids=[1],
        refusal_reason=None,
    )
    provider, client = make_provider(draft)

    provider.generate(
        query="Where is the prediction endpoint?",
        context=make_context(),
    )

    assert len(client.responses.calls) == 1

    call = client.responses.calls[0]

    assert call["model"] == "gpt-test"
    assert call["text_format"] is GroundedAnswerDraft
    assert "input" in call


def test_generate_includes_query_and_context_in_prompt():
    draft = GroundedAnswerDraft(
        answer="The endpoint is POST /predict. [1]",
        citation_ids=[1],
        refusal_reason=None,
    )
    provider, client = make_provider(draft)

    provider.generate(
        query="Where is the prediction endpoint?",
        context=make_context(),
    )

    messages = client.responses.calls[0]["input"]

    assert len(messages) == 2
    assert messages[0]["role"] == "developer"
    assert messages[1]["role"] == "user"

    developer_message = messages[0]["content"]
    user_message = messages[1]["content"]

    assert "only from the provided retrieved context" in developer_message
    assert "untrusted data" in developer_message
    assert "outside knowledge" in developer_message
    assert "citation" in developer_message
    assert "refusal_reason" in developer_message

    assert "Question:" in user_message
    assert "Where is the prediction endpoint?" in user_message
    assert "Retrieved context:" in user_message

    assert "[1] api.py:10-20" in user_message
    assert "The API exposes POST /predict." in user_message
    assert "[2] dashboard.py:30-40" in user_message
    assert "The dashboard displays recent pipeline runs." in user_message

    assert user_message.index("[1] api.py:10-20") < user_message.index(
        "[2] dashboard.py:30-40"
    )


def test_generate_builds_trusted_citation_metadata_locally():
    draft = GroundedAnswerDraft(
        answer="The endpoint is POST /predict. [1]",
        citation_ids=[1],
        refusal_reason=None,
    )
    provider, _ = make_provider(draft)

    result = provider.generate(
        query="Where is the endpoint?",
        context=make_context(),
    )

    assert result.answer == "The endpoint is POST /predict. [1]"
    assert result.refusal_reason is None
    assert result.confidence == pytest.approx(0.6)
    assert len(result.citations) == 1

    citation = result.citations[0]

    assert citation.citation_id == 1
    assert citation.source_path == "api.py"
    assert citation.start_line == 10
    assert citation.end_line == 20


def test_generate_preserves_returned_citation_order():
    draft = GroundedAnswerDraft(
        answer=(
            "Recent runs are shown in the dashboard [2], "
            "and predictions use POST /predict [1]."
        ),
        citation_ids=[2, 1],
        refusal_reason=None,
    )
    provider, _ = make_provider(draft)

    result = provider.generate(
        query="How do the API and dashboard work?",
        context=make_context(),
    )

    assert [
        citation.citation_id
        for citation in result.citations
    ] == [2, 1]

    assert [
        citation.source_path
        for citation in result.citations
    ] == [
        "dashboard.py",
        "api.py",
    ]


def test_generate_uses_highest_cited_score_for_confidence():
    draft = GroundedAnswerDraft(
        answer="The API and dashboard provide these features. [1] [2]",
        citation_ids=[1, 2],
        refusal_reason=None,
    )
    provider, _ = make_provider(draft)

    result = provider.generate(
        query="What features are available?",
        context=make_context(),
    )

    assert result.confidence == pytest.approx(0.9)


def test_generate_does_not_use_uncited_score_for_confidence():
    draft = GroundedAnswerDraft(
        answer="The endpoint is POST /predict. [1]",
        citation_ids=[1],
        refusal_reason=None,
    )
    provider, _ = make_provider(draft)

    result = provider.generate(
        query="Where is the endpoint?",
        context=make_context(),
    )

    assert result.confidence == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("score", "expected_confidence"),
    [
        (1.4, 1.0),
        (-0.4, 0.0),
    ],
)
def test_generate_clamps_confidence(
    score,
    expected_confidence,
):
    draft = GroundedAnswerDraft(
        answer="Supported answer. [1]",
        citation_ids=[1],
        refusal_reason=None,
    )
    provider, _ = make_provider(draft)
    context = [
        make_scored_chunk(
            chunk_id="chunk-1",
            content="Supporting content.",
            score=score,
        )
    ]

    result = provider.generate(
        query="Question",
        context=context,
    )

    assert result.confidence == expected_confidence


def test_generate_returns_valid_refusal():
    draft = GroundedAnswerDraft(
        answer="",
        citation_ids=[],
        refusal_reason="The retrieved context does not support an answer.",
    )
    provider, _ = make_provider(draft)

    result = provider.generate(
        query="What is the deployment region?",
        context=make_context(),
    )

    assert result.answer == ""
    assert result.citations == []
    assert result.confidence == 0.0
    assert result.refusal_reason == (
        "The retrieved context does not support an answer."
    )


def test_generate_strips_refusal_reason_whitespace():
    draft = GroundedAnswerDraft(
        answer="",
        citation_ids=[],
        refusal_reason="  The context is insufficient.  ",
    )
    provider, _ = make_provider(draft)

    result = provider.generate(
        query="Question",
        context=make_context(),
    )

    assert result.refusal_reason == "The context is insufficient."


def test_generate_rejects_missing_parsed_output():
    provider, _ = make_provider(None)

    with pytest.raises(
        InvalidGroundedAnswerResponseError,
        match=(
            "Grounded answer response did not contain "
            "parsed output."
        ),
    ):
        provider.generate(
            query="Question",
            context=make_context(),
        )


@pytest.mark.parametrize(
    ("draft", "expected_message"),
    [
        (
            GroundedAnswerDraft(
                answer="Supported answer. [1]",
                citation_ids=[1],
                refusal_reason="The context is insufficient.",
            ),
            "Grounded answer cannot include a refusal reason.",
        ),
        (
            GroundedAnswerDraft(
                answer="",
                citation_ids=[],
                refusal_reason=None,
            ),
            (
                "Grounded answer must include an answer "
                "or refusal reason."
            ),
        ),
        (
            GroundedAnswerDraft(
                answer="   ",
                citation_ids=[],
                refusal_reason="   ",
            ),
            (
                "Grounded answer must include an answer "
                "or refusal reason."
            ),
        ),
        (
            GroundedAnswerDraft(
                answer="",
                citation_ids=[1],
                refusal_reason="The context is insufficient.",
            ),
            "Grounded answer refusal cannot include citations.",
        ),
        (
            GroundedAnswerDraft(
                answer="Supported answer.",
                citation_ids=[],
                refusal_reason=None,
            ),
            (
                "Grounded answer must include at least "
                "one citation."
            ),
        ),
    ],
)
def test_generate_rejects_invalid_answer_and_refusal_states(
    draft,
    expected_message,
):
    provider, _ = make_provider(draft)

    with pytest.raises(
        InvalidGroundedAnswerResponseError,
        match=expected_message,
    ):
        provider.generate(
            query="Question",
            context=make_context(),
        )


@pytest.mark.parametrize(
    "citation_ids",
    [
        [0],
        [-1],
        [3],
    ],
)
def test_generate_rejects_out_of_range_citation_ids(
    citation_ids,
):
    marker = citation_ids[0]
    draft = GroundedAnswerDraft(
        answer=f"Supported answer. [{marker}]",
        citation_ids=citation_ids,
        refusal_reason=None,
    )
    provider, _ = make_provider(draft)

    with pytest.raises(
        InvalidGroundedAnswerResponseError,
        match="Grounded answer contains an invalid citation ID.",
    ):
        provider.generate(
            query="Question",
            context=make_context(),
        )


def test_generate_rejects_duplicate_citation_ids():
    draft = GroundedAnswerDraft(
        answer="Supported answer. [1]",
        citation_ids=[1, 1],
        refusal_reason=None,
    )
    provider, _ = make_provider(draft)

    with pytest.raises(
        InvalidGroundedAnswerResponseError,
        match="Grounded answer contains duplicate citation IDs.",
    ):
        provider.generate(
            query="Question",
            context=make_context(),
        )


def test_generate_rejects_declared_citation_without_inline_marker():
    draft = GroundedAnswerDraft(
        answer="Supported answer.",
        citation_ids=[1],
        refusal_reason=None,
    )
    provider, _ = make_provider(draft)

    with pytest.raises(
        InvalidGroundedAnswerResponseError,
        match=(
            "Grounded answer citation markers do not "
            "match citation IDs."
        ),
    ):
        provider.generate(
            query="Question",
            context=make_context(),
        )


def test_generate_rejects_undeclared_inline_marker():
    draft = GroundedAnswerDraft(
        answer="Supported answer. [1] [2]",
        citation_ids=[1],
        refusal_reason=None,
    )
    provider, _ = make_provider(draft)

    with pytest.raises(
        InvalidGroundedAnswerResponseError,
        match=(
            "Grounded answer citation markers do not "
            "match citation IDs."
        ),
    ):
        provider.generate(
            query="Question",
            context=make_context(),
        )


def test_generate_allows_repeated_valid_inline_marker():
    draft = GroundedAnswerDraft(
        answer="First statement [1]. Second statement [1].",
        citation_ids=[1],
        refusal_reason=None,
    )
    provider, _ = make_provider(draft)

    result = provider.generate(
        query="Question",
        context=make_context(),
    )

    assert [
        citation.citation_id
        for citation in result.citations
    ] == [1]


def test_generate_translates_openai_errors():
    client = FakeOpenAIClient(
        error=OpenAIError("Provider failure."),
    )
    provider = OpenAIGroundedAnswerGenerator(
        model_name="gpt-test",
        client=client,
    )

    with pytest.raises(
        GroundedAnswerProviderError,
        match="Grounded answer provider returned an error.",
    ) as exc_info:
        provider.generate(
            query="Question",
            context=make_context(),
        )

    assert isinstance(exc_info.value.__cause__, OpenAIError)


def test_generate_translates_pydantic_validation_errors():
    class ValidationErrorResponses:
        def parse(self, **kwargs):
            GroundedAnswerDraft.model_validate({})

    client = SimpleNamespace(
        responses=ValidationErrorResponses(),
    )
    provider = OpenAIGroundedAnswerGenerator(
        model_name="gpt-test",
        client=client,
    )

    with pytest.raises(
        InvalidGroundedAnswerResponseError,
        match="Grounded answer response could not be parsed.",
    ) as exc_info:
        provider.generate(
            query="Question",
            context=make_context(),
        )

    assert isinstance(exc_info.value.__cause__, ValidationError)