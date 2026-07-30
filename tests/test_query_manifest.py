import json
from collections.abc import Sequence
from unittest.mock import patch

import pytest

import scripts.query_manifest as query_manifest_module
from copilot.ingestion.manifest import write_chunk_manifest
from copilot.providers.errors import GroundedAnswerProviderError
from copilot.schemas.answer import Citation, GroundedAnswer
from copilot.schemas.chunk import SourceChunk
from copilot.schemas.retrieval import ScoredChunk
from scripts.query_manifest import main


class RecordingGroundedAnswerGenerator:
    provider_name = "recording"
    model_name = "recording-test"

    def __init__(
        self,
        *,
        result: GroundedAnswer | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    def generate(
        self,
        query: str,
        context: Sequence[ScoredChunk],
    ) -> GroundedAnswer:
        self.calls.append(
            {
                "query": query,
                "context": context,
            }
        )

        if self.error is not None:
            raise self.error

        assert self.result is not None
        return self.result


def create_test_manifest(tmp_path):
    manifest_path = tmp_path / "chunks.json"

    write_chunk_manifest(
        [
            SourceChunk(
                chunk_id="chunk-1",
                source_id="api.py",
                project_name="test-project",
                source_type="python",
                source_path="api.py",
                chunk_index=0,
                content="The API exposes POST /predict.",
                start_line=10,
                end_line=20,
            )
        ],
        manifest_path,
    )

    return manifest_path


def patch_successful_retrieval(monkeypatch) -> None:
    chunk = SourceChunk(
        chunk_id="chunk-1",
        source_id="api.py",
        project_name="test-project",
        source_type="python",
        source_path="api.py",
        chunk_index=0,
        content="The API exposes POST /predict.",
        start_line=10,
        end_line=20,
    )

    monkeypatch.setattr(
        query_manifest_module,
        "retrieve_relevant_chunks",
        lambda *args, **kwargs: [
            ScoredChunk(
                chunk=chunk,
                score=0.9,
            )
        ],
    )


def test_query_manifest_loads_manifest_and_prints_query_results(
    tmp_path,
    capsys,
):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk("chunk-1", "prediction api endpoint"),
        make_chunk("chunk-2", "dashboard summary view"),
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--top-k",
            "2",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Score:" in captured.out
    assert "Source:" in captured.out
    assert "source.py" in captured.out


def test_query_manifest_respects_top_k(tmp_path, capsys):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk("chunk-1", "prediction api endpoint"),
        make_chunk("chunk-2", "dashboard summary view"),
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--top-k",
            "1",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.count("Score:") == 1


def test_query_manifest_rejects_non_positive_top_k(tmp_path):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk("chunk-1", "prediction api endpoint"),
        make_chunk("chunk-2", "dashboard summary view"),
    ]
    write_chunk_manifest(chunks, manifest_path)

    with pytest.raises(ValueError):
        main(
            [
                str(manifest_path),
                "prediction api",
                "--top-k",
                "0",
            ]
        )


def test_query_manifest_answer_mode_prints_grounded_answer_fields(
    tmp_path,
    capsys,
):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk("chunk-1", "prediction api endpoint"),
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--answer",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Answer:" in captured.out
    assert "Confidence:" in captured.out
    assert "Citations:" in captured.out


def test_query_manifest_answer_mode_includes_citation_source(
    monkeypatch,
    tmp_path,
    capsys,
):
    patch_successful_retrieval(monkeypatch)
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            content="prediction api endpoint",
            source_path="api.py",
            start_line=10,
            end_line=20,
        ),
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--answer",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "[1] api.py:10-20" in captured.out


def test_query_manifest_answer_mode_respects_top_k(
    tmp_path,
    capsys,
):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk(
            "chunk-1",
            "prediction api endpoint",
            source_path="api.py",
        ),
        make_chunk(
            "chunk-2",
            "dashboard summary view",
            source_path="dashboard.py",
        ),
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--top-k",
            "1",
            "--answer",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "[1]" in captured.out
    assert "[2]" not in captured.out


def test_query_manifest_answer_json_mode_prints_valid_json(
    tmp_path,
    capsys,
):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk("chunk-1", "prediction api endpoint"),
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--answer",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert isinstance(data, dict)


def test_query_manifest_answer_json_mode_includes_grounded_answer_keys(
    tmp_path,
    capsys,
):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk("chunk-1", "prediction api endpoint"),
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--answer",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert "answer" in data
    assert "confidence" in data
    assert "citations" in data
    assert "refusal_reason" in data


def test_query_manifest_answer_json_mode_includes_citation_fields(
    monkeypatch,
    tmp_path,
    capsys,
):
    patch_successful_retrieval(monkeypatch)
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            content="prediction api endpoint",
            source_path="api.py",
            start_line=10,
            end_line=20,
        ),
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--answer",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    citation = data["citations"][0]

    assert exit_code == 0
    assert citation["citation_id"] == 1
    assert citation["source_path"] == "api.py"
    assert citation["start_line"] == 10
    assert citation["end_line"] == 20


def test_query_manifest_answer_json_mode_preserves_readable_answer_mode(
    tmp_path,
    capsys,
):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk("chunk-1", "prediction api endpoint"),
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--answer",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Answer:" in captured.out
    assert "Confidence:" in captured.out
    assert "Citations:" in captured.out

    with pytest.raises(json.JSONDecodeError):
        json.loads(captured.out)


def test_query_manifest_answer_json_mode_preserves_raw_retrieval_mode(
    tmp_path,
    capsys,
):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk("chunk-1", "prediction api endpoint"),
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Score:" in captured.out
    assert "Source:" in captured.out
    assert "Preview:" in captured.out


def test_query_manifest_answer_json_mode_respects_top_k(
    tmp_path,
    capsys,
):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk(
            "chunk-1",
            "prediction api endpoint",
            source_path="api.py",
        ),
        make_chunk(
            "chunk-2",
            "dashboard summary view",
            source_path="dashboard.py",
        ),
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--top-k",
            "1",
            "--answer",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert len(data["citations"]) == 1


def test_query_manifest_answer_mode_refuses_when_min_score_is_too_high(
    monkeypatch,
    tmp_path,
    capsys,
):
    patch_successful_retrieval(monkeypatch)

    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk("chunk-1", "prediction api endpoint"),
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--answer",
            "--min-score",
            "1.1",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Answer:" in captured.out
    assert "Confidence: 0.9000" in captured.out
    assert (
        "Refusal reason: Retrieved context was below the "
        "confidence threshold."
        in captured.out
    )
    assert "Citations:" in captured.out


def test_query_manifest_answer_json_mode_refuses_when_min_score_is_too_high(
    monkeypatch,
    tmp_path,
    capsys,
):
    patch_successful_retrieval(monkeypatch)

    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk("chunk-1", "prediction api endpoint"),
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--answer",
            "--json",
            "--min-score",
            "1.1",
        ]
    )

    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["answer"] == ""
    assert data["citations"] == []
    assert data["confidence"] == pytest.approx(0.9)
    assert (
        data["refusal_reason"]
        == "Retrieved context was below the confidence threshold."
    )


def test_query_manifest_answer_show_context_contains_all_sections(
    tmp_path,
    capsys,
):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk("chunk-1", "prediction api endpoint"),
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--answer",
            "--show-context",
        ]
    )

    captured = capsys.readouterr()
    data = captured.out

    assert exit_code == 0
    assert "Answer:" in data
    assert "Confidence:" in data
    assert "Citations:" in data
    assert "Context:" in data


def test_query_manifest_answer_show_context_includes_source_path_line_range(
    tmp_path,
    capsys,
):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk(
            "chunk-1",
            "prediction api endpoint",
            source_path="source.py",
            start_line=10,
            end_line=20,
        )
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--answer",
            "--show-context",
        ]
    )

    captured = capsys.readouterr()
    data = captured.out

    assert exit_code == 0
    assert "source.py:10-20" in data


def test_query_manifest_answer_show_context_includes_chunk_content(
    tmp_path,
    capsys,
):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk(
            "chunk-1",
            "prediction api endpoint",
            source_path="source.py",
            start_line=10,
            end_line=20,
        )
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--answer",
            "--show-context",
        ]
    )

    captured = capsys.readouterr()
    data = captured.out

    assert exit_code == 0
    assert chunks[0].content in data


def test_query_manifest_answer_json_show_context_still_outputs_only_json(
    tmp_path,
    capsys,
):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk("chunk-1", "prediction api endpoint"),
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--answer",
            "--json",
            "--show-context",
        ]
    )

    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert isinstance(data, dict)
    assert "Context:" not in captured.out


def test_raw_mode_does_not_create_grounded_answer_generator(
    tmp_path,
    capsys,
):
    manifest_path = create_test_manifest(tmp_path)

    create_generator = patch(
        "scripts.query_manifest.create_grounded_answer_generator"
    )
    load_settings = patch(
        "scripts.query_manifest.load_api_settings"
    )

    with create_generator as generator_factory:
        with load_settings as settings_loader:
            exit_code = main(
                [
                    str(manifest_path),
                    "prediction api",
                ]
            )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Score:" in output
    assert "Source:" in output
    assert "Preview:" in output

    generator_factory.assert_not_called()
    settings_loader.assert_not_called()


def test_answer_mode_loads_settings_and_creates_generator(
    monkeypatch,
    tmp_path,
):
    manifest_path = create_test_manifest(tmp_path)
    settings = object()
    generator = RecordingGroundedAnswerGenerator(
        result=GroundedAnswer(
            answer="The endpoint is POST /predict. [1]",
            citations=[
                Citation(
                    citation_id=1,
                    source_path="api.py",
                    start_line=10,
                    end_line=20,
                )
            ],
            confidence=0.9,
            refusal_reason=None,
        )
    )

    monkeypatch.setattr(
        query_manifest_module,
        "load_api_settings",
        lambda: settings,
    )

    captured = {}

    def fake_create_generator(received_settings):
        captured["settings"] = received_settings
        return generator

    monkeypatch.setattr(
        query_manifest_module,
        "create_grounded_answer_generator",
        fake_create_generator,
    )

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--answer",
        ]
    )

    assert exit_code == 0
    assert captured["settings"] is settings


def test_answer_mode_passes_generator_to_builder(
    monkeypatch,
    tmp_path,
):
    manifest_path = create_test_manifest(tmp_path)
    generator = RecordingGroundedAnswerGenerator(
        result=GroundedAnswer(
            answer="Generated answer. [1]",
            citations=[
                Citation(
                    citation_id=1,
                    source_path="api.py",
                    start_line=10,
                    end_line=20,
                )
            ],
            confidence=0.8,
            refusal_reason=None,
        )
    )
    captured = {}

    monkeypatch.setattr(
        query_manifest_module,
        "load_api_settings",
        lambda: object(),
    )
    monkeypatch.setattr(
        query_manifest_module,
        "create_grounded_answer_generator",
        lambda settings: generator,
    )

    def fake_build_grounded_answer(
        query,
        scored_chunks,
        min_score,
        *,
        generator,
    ):
        captured["query"] = query
        captured["scored_chunks"] = scored_chunks
        captured["min_score"] = min_score
        captured["generator"] = generator

        return GroundedAnswer(
            answer="Generated answer. [1]",
            citations=[
                Citation(
                    citation_id=1,
                    source_path="api.py",
                    start_line=10,
                    end_line=20,
                )
            ],
            confidence=0.8,
            refusal_reason=None,
        )

    monkeypatch.setattr(
        query_manifest_module,
        "build_grounded_answer",
        fake_build_grounded_answer,
    )

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--answer",
            "--min-score",
            "0.4",
        ]
    )

    assert exit_code == 0
    assert captured["query"] == "prediction api"
    assert captured["min_score"] == 0.4
    assert captured["generator"] is generator
    assert len(captured["scored_chunks"]) == 1


def test_answer_mode_prints_human_readable_provider_result(
    monkeypatch,
    tmp_path,
    capsys,
):
    patch_successful_retrieval(monkeypatch)

    manifest_path = create_test_manifest(tmp_path)
    generator = RecordingGroundedAnswerGenerator(
        result=GroundedAnswer(
            answer="The endpoint is POST /predict. [1]",
            citations=[
                Citation(
                    citation_id=1,
                    source_path="api.py",
                    start_line=10,
                    end_line=20,
                )
            ],
            confidence=0.9,
            refusal_reason=None,
        )
    )

    monkeypatch.setattr(
        query_manifest_module,
        "load_api_settings",
        lambda: object(),
    )
    monkeypatch.setattr(
        query_manifest_module,
        "create_grounded_answer_generator",
        lambda settings: generator,
    )

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--answer",
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Answer: The endpoint is POST /predict. [1]" in output
    assert "Confidence: 0.9000" in output
    assert "Citations:" in output
    assert "[1] api.py:10-20" in output


def test_answer_mode_prints_json_provider_result(
    monkeypatch,
    tmp_path,
    capsys,
):
    patch_successful_retrieval(monkeypatch)

    manifest_path = create_test_manifest(tmp_path)
    generator = RecordingGroundedAnswerGenerator(
        result=GroundedAnswer(
            answer="The endpoint is POST /predict. [1]",
            citations=[
                Citation(
                    citation_id=1,
                    source_path="api.py",
                    start_line=10,
                    end_line=20,
                )
            ],
            confidence=0.9,
            refusal_reason=None,
        )
    )

    monkeypatch.setattr(
        query_manifest_module,
        "load_api_settings",
        lambda: object(),
    )
    monkeypatch.setattr(
        query_manifest_module,
        "create_grounded_answer_generator",
        lambda settings: generator,
    )

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--answer",
            "--json",
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"answer":"The endpoint is POST /predict. [1]"' in output
    assert '"confidence":0.9' in output
    assert '"refusal_reason":null' in output


def test_answer_mode_prints_refusal(
    monkeypatch,
    tmp_path,
    capsys,
):
    patch_successful_retrieval(monkeypatch)

    manifest_path = create_test_manifest(tmp_path)
    generator = RecordingGroundedAnswerGenerator(
        result=GroundedAnswer(
            answer="",
            citations=[],
            confidence=0.0,
            refusal_reason="The context is insufficient.",
        )
    )

    monkeypatch.setattr(
        query_manifest_module,
        "load_api_settings",
        lambda: object(),
    )
    monkeypatch.setattr(
        query_manifest_module,
        "create_grounded_answer_generator",
        lambda settings: generator,
    )

    exit_code = main(
        [
            str(manifest_path),
            "prediction api",
            "--answer",
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Answer: " in output
    assert "Confidence: 0.0000" in output
    assert "Refusal reason: The context is insufficient." in output
    assert "Citations:" in output


def test_answer_mode_provider_error_propagates(
    monkeypatch,
    tmp_path,
):
    patch_successful_retrieval(monkeypatch)

    manifest_path = create_test_manifest(tmp_path)
    generator = RecordingGroundedAnswerGenerator(
        error=GroundedAnswerProviderError(
            "Provider request failed."
        )
    )

    monkeypatch.setattr(
        query_manifest_module,
        "load_api_settings",
        lambda: object(),
    )
    monkeypatch.setattr(
        query_manifest_module,
        "create_grounded_answer_generator",
        lambda settings: generator,
    )

    with pytest.raises(
        GroundedAnswerProviderError,
        match="Provider request failed.",
    ):
        main(
            [
                str(manifest_path),
                "prediction api",
                "--answer",
            ]
        )


def make_chunk(
    chunk_id: str,
    content: str,
    source_path: str = "source.py",
    start_line: int = 1,
    end_line: int = 2,
) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        source_id=source_path,
        project_name="test-project",
        source_type="python",
        source_path=source_path,
        chunk_index=0,
        content=content,
        start_line=start_line,
        end_line=end_line,
    )