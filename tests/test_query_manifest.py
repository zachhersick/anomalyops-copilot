import pytest
import json

from copilot.ingestion.manifest import write_chunk_manifest
from copilot.schemas.chunk import SourceChunk
from scripts.query_manifest import main


def test_query_manifest_loads_manifest_and_prints_query_results(tmp_path, capsys):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk("chunk-1", "prediction api endpoint"),
        make_chunk("chunk-2", "dashboard summary view"),
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main([str(manifest_path), "prediction api", "--top-k", "2"])

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

    exit_code = main([str(manifest_path), "prediction api", "--top-k", "1"])

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
        main([str(manifest_path), "prediction api", "--top-k", "0"])


def test_query_manifest_answer_mode_prints_grounded_answer_fields(tmp_path, capsys):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk("chunk-1", "prediction api endpoint"),
    ]
    write_chunk_manifest(chunks, manifest_path)

    exit_code = main([str(manifest_path), "prediction api", "--answer"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Answer:" in captured.out
    assert "Confidence:" in captured.out
    assert "Citations:" in captured.out


def test_query_manifest_answer_mode_includes_citation_source(tmp_path, capsys):
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

    exit_code = main([str(manifest_path), "prediction api", "--answer"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "[1] api.py:10-20" in captured.out


def test_query_manifest_answer_mode_respects_top_k(tmp_path, capsys):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk("chunk-1", "prediction api endpoint", source_path="api.py"),
        make_chunk("chunk-2", "dashboard summary view", source_path="dashboard.py"),
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
    
    
def test_query_manifest_answer_json_mode_prints_valid_json(tmp_path, capsys):
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
    tmp_path,
    capsys,
):
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

    exit_code = main([str(manifest_path), "prediction api", "--answer"])

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

    exit_code = main([str(manifest_path), "prediction api"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Score:" in captured.out
    assert "Source:" in captured.out
    assert "Preview:" in captured.out
    
    
def test_query_manifest_answer_json_mode_respects_top_k(tmp_path, capsys):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk("chunk-1", "prediction api endpoint", source_path="api.py"),
        make_chunk("chunk-2", "dashboard summary view", source_path="dashboard.py"),
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
            "--min-score",
            "1.1",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Answer:" in captured.out
    assert "Confidence: 0.0000" in captured.out
    assert (
        "Refusal reason: Retrieved context was below the confidence threshold."
        in captured.out
    )
    assert "Citations:" in captured.out
    
    
def test_query_manifest_answer_json_mode_refuses_when_min_score_is_too_high(
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
            "--min-score",
            "1.1",
        ]
    )

    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["answer"] == ""
    assert data["citations"] == []
    assert data["confidence"] == 0.0
    assert (
        data["refusal_reason"]
        == "Retrieved context was below the confidence threshold."
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