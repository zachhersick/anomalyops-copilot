import pytest

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