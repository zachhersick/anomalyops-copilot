from copilot.ingestion.manifest import write_chunk_manifest
from copilot.schemas.chunk import SourceChunk
from scripts.query_manifest import main


def test_local_cli_builds_grounded_answer_with_context_from_manifest(tmp_path, capsys):
    manifest_path = tmp_path / "chunks.json"
    chunks = [
        make_chunk(
            "chunk-1",
            "The prediction API exposes a POST /predict endpoint for model inference.",
            source_path="api.py",
            start_line=10,
            end_line=20
        ),
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
    output = captured.out

    assert exit_code == 0
    assert "Answer:" in output
    assert "Confidence:" in output
    assert "Citations:" in output
    assert "Context:" in output
    assert "[1] api.py:10-20" in output
    
    
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