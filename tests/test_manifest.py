import json
    
from copilot.schemas.chunk import SourceChunk
from copilot.ingestion.manifest import write_chunk_manifest, load_chunk_manifest

def test_write_chunk_manifest_writes_chunks_to_json(tmp_path):
    output_path = tmp_path / "chunks.json"
    chunk = make_source_chunk(
        content="print('hello')",
        chunk_index=1,
        start_line=1,
        end_line=1,
    )
    
    write_chunk_manifest([chunk], output_path)
    
    assert output_path.exists()
    
    manifest_data = json.loads(output_path.read_text(encoding="utf-8"))
    
    assert isinstance(manifest_data, list)
    assert len(manifest_data) == 1
    assert manifest_data[0]["chunk_id"] == "project:source.py#chunk-0001"
    assert manifest_data[0]["content"] == chunk.content
    
    
def test_load_chunk_manifest_returns_source_chunks(tmp_path):
    output_path = tmp_path / "chunks.json"
    original_chunk = make_source_chunk(
        content="print('hello')",
        chunk_index=1,
        start_line=1,
        end_line=1,
    )
    
    write_chunk_manifest([original_chunk], output_path)
    
    loaded_chunks = load_chunk_manifest(output_path)
    
    assert len(loaded_chunks) == 1
    assert isinstance(loaded_chunks[0], SourceChunk)
    assert loaded_chunks[0] == original_chunk
    
    
def make_source_chunk(content: str, chunk_index: int, start_line: int, end_line: int    ) -> SourceChunk:
    return SourceChunk(
        chunk_id=f"project:source.py#chunk-{chunk_index:04d}",
        source_id="project:source.py",
        project_name="project",
        source_type="python",
        source_path="source.py",
        chunk_index=chunk_index,
        content=content,
        start_line=start_line,
        end_line=end_line,
    )