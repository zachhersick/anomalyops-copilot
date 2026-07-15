from copilot.storage.chunks import source_chunk_to_values
from copilot.schemas.chunk import SourceChunk
from copilot.retrieval.embeddings import embed_text
from copilot.storage.models import EMBEDDING_DIMENSIONS


def test_source_chunk_to_values_every_chunk_field_is_copied():
    values_dict = source_chunk_to_values(make_chunk("chunk-1", "text"))
    
    assert len(values_dict) == 10
    assert values_dict["chunk_id"] == "chunk-1"
    assert values_dict["source_id"] == "source.py"
    assert values_dict["project_name"] == "test-project"
    assert values_dict["source_type"] == "python"
    assert values_dict["source_path"] == "source.py"
    assert values_dict["chunk_index"] == 0
    assert values_dict["content"] == "text"
    assert values_dict["start_line"] == 1
    assert values_dict["end_line"] == 2
    assert values_dict["embedding"] == embed_text("text", EMBEDDING_DIMENSIONS)
    
    
def test_source_chunk_to_values_embedding_contains_16_values():
    values_dict = source_chunk_to_values(make_chunk("chunk-1", "text"))
    
    assert len(values_dict["embedding"]) == EMBEDDING_DIMENSIONS


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