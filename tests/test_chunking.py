import pytest

from copilot.ingestion.chunking import (
    build_chunk_id,
    chunk_source_document,
    split_text_by_lines,
    chunk_source_documents,
)
from copilot.schemas.source import SourceDocument


def test_short_text_becomes_one_chunk():
    text = make_lines(3)
    
    chunks = split_text_by_lines(text, max_lines=10, overlap_lines=2)
    
    assert chunks == [(1, 3, "line 1\nline 2\nline 3")]
    
    
def test_long_text_splits_into_overlapping_chunks():
    text = make_lines(10)

    chunks = split_text_by_lines(text, max_lines=4, overlap_lines=1)

    assert chunks == [
        (1, 4, "line 1\nline 2\nline 3\nline 4"),
        (4, 7, "line 4\nline 5\nline 6\nline 7"),
        (7, 10, "line 7\nline 8\nline 9\nline 10"),
    ]
    
    
def test_split_text_by_line_without_overlap():
    text = make_lines(6)
    
    chunks = split_text_by_lines(text, max_lines=3, overlap_lines=0)
    
    assert chunks == [
        (1, 3, "line 1\nline 2\nline 3"),
        (4, 6, "line 4\nline 5\nline 6"),
    ]
    
    
def test_split_text_rejects_non_positive_max_lines():
    with pytest.raises(ValueError):
        split_text_by_lines("line 1", max_lines=0, overlap_lines=0)
        
        
def test_split_text_rejects_negative_overlap():
    with pytest.raises(ValueError):
        split_text_by_lines("line 1", max_lines=10, overlap_lines=-1)
        

def test_split_test_rejects_overlap_greater_than_or_equal_to_max_lines():
    with pytest.raises(ValueError):
        split_text_by_lines("line 1", max_lines=10, overlap_lines=10)
        
        
def test_build_chun_id_zero_pads_chunk_index():
    chunk_id = build_chunk_id("project:source.py", 7)
    
    assert chunk_id == "project:source.py#chunk-0007"
    
    
def test_build_chunk_id_preserves_multi_digit_index():
    chunk_id = build_chunk_id("project:source.py", 123)
    
    assert chunk_id == "project:source.py#chunk-0123"
    
    
def test_chunk_source_document_preserves_metadata():
    document = make_source_document(make_lines(3))
    
    chunks = chunk_source_document(document, max_lines=10, overlap_lines=2)
    
    assert len(chunks) == 1
    
    chunk = chunks[0]
    assert chunk.chunk_id == "project:source.py#chunk-0000"
    assert chunk.source_id == "project:source.py"
    assert chunk.project_name == "project"
    assert chunk.source_type == "python"
    assert chunk.source_path == "source.py"
    assert chunk.chunk_index == 0
    assert chunk.content == "line 1\nline 2\nline 3"
    assert chunk.start_line == 1
    assert chunk.end_line == 3
    
    
def test_chunk_source_documents_returns_flat_list():
    first_document = make_source_document(make_lines(3))
    second_document = SourceDocument(
        source_id="project:other.py",
        project_name="project",
        source_type="python",
        path="other.py",
        file_name="other.py",
        extension=".py",
        content=make_lines(3),
    )
    
    chunks = chunk_source_documents(
        [first_document, second_document],
        max_lines=10,
        overlap_lines=2,
    )
    
    assert len(chunks) == 2
    assert chunks[0].source_id == "project:source.py"
    assert chunks[1].source_id == "project:other.py"
    
    
def test_chunk_source_documents_flattens_multiple_chunks_per_document():
    first_document = make_source_document(make_lines(6))
    second_document = SourceDocument(
        source_id="project:other.py",
        project_name="project",
        source_type="python",
        path="other.py",
        file_name="other.py",
        extension=".py",
        content=make_lines(6),
    )
    
    chunks = chunk_source_documents(
        [first_document, second_document],
        max_lines=3,
        overlap_lines=0,
    )
    
    assert len(chunks) == 4
    assert [chunk.source_id for chunk in chunks] == [
        "project:source.py",
        "project:source.py",
        "project:other.py",
        "project:other.py",
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 0, 1]
    
    
def make_lines(count: int) -> str:
    return "\n".join(f"line {number}" for number in range(1, count+1))


def make_source_document(content: str) -> SourceDocument:
    return SourceDocument(
        source_id="project:source.py",
        project_name="project",
        source_type="python",
        path="source.py",
        file_name="source.py",
        extension=".py",
        content=content,
    )