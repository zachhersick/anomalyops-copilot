from copilot.ingestion.pipeline import ingest_local_sources


def test_ingest_local_sources_returns_chunks_from_both_files(tmp_path):
    source_root = tmp_path / "sample_project"
    source_root.mkdir()
    
    first_file = source_root / "first.py"
    first_file.write_text("print('one')\nprint('two')\n", encoding="utf-8")
    
    second_file = source_root / "second.py"
    second_file.write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")
    
    chunks = ingest_local_sources(source_root, 10, 2)
    
    assert len(chunks) == 2
    
    source_paths = {chunk.source_path for chunk in chunks}
    assert source_paths == {"first.py", "second.py"}
    
    contents = {chunk.content for chunk in chunks}
    assert "print('one')\nprint('two')" in contents
    assert "x = 1\ny = 2\nz = 3" in contents
    
    for chunk in chunks:
        assert chunk.project_name == "sample_project"
        assert chunk.source_type == "python"
        assert chunk.chunk_index == 0
        assert chunk.start_line == 1
        assert chunk.end_line in {2, 3}