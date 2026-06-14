from scripts.ingest_sources import main


def test_ingest_sources_script_prints_chunk_summary(tmp_path, capsys):
    source_root = tmp_path / "sample_project"
    source_root.mkdir()

    source_file = source_root / "example.py"
    source_file.write_text("print('hello')\nprint('world')\n", encoding="utf-8")
    
    exit_code = main([str(source_root)])
    
    captured = capsys.readouterr()
    
    assert exit_code == 0
    assert "Source documents:" in captured.out
    assert "Source chunks:" in captured.out
    assert "1" in captured.out