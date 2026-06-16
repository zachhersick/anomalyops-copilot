import json

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
    
    
def test_script_outputs_json_chunk_records(tmp_path, capsys):
    output_path = tmp_path / "sample.json"
    
    source_root = tmp_path / "sample_project"
    source_root.mkdir()
    
    source_file = source_root / "example.py"
    source_file.write_text("print('hello')\n", encoding="utf-8")
    
    exit_code = main([str(source_root), "--output", str(output_path)])
    
    captured = capsys.readouterr()
    
    assert exit_code == 0
    assert output_path.exists()
    
    manifest_data = json.loads(output_path.read_text(encoding="utf-8"))
    
    assert isinstance(manifest_data, list)
    assert len(manifest_data) == 1
    assert manifest_data[0]["source_path"] == "example.py"
    assert manifest_data[0]["content"] == "print('hello')"
    assert "Wrote chunk manifest:" in captured.out