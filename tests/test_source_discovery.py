from pathlib import Path

from copilot.ingestion.source_discovery import (
    build_source_id,
    discover_source_documents,
    get_source_type,
    should_include_file,
)


def test_should_include_python_file(tmp_path):
    path = tmp_path / "api.py"
    path.write_text("print('hello')", encoding="utf-8")

    assert should_include_file(path)


def test_should_skip_unsupported_file(tmp_path):
    path = tmp_path / "diagram.png"
    path.write_text("fake image", encoding="utf-8")

    assert not should_include_file(path)


def test_should_skip_file_inside_ignored_directory(tmp_path):
    ignored_dir = tmp_path / "__pycache__"
    ignored_dir.mkdir()

    path = ignored_dir / "api.py"
    path.write_text("print('hello')", encoding="utf-8")

    assert not should_include_file(path)


def test_get_source_type_for_python_file():
    path = Path("source_code/api.py")

    assert get_source_type(path) == "python"


def test_get_source_type_for_yaml_file():
    path = Path(".github/workflows/ci.yml")

    assert get_source_type(path) == "yaml"


def test_build_source_id_uses_relative_path():
    root = Path("data_sources/anomaly_detection_platform")
    path = root / "source_code" / "api.py"

    assert build_source_id(path, root) == "anomaly_detection_platform:source_code/api.py"


def test_discover_source_documents_returns_documents():
    root = Path("data_sources/anomaly_detection_platform")

    documents = discover_source_documents(root)

    assert documents
    assert any(document.path == "source_code/api.py" for document in documents)
    assert all(document.source_id for document in documents)
    assert all(document.content for document in documents)


def test_discover_source_documents_returns_stable_order():
    root = Path("data_sources/anomaly_detection_platform")

    documents = discover_source_documents(root)
    source_ids = [document.source_id for document in documents]

    assert source_ids == sorted(source_ids)
    
    
def test_should_include_dockerfile_api(tmp_path):
    path = tmp_path / "Dockerfile.api"
    path.write_text("FROM python:3.11-slim", encoding="utf-8")

    assert should_include_file(path)
    
    
def test_should_include_dockerfile_dashboard(tmp_path):
    path = tmp_path / "Dockerfile.dashboard"
    path.write_text("FROM python:3.11-slim", encoding="utf-8")

    assert should_include_file(path)
    
    
def test_discover_source_documents_includes_dockerfiles():
    root = Path("data_sources/anomaly_detection_platform")

    documents = discover_source_documents(root)
    paths = {document.path for document in documents}

    assert "source_code/Dockerfile.api" in paths
    assert "source_code/Dockerfile.dashboard" in paths
    
    
def test_get_source_type_for_dockerfile_variant():
    path = Path("source_code/Dockerfile.api")

    assert get_source_type(path) == "dockerfile"