from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_exists():
    assert (ROOT / "Dockerfile").is_file()


def test_dockerignore_exists():
    assert (ROOT / ".dockerignore").is_file()


def test_dockerfile_uses_python_311():
    dockerfile = (
        ROOT / "Dockerfile"
    ).read_text(
        encoding="utf-8",
    )

    assert "FROM python:3.11-slim" in dockerfile


def test_dockerfile_installs_project():
    dockerfile = (
        ROOT / "Dockerfile"
    ).read_text(
        encoding="utf-8",
    )

    assert (
        "RUN pip install --no-cache-dir ."
        in dockerfile
    )


def test_dockerfile_generates_manifest():
    dockerfile = (
        ROOT / "Dockerfile"
    ).read_text(
        encoding="utf-8",
    )

    assert (
        "python scripts/ingest_sources.py"
        in dockerfile
    )
    assert (
        "data_sources/anomaly_detection_platform"
        in dockerfile
    )
    assert (
        "--output outputs/chunks.json"
        in dockerfile
    )


def test_dockerfile_configures_manifest_backend():
    dockerfile = (
        ROOT / "Dockerfile"
    ).read_text(
        encoding="utf-8",
    )

    assert (
        "ANOMALYOPS_RETRIEVAL_BACKEND=manifest"
        in dockerfile
    )
    assert (
        "ANOMALYOPS_MANIFEST_PATH="
        "/app/outputs/chunks.json"
        in dockerfile
    )


def test_dockerfile_defaults_to_deterministic_provider():
    dockerfile = (
        ROOT / "Dockerfile"
    ).read_text(
        encoding="utf-8",
    )

    assert (
        "ANOMALYOPS_AI_PROVIDER=deterministic"
        in dockerfile
    )


def test_dockerfile_exposes_api_port():
    dockerfile = (
        ROOT / "Dockerfile"
    ).read_text(
        encoding="utf-8",
    )

    assert "EXPOSE 8000" in dockerfile


def test_dockerfile_has_healthcheck():
    dockerfile = (
        ROOT / "Dockerfile"
    ).read_text(
        encoding="utf-8",
    )

    assert "HEALTHCHECK" in dockerfile
    assert "/health" in dockerfile


def test_dockerfile_starts_uvicorn():
    dockerfile = (
        ROOT / "Dockerfile"
    ).read_text(
        encoding="utf-8",
    )

    assert '"uvicorn"' in dockerfile
    assert '"copilot.api.app:app"' in dockerfile
    assert '"0.0.0.0"' in dockerfile
    assert '"8000"' in dockerfile


def test_dockerignore_excludes_local_environment():
    dockerignore = (
        ROOT / ".dockerignore"
    ).read_text(
        encoding="utf-8",
    )

    assert ".venv" in dockerignore
    assert ".env" in dockerignore
    assert "__pycache__" in dockerignore
    assert "outputs" in dockerignore


def test_compose_exposes_api():
    compose = (
        ROOT / "docker-compose.yml"
    ).read_text(
        encoding="utf-8",
    )

    assert "api:" in compose
    assert '"8000:8000"' in compose


def test_compose_keeps_pgvector():
    compose = (
        ROOT / "docker-compose.yml"
    ).read_text(
        encoding="utf-8",
    )

    assert (
        "pgvector/pgvector:pg16"
        in compose
    )
    assert (
        "anomalyops_postgres_data"
        in compose
    )


def test_compose_uses_deterministic_manifest_defaults():
    compose = (
        ROOT / "docker-compose.yml"
    ).read_text(
        encoding="utf-8",
    )

    assert (
        "ANOMALYOPS_RETRIEVAL_BACKEND: manifest"
        in compose
    )
    assert (
        "ANOMALYOPS_AI_PROVIDER: deterministic"
        in compose
    )
    assert (
        "/app/outputs/chunks.json"
        in compose
    )