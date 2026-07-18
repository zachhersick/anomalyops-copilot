from pathlib import Path

from copilot.api.app import create_app
from copilot.api.settings import ApiSettings, load_api_settings


def test_api_settings_defaults_manifest_path_to_none():
    settings = ApiSettings()

    assert settings.manifest_path is None


def test_api_settings_accepts_manifest_path():
    manifest_path = Path("outputs/chunks.json")

    settings = ApiSettings(manifest_path=manifest_path)

    assert settings.manifest_path == manifest_path


def test_create_app_stores_default_settings():
    app = create_app()

    assert isinstance(app.state.settings, ApiSettings)
    assert app.state.settings.manifest_path is None


def test_create_app_stores_provided_settings():
    manifest_path = Path("outputs/chunks.json")
    settings = ApiSettings(manifest_path=manifest_path)

    app = create_app(settings=settings)

    assert app.state.settings is settings
    assert app.state.settings.manifest_path == manifest_path
    
    
def test_load_api_settings_defaults_manifest_path_to_none(monkeypatch):
    monkeypatch.delenv("ANOMALYOPS_MANIFEST_PATH", raising=False)

    settings = load_api_settings()

    assert settings.manifest_path is None


def test_load_api_settings_reads_manifest_path_from_environment(monkeypatch):
    monkeypatch.setenv("ANOMALYOPS_MANIFEST_PATH", "outputs/chunks.json")

    settings = load_api_settings()

    assert settings.manifest_path == Path("outputs/chunks.json")
    
    
def test_api_settings_defaults_retrieval_backend_to_manifest():
    settings = ApiSettings()
    
    assert settings.retrieval_backend == "manifest"
    
    
def test_api_settings_defaults_database_url_to_none():
    settings = ApiSettings(retrieval_backend="pgvector")
    
    assert settings.database_url is None
    
    
def test_load_api_settings_defaults_to_manifest_backend(monkeypatch):
    monkeypatch.delenv(
        "ANOMALYOPS_RETRIEVAL_BACKEND",
        raising=False,
    )
    monkeypatch.delenv(
        "ANOMALYOPS_MANIFEST_PATH",
        raising=False,
    )
    monkeypatch.delenv(
        "ANOMALYOPS_DATABASE_URL",
        raising=False,
    )

    settings = load_api_settings()

    assert settings.retrieval_backend == "manifest"


def test_load_api_settings_reads_manifest_path(monkeypatch):
    monkeypatch.setenv(
        "ANOMALYOPS_MANIFEST_PATH",
        "outputs/chunks.json",
    )

    settings = load_api_settings()

    assert settings.manifest_path == Path("outputs/chunks.json")


def test_load_api_settings_reads_pgvector_backend(monkeypatch):
    monkeypatch.setenv(
        "ANOMALYOPS_RETRIEVAL_BACKEND",
        "pgvector",
    )

    settings = load_api_settings()

    assert settings.retrieval_backend == "pgvector"


def test_load_api_settings_reads_database_url(monkeypatch):
    database_url = (
        "postgresql+psycopg://"
        "anomalyops:anomalyops@localhost:5432/anomalyops"
    )

    monkeypatch.setenv(
        "ANOMALYOPS_DATABASE_URL",
        database_url,
    )

    settings = load_api_settings()

    assert settings.database_url == database_url