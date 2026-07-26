import pytest

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
    
    
def test_load_api_settings_reads_anomaly_api_base_url(monkeypatch):
    anomaly_api_base_url = "http://anomaly-api.test"

    monkeypatch.setenv(
        "ANOMALYOPS_ANOMALY_API_BASE_URL",
        anomaly_api_base_url,
    )

    settings = load_api_settings()

    assert settings.anomaly_api_base_url == anomaly_api_base_url


def test_load_api_settings_defaults_anomaly_api_base_url_to_none(monkeypatch):
    monkeypatch.delenv(
        "ANOMALYOPS_ANOMALY_API_BASE_URL",
        raising=False,
    )

    settings = load_api_settings()

    assert settings.anomaly_api_base_url is None
    

def test_api_settings_has_deterministic_ai_defaults():
    settings = ApiSettings()

    assert settings.ai_provider == "deterministic"
    assert settings.embedding_model is None
    assert settings.grounded_answer_model is None
    assert settings.triage_model is None
    assert settings.openai_api_key is None


def test_load_api_settings_uses_deterministic_ai_defaults(
    monkeypatch,
):
    monkeypatch.delenv("ANOMALYOPS_AI_PROVIDER", raising=False)
    monkeypatch.delenv("ANOMALYOPS_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv(
        "ANOMALYOPS_GROUNDED_ANSWER_MODEL",
        raising=False,
    )
    monkeypatch.delenv("ANOMALYOPS_TRIAGE_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = load_api_settings()

    assert settings.ai_provider == "deterministic"
    assert settings.embedding_model is None
    assert settings.grounded_answer_model is None
    assert settings.triage_model is None
    assert settings.openai_api_key is None


def test_load_api_settings_reads_ai_configuration(
    monkeypatch,
):
    monkeypatch.setenv(
        "ANOMALYOPS_AI_PROVIDER",
        "openai",
    )
    monkeypatch.setenv(
        "ANOMALYOPS_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )
    monkeypatch.setenv(
        "ANOMALYOPS_GROUNDED_ANSWER_MODEL",
        "grounded-answer-model",
    )
    monkeypatch.setenv(
        "ANOMALYOPS_TRIAGE_MODEL",
        "triage-model",
    )
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-api-key",
    )

    settings = load_api_settings()

    assert settings.ai_provider == "openai"
    assert settings.embedding_model == "text-embedding-3-small"
    assert (
        settings.grounded_answer_model
        == "grounded-answer-model"
    )
    assert settings.triage_model == "triage-model"
    assert settings.openai_api_key == "test-api-key"


def test_api_settings_accepts_openai_provider():
    settings = ApiSettings(
        ai_provider="openai",
    )

    assert settings.ai_provider == "openai"


def test_load_api_settings_rejects_invalid_ai_provider(
    monkeypatch,
):
    monkeypatch.setenv(
        "ANOMALYOPS_AI_PROVIDER",
        "unsupported",
    )

    with pytest.raises(
        ValueError,
        match=(
            "ANOMALYOPS_AI_PROVIDER must be "
            "'deterministic' or 'openai'"
        ),
    ):
        load_api_settings()
        
        
def test_api_settings_defaults_embedding_dimensions_to_16():
    settings = ApiSettings()

    assert settings.embedding_dimensions == 16


def test_load_api_settings_defaults_embedding_dimensions_to_16(
    monkeypatch,
):
    monkeypatch.delenv(
        "ANOMALYOPS_EMBEDDING_DIMENSIONS",
        raising=False,
    )

    settings = load_api_settings()

    assert settings.embedding_dimensions == 16


def test_load_api_settings_reads_embedding_dimensions(
    monkeypatch,
):
    monkeypatch.setenv(
        "ANOMALYOPS_EMBEDDING_DIMENSIONS",
        "1536",
    )

    settings = load_api_settings()

    assert settings.embedding_dimensions == 1536
    assert isinstance(settings.embedding_dimensions, int)


def test_load_api_settings_rejects_non_integer_embedding_dimensions(
    monkeypatch,
):
    monkeypatch.setenv(
        "ANOMALYOPS_EMBEDDING_DIMENSIONS",
        "abc",
    )

    with pytest.raises(
        ValueError,
        match=(
            "ANOMALYOPS_EMBEDDING_DIMENSIONS "
            "must be an integer"
        ),
    ):
        load_api_settings()


@pytest.mark.parametrize(
    "dimensions",
    ["0", "-1", "-1536"],
)
def test_load_api_settings_rejects_non_positive_embedding_dimensions(
    monkeypatch,
    dimensions,
):
    monkeypatch.setenv(
        "ANOMALYOPS_EMBEDDING_DIMENSIONS",
        dimensions,
    )

    with pytest.raises(
        ValueError,
        match=(
            "ANOMALYOPS_EMBEDDING_DIMENSIONS "
            "must be positive"
        ),
    ):
        load_api_settings()