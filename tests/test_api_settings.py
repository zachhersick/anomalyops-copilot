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