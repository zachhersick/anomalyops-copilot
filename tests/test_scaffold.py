from pathlib import Path


def test_project_scaffold_exists():
    assert Path("copilot").exists()
    assert Path("data_sources/anomaly_detection_platform").exists()
    assert Path("docker-compose.yml").exists()
    assert Path("pyproject.toml").exists()


def test_source_snapshot_contains_old_platform_context():
    source_root = Path("data_sources/anomaly_detection_platform/source_code")

    assert (source_root / "api.py").exists()
    assert (source_root / "db.py").exists()
    assert (source_root / "Dockerfile.api").exists()
    assert (source_root / "Dockerfile.dashboard").exists()