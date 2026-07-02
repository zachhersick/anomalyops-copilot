from fastapi.testclient import TestClient

from copilot.api.app import create_app


def test_health_check_returns_ok():
    test_app = create_app()
    client = TestClient(test_app)
    
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}