from fastapi.testclient import TestClient

from copilot.api.app import create_app


def test_query_frontend_is_served_safely():
    response = TestClient(create_app()).get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="query-form"' in response.text
    assert "Ask Copilot" in response.text
    assert "innerHTML" not in response.text
