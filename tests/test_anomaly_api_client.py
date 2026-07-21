import httpx

from copilot.clients.anomaly_api import AnomalyApiClient
from copilot.schemas.anomaly import LatestRun

def handler(request):
    assert request.method == "GET"
    assert request.url.path == "/runs/latest"
    
    return httpx.Response(200, json={"run_id": 5})


def test_get_latest_run_is_valid():
    transport = httpx.MockTransport(handler)
    with AnomalyApiClient(
        "http://anomaly-api.test",
        transport=transport,
    ) as client:
        response = client.get_latest_run()
    
    assert isinstance(response, LatestRun)
    assert response.run_id == 5