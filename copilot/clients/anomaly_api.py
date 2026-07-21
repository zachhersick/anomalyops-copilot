import httpx

from copilot.schemas.anomaly import LatestRun


class AnomalyApiClient:
    
    
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url cannot be empty or whitespace-only.")
        
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout_seconds,
            transport=transport,
        )
        
        
    def close(self) -> None:
        self._client.close()
        
        
    def __enter__(self) -> "AnomalyApiClient":
        return self
    
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
        
        
    def get_latest_run(self) -> LatestRun:
        response = self._client.get("/runs/latest")
        payload = response.json()
        
        validated_model = LatestRun.model_validate(payload)
        
        return validated_model