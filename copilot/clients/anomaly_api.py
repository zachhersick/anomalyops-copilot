import json

import httpx
from pydantic import ValidationError

from copilot.schemas.anomaly import LatestRun, RunSummary, AlertEvent, RowAlert


class AnomalyApiError(Exception):
    pass


class AnomalyResourceNotFoundError(AnomalyApiError):
    pass


class InvalidAnomalyApiResponseError(AnomalyApiError):
    pass


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
        
    
    def _raise_for_status(
        self,
        response: httpx.Response,
    ) -> None:
        if response.status_code == 404:
            raise AnomalyResourceNotFoundError(
                "Anomaly API resource was not found."
            )
        
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AnomalyApiError(
                f"Anomaly API request failed with status {response.status_code}."
            ) from exc
            
    
    def _decode_response_json(
        self,
        response: httpx.Response
    ) -> object:
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise InvalidAnomalyApiResponseError(
                "Anomaly API returned invalid JSON."
            ) from exc
            
        
    def _get(
        self,
        path: str,
        params: dict[str, str | int] | None = None,
    ) -> httpx.Response:
        try:
            response = self._client.get(
                path,
                params=params,
            )
        except httpx.RequestError as exc:
            raise AnomalyApiError(
                "Could not connect to the anomaly API."
            ) from exc
            
        self._raise_for_status(response)
        
        return response
        
        
    def get_latest_run(self) -> LatestRun:
        response = self._get("/runs/latest")
        
        payload = self._decode_response_json(response)
        
        try:
            return LatestRun.model_validate(payload)
        except ValidationError as exc:
            raise InvalidAnomalyApiResponseError(
                "Anomaly API response did not match the expected schema."
            ) from exc
    
    
    def get_run_summary(self, run_id: int) -> RunSummary:
        if run_id <= 0:
            raise ValueError("run_id must be positive.")
        
        response = self._get(f"/runs/{run_id}/summary")
        
        payload = self._decode_response_json(response)
            
        try:
            return RunSummary.model_validate(payload)
        except ValidationError as exc:
            raise InvalidAnomalyApiResponseError(
                "Anomaly API response did not match the expected schema."
            ) from exc
    
    
    def list_alert_events(
        self,
        run_id: int,
        severity: str | None = None,
        sensor: str | None = None,
        anomaly_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AlertEvent]:
        if run_id <= 0:
            raise ValueError("run_id must be positive.")
        if limit <= 0:
            raise ValueError("limit must be positive.")
        if limit > 500:
            raise ValueError("limit must not be greater than 500.")
        if offset < 0:
            raise ValueError("offset must not be negative.")

        params: dict[str, str | int] = {
            "limit": limit,
            "offset": offset,
        }
        
        if severity is not None:
            params["severity"] = severity
        if sensor is not None:
            params["sensor"] = sensor
        if anomaly_type is not None:
            params["anomaly_type"] = anomaly_type
        
        response = self._get(
            f"/runs/{run_id}/events",
            params=params,
        )
        
        payload = self._decode_response_json(response)
            
        try:
            if not isinstance(payload, list):
                raise InvalidAnomalyApiResponseError(
                    "Anomaly API response did not match the expected schema."
                )
            alert_events = []
            for json_dict in payload:
                validated_model = AlertEvent.model_validate(json_dict)
                alert_events.append(validated_model)
                
            return alert_events
        except ValidationError as exc:
            raise InvalidAnomalyApiResponseError(
                "Anomaly API response did not match the expected schema."
            ) from exc
    
    
    def get_event_alerts(
        self,
        run_id: int,
        event_id: int,
    ) -> list[RowAlert]:
        if run_id <= 0:
            raise ValueError("run_id must be positive.")
        if event_id <= 0:
            raise ValueError("event_id must be positive.")
        
        response = self._get(
            f"/runs/{run_id}/events/{event_id}/alerts",
        )
        
        payload = self._decode_response_json(response)
            
        try:
            if not isinstance(payload, list):
                raise InvalidAnomalyApiResponseError(
                    "Anomaly API response did not match the expected schema."
                )
            event_alerts = []
            for json_dict in payload:
                validated_model = RowAlert.model_validate(json_dict)
                event_alerts.append(validated_model)
                
            return event_alerts
        except ValidationError as exc:
            raise InvalidAnomalyApiResponseError(
                "Anomaly API response did not match the expected schema."
            ) from exc