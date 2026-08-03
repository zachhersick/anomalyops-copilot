from copilot.providers.errors import (
    TriageAgentResourceNotFoundError,
    TriageAgentToolError,
)
from copilot.schemas.triage import (
    TriageReport,
    TriageRequest,
)
from copilot.services.triage import (
    TriageRunNotFoundError,
    TriageService,
    TriageServiceError,
)
from copilot.tools.anomaly import AnomalyOperationalTools


class DeterministicTriageAgent:
    provider_name = "deterministic"
    model_name = "deterministic-triage-v1"

    def __init__(
        self,
        tools: AnomalyOperationalTools,
    ) -> None:
        self._service = TriageService(tools)

    def triage(
        self,
        request: TriageRequest,
    ) -> TriageReport:
        try:
            return self._service.triage(request)
        except TriageRunNotFoundError as exc:
            raise TriageAgentResourceNotFoundError(
                "The requested run was not found."
            ) from exc
        except TriageServiceError as exc:
            raise TriageAgentToolError(
                "Triage failed while retrieving operational evidence."
            ) from exc