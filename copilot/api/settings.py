import os

from pathlib import Path
from typing import Literal

from pydantic import BaseModel


RETRIEVAL_BACKEND_ENV_VAR = "ANOMALYOPS_RETRIEVAL_BACKEND"
DATABASE_URL_ENV_VAR = "ANOMALYOPS_DATABASE_URL"
MANIFEST_PATH_ENV_VAR = "ANOMALYOPS_MANIFEST_PATH"


class ApiSettings(BaseModel):
    retrieval_backend: Literal["manifest", "pgvector"] = "manifest"
    manifest_path: Path | None = None
    database_url: str | None = None
    anomaly_api_base_url: str | None = None
    
    
def load_api_settings() -> ApiSettings:
    retrieval_backend_value = os.environ.get(
        RETRIEVAL_BACKEND_ENV_VAR,
        "manifest",
    )

    if retrieval_backend_value == "manifest":
        retrieval_backend: Literal["manifest", "pgvector"] = "manifest"
    elif retrieval_backend_value == "pgvector":
        retrieval_backend = "pgvector"
    else:
        raise ValueError(
            "ANOMALYOPS_RETRIEVAL_BACKEND must be "
            "'manifest' or 'pgvector'"
        )

    manifest_path_value = os.environ.get(
        MANIFEST_PATH_ENV_VAR,
    )
    database_url = os.environ.get(
        DATABASE_URL_ENV_VAR,
    )

    manifest_path = (
        Path(manifest_path_value)
        if manifest_path_value is not None
        else None
    )
    
    api_base_url = os.getenv("ANOMALYOPS_ANOMALY_API_BASE_URL")

    return ApiSettings(
        retrieval_backend=retrieval_backend,
        manifest_path=manifest_path,
        database_url=database_url,
        anomaly_api_base_url=api_base_url,
    )