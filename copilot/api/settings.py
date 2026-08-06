import os

from pathlib import Path
from typing import Literal

from pydantic import BaseModel


RETRIEVAL_BACKEND_ENV_VAR = "ANOMALYOPS_RETRIEVAL_BACKEND"
DATABASE_URL_ENV_VAR = "ANOMALYOPS_DATABASE_URL"
MANIFEST_PATH_ENV_VAR = "ANOMALYOPS_MANIFEST_PATH"
ANOMALY_API_BASE_URL_ENV_VAR = "ANOMALYOPS_ANOMALY_API_BASE_URL"
AI_PROVIDER_ENV_VAR = "ANOMALYOPS_AI_PROVIDER"
EMBEDDING_MODEL_ENV_VAR = "ANOMALYOPS_EMBEDDING_MODEL"
GROUNDED_ANSWER_MODEL_ENV_VAR = "ANOMALYOPS_GROUNDED_ANSWER_MODEL"
TRIAGE_MODEL_ENV_VAR = "ANOMALYOPS_TRIAGE_MODEL"
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
EMBEDDING_DIMENSIONS_ENV_VAR = "ANOMALYOPS_EMBEDDING_DIMENSIONS"


class ApiSettings(BaseModel):
    retrieval_backend: Literal["manifest", "pgvector"] = "manifest"
    manifest_path: Path | None = None
    database_url: str | None = None
    anomaly_api_base_url: str | None = None
    ai_provider: Literal["deterministic", "openai"] = "deterministic"
    embedding_model: str | None = None
    grounded_answer_model: str | None = None
    triage_model: str | None = None
    openai_api_key: str | None = None
    embedding_dimensions: int = 256
    
    
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
    
    api_base_url = os.environ.get(
        ANOMALY_API_BASE_URL_ENV_VAR,
    )
    
    ai_provider_value = os.environ.get(
        AI_PROVIDER_ENV_VAR,
        "deterministic",
    )

    if ai_provider_value == "deterministic":
        ai_provider: Literal["deterministic", "openai"] = "deterministic"
    elif ai_provider_value == "openai":
        ai_provider = "openai"
    else:
        raise ValueError(
            "ANOMALYOPS_AI_PROVIDER must be "
            "'deterministic' or 'openai'"
        )

    embedding_model = os.environ.get(
        EMBEDDING_MODEL_ENV_VAR,
    )
    grounded_answer_model = os.environ.get(
        GROUNDED_ANSWER_MODEL_ENV_VAR,
    )
    triage_model = os.environ.get(
        TRIAGE_MODEL_ENV_VAR,
    )
    openai_api_key = os.environ.get(
        OPENAI_API_KEY_ENV_VAR,
    )
    
    embedding_dimensions_value = os.environ.get(
        EMBEDDING_DIMENSIONS_ENV_VAR,
        "256",
    )
    
    try:
        embedding_dimensions = int(
            embedding_dimensions_value,
        )
    except ValueError as exc:
        raise ValueError(
            "ANOMALYOPS_EMBEDDING_DIMENSIONS must be an integer."
        ) from exc
        
    if embedding_dimensions <= 0:
        raise ValueError(
            "ANOMALYOPS_EMBEDDING_DIMENSIONS must be positive."
        )
    
    if not isinstance(embedding_dimensions, int):
        raise ValueError("ANOMALYOPS_EMBEDDING_DIMENSIONS must be an integer.")
    
    if embedding_dimensions <= 0:
        raise ValueError("ANOMALYOPS_EMBEDDING_DIMENSIONS must be positive.")

    return ApiSettings(
        retrieval_backend=retrieval_backend,
        manifest_path=manifest_path,
        database_url=database_url,
        anomaly_api_base_url=api_base_url,
        ai_provider=ai_provider,
        embedding_model=embedding_model,
        grounded_answer_model=grounded_answer_model,
        triage_model=triage_model,
        openai_api_key=openai_api_key,
        embedding_dimensions=embedding_dimensions,
    )
