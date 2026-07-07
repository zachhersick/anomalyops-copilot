import os
from pathlib import Path

from pydantic import BaseModel


MANIFEST_PATH_ENV_VAR = "ANOMALYOPS_MANIFEST_PATH"


class ApiSettings(BaseModel):
    manifest_path: Path | None = None
    
    
def load_api_settings() -> ApiSettings:
    manifest_path = os.environ.get(MANIFEST_PATH_ENV_VAR)
    
    if manifest_path is None:
        return ApiSettings()
    
    return ApiSettings(manifest_path=Path(manifest_path))
    