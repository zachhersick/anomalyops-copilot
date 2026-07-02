from pathlib import Path
from pydantic import BaseModel


class ApiSettings(BaseModel):
    manifest_path: Path | None = None