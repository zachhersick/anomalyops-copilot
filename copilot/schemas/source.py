from pydantic import BaseModel


class SourceDocument(BaseModel):
    source_id: str
    project_name: str
    source_type: str
    path: str
    file_name: str
    extension: str
    content: str