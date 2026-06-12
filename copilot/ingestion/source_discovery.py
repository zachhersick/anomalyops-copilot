from pathlib import Path

from copilot.schemas.source import SourceDocument


ALLOWED_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".csv",
}

SKIP_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".venv",
}


def should_include_file(path: Path) -> bool:
    """
    should_include_file returns True only for real files with allowed extensions that are not inside skipped directories.
    """
    if not path.is_file():
        return False
    
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False
    
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    
    return True


def get_source_type(path: Path) -> str:
    extension = path.suffix.lower()
    
    if extension == ".py":
        return "python"
        
    if extension == ".md":
        return "markdown"
    
    if extension in {".yml", ".yaml"}:
        return "yaml"
    
    if extension == ".json":
        return "json"
    
    if extension == ".toml":
        return "toml"
    
    if extension == ".csv":
        return "csv"
    
    return "text"


def build_source_id(path: Path, root: Path) -> str:
    relative_path = path.relative_to(root).as_posix()
    project_name = root.name
    return f"{project_name}:{relative_path}"


def load_source_document(path: Path, root: Path) -> SourceDocument:
    relative_path = path.relative_to(root).as_posix()
    
    return SourceDocument(
        source_id=build_source_id(path, root),
        project_name=root.name,
        source_type=get_source_type(path),
        path=relative_path,
        file_name=path.name,
        extension=path.suffix.lower(),
        content=path.read_text(encoding="utf-8"),
    )
    
    
def discover_source_documents(root: Path) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    
    for path in root.rglob("*"):
        if should_include_file(path):
            documents.append(load_source_document(path, root))
            
    return sorted(documents, key=lambda document: document.source_id)