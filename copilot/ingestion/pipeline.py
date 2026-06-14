from pathlib import Path

from copilot.schemas.chunk import SourceChunk
from copilot.ingestion.source_discovery import discover_source_documents
from copilot.ingestion.chunking import chunk_source_documents


def ingest_local_sources(
    root: Path,
    max_lines: int = 80,
    overlap_lines: int = 10,
) -> list[SourceChunk]:
    discovered_documents = discover_source_documents(root)
    
    return chunk_source_documents(
        discovered_documents,
        max_lines=max_lines,
        overlap_lines=overlap_lines
    )