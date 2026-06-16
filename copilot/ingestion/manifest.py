import json
from pathlib import Path

from copilot.schemas.chunk import SourceChunk

def write_chunk_manifest(chunks: list[SourceChunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = []
    
    for chunk in chunks:
        chunk_dict = chunk.model_dump()
        data.append(chunk_dict)
        
    json_text = json.dumps(data, indent=2)
    
    output_path.write_text(json_text, encoding="utf-8")
    
    
def load_chunk_manifest(input_path: Path) -> list[SourceChunk]:
    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        
    source_chunks = []
    
    for chunk_dict in data:
        source_chunk = SourceChunk(**chunk_dict)
        source_chunks.append(source_chunk)
        
    return source_chunks
    