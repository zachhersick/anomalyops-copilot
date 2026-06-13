from copilot.schemas.chunk import SourceChunk
from copilot.schemas.source import SourceDocument

def split_text_by_lines(text: str, max_lines: int, overlap_lines: int) -> list[tuple[int, int, str]]:
    if max_lines <= 0:
        raise ValueError("max_lines must be greater than 0")
    
    if overlap_lines < 0:
        raise ValueError("overlap_lines must be greater than or equal to 0")
    
    if overlap_lines >= max_lines:
        raise ValueError("overlap_lines must be less than max_lines")
    
    lines = text.splitlines()
    chunks = []
    step = max_lines - overlap_lines
    start_index = 0
    
    while start_index < len(lines):
        end_index = min(start_index + max_lines, len(lines))
        chunk_lines = lines[start_index:end_index]
        
        start_line = start_index + 1
        end_line = end_index
        content = "\n".join(chunk_lines)
        
        chunks.append((start_line, end_line, content))
        
        if end_index == len(lines):
            break
        
        start_index += step
            
    return chunks


def build_chunk_id(source_id: str, chunk_index: int) -> str:
    return f'{source_id}#chunk-{chunk_index:04d}'
            
        
def chunk_source_document(
    document: SourceDocument,
    max_lines: int=80,
    overlap_lines: int=10,
) -> list[SourceChunk]:
    line_chunks = split_text_by_lines(document.content, max_lines, overlap_lines)
    chunks = []
    
    for chunk_index, (start_line, end_line, content) in enumerate(line_chunks):
        source_chunk = SourceChunk(
            chunk_id=build_chunk_id(document.source_id, chunk_index),
            source_id=document.source_id,
            project_name=document.project_name,
            source_type=document.source_type,
            source_path=document.path,
            chunk_index=chunk_index,
            content=content,
            start_line=start_line,
            end_line=end_line,
        )
        
        chunks.append(source_chunk)
    
    return chunks
    