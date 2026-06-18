from copilot.schemas.retrieval import ScoredChunk


def format_retrieval_context(scored_chunks: list[ScoredChunk]) -> str:
    if not scored_chunks:
        return ""
    
    formatted_blocks = []
    
    for i, scored_chunk in enumerate(scored_chunks, start=1):
        chunk = scored_chunk.chunk
        header = (
            f"[{i}] "
            f"{chunk.source_path}:{chunk.start_line}-{chunk.end_line} "
            f"score={scored_chunk.score:.4f}"
        )
        block = f"{header}\n{chunk.content}"
        formatted_blocks.append(block)
    
    return "\n\n".join(formatted_blocks)
    