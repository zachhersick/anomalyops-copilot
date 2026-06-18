import argparse
from pathlib import Path

from copilot.ingestion.manifest import load_chunk_manifest
from copilot.retrieval.search import retrieve_relevant_chunks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    
    source_chunks = load_chunk_manifest(Path(args.manifest_path))
    
    scored_chunks = retrieve_relevant_chunks(args.query, source_chunks, args.top_k)
    
    for scored_chunk in scored_chunks:
        chunk = scored_chunk.chunk
        print(f"Score: {scored_chunk.score}")
        print(f"Source: {chunk.source_path}:{chunk.start_line}-{chunk.end_line}")
        print(f"Preview: {chunk.content[:200]}")
        print()
    
    return 0
        


if __name__ == "__main__":
    raise SystemExit(main())