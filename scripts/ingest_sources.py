






























import argparse
from pathlib import Path

from copilot.ingestion.pipeline import ingest_local_sources
from copilot.ingestion.manifest import write_chunk_manifest


def main(argv: list[str] | None = None) -> int:  
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root")
    parser.add_argument("--output")
    
    args = parser.parse_args(argv)
    
    source_root = Path(args.source_root)
    chunks = ingest_local_sources(source_root)
    
    source_ids = {chunk.source_id for chunk in chunks}
    
    print(f"Source documents: {len(source_ids)}")
    print(f"Source chunks: {len(chunks)}")
    
    if args.output is not None:
        output_path = Path(args.output)
        
        write_chunk_manifest(chunks, output_path)
        print(f"Wrote chunk manifest: {output_path}")
            
    return 0


if __name__ == "__main__":
    raise SystemExit(main())