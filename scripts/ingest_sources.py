import argparse
from pathlib import Path

from copilot.ingestion.pipeline import ingest_local_sources


def main(argv: list[str] | None = None) -> int:  
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root")
    
    args = parser.parse_args(argv)
    
    source_root = Path(args.source_root)
    chunks = ingest_local_sources(source_root)
    
    source_ids = {chunk.source_id for chunk in chunks}
    
    print(f"Source documents: {len(source_ids)}")
    print(f"Source chunks: {len(chunks)}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())