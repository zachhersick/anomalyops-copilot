import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from copilot.ingestion.manifest import load_chunk_manifest
from copilot.storage.chunks import store_source_chunks
from copilot.storage.database import (
    create_engine_from_url,
    create_session_factory,
    initialize_database,
)


def main(argv: list[str] | None = None) -> int:  
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path")
    
    args = parser.parse_args(argv)
    
    load_dotenv()
    
    database_url = os.getenv("ANOMALYOPS_DATABASE_URL")
    
    if database_url is None:
        raise RuntimeError(
            "ANOMALYOPS_DATABASE_URL is not configured"
        )
        
    chunks = load_chunk_manifest(Path(args.manifest_path))
        
    engine = create_engine_from_url(database_url)
    initialize_database(engine)
    
    SessionFactory = create_session_factory(engine)
    
    with SessionFactory() as session:
        stored_chunks = store_source_chunks(session, chunks)
    
    print(f"Stored {stored_chunks} source chunks.")
                
    return 0


if __name__ == "__main__":
    raise SystemExit(main())