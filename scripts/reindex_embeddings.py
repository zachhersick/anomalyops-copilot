import argparse
from pathlib import Path

from dotenv import load_dotenv

from copilot.ingestion.manifest import load_chunk_manifest
from copilot.storage.chunks import store_source_chunks
from copilot.storage.database import (
    create_engine_from_url,
    create_session_factory,
    initialize_database,
    rebuild_source_chunks_table,
)
from copilot.api.settings import load_api_settings
from copilot.providers.factory import create_embedding_provider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path")
    
    args = parser.parse_args(argv)
    
    load_dotenv()
    
    settings = load_api_settings()
    
    if settings.database_url is None:
        raise RuntimeError(
            "ANOMALYOPS_DATABASE_URL is not configured"
        )
        
    embedding_provider = create_embedding_provider(settings)
        
    chunks = load_chunk_manifest(Path(args.manifest_path))
        
    engine = create_engine_from_url(settings.database_url)
    initialize_database(engine)
    
    rebuild_source_chunks_table(engine)
    
    SessionFactory = create_session_factory(engine)
    
    with SessionFactory() as session:
        stored_chunks = store_source_chunks(
            session=session,
            chunks=chunks,
            embedding_provider=embedding_provider,
        )
    
    print(f"Reindexed {stored_chunks} source chunks.")
                
    return 0


if __name__ == "__main__":
    raise SystemExit(main())