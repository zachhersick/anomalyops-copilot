import argparse
import os

from dotenv import load_dotenv

from copilot.retrieval.pgvector import retrieve_relevant_chunks_from_pgvector
from copilot.storage.database import (
    create_engine_from_url,
    create_session_factory,
)


def main(argv: list[str] | None = None) -> int:  
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    
    args = parser.parse_args(argv)
    
    load_dotenv()
    
    database_url = os.getenv("ANOMALYOPS_DATABASE_URL")
    
    if database_url is None:
        raise RuntimeError(
            "ANOMALYOPS_DATABASE_URL is not configured"
        )
        
    engine = create_engine_from_url(database_url)
    
    SessionFactory = create_session_factory(engine)
    
    with SessionFactory() as session:
        chunks = retrieve_relevant_chunks_from_pgvector(
            session,
            args.query,
            top_k=args.top_k
        )
    
    for i, chunk in enumerate(chunks):
        preview = chunk.chunk.content[:120].replace("\n", " ")
        
        print(
        f"[{i+1}] "
        f"{chunk.score:.4f} "
        f"{chunk.chunk.source_path}:"
        f"{chunk.chunk.start_line}-"
        f"{chunk.chunk.end_line} "
        f"{preview}"
    )
                
    return 0


if __name__ == "__main__":
    raise SystemExit(main())