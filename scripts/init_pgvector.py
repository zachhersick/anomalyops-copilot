import os

from dotenv import load_dotenv

from copilot.storage.database import (
    create_engine_from_url,
    initialize_database,
)


def main() -> int:
    load_dotenv()
    
    database_url = os.getenv("ANOMALYOPS_DATABASE_URL")
    
    if database_url is None:
        raise RuntimeError(
            "ANOMALYOPS_DATABASE_URL is not configured"
        )
        
    engine = create_engine_from_url(database_url)
    initialize_database(engine)
    
    print("PostgreSQL and pgvector schema intialized.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())