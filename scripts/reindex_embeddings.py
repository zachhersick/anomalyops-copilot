import argparse
from pathlib import Path

from dotenv import load_dotenv

from copilot.api.settings import (
    load_api_settings,
)
from copilot.ingestion.manifest import (
    load_chunk_manifest,
)
from copilot.providers.errors import (
    EmbeddingProviderError,
)
from copilot.providers.factory import (
    create_embedding_provider,
)
from copilot.storage.chunks import (
    replace_source_chunks,
)
from copilot.storage.database import (
    create_engine_from_url,
    create_session_factory,
    initialize_database,
)


def main(
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Safely replace the pgvector source "
            "chunk index."
        )
    )
    parser.add_argument(
        "manifest_path",
        type=Path,
    )

    args = parser.parse_args(argv)

    load_dotenv()

    settings = load_api_settings()

    if settings.database_url is None:
        raise RuntimeError(
            "ANOMALYOPS_DATABASE_URL "
            "is not configured"
        )

    embedding_provider = (
        create_embedding_provider(
            settings
        )
    )

    chunks = load_chunk_manifest(
        args.manifest_path
    )

    engine = create_engine_from_url(
        settings.database_url
    )
    initialize_database(engine)

    SessionFactory = (
        create_session_factory(engine)
    )

    try:
        with SessionFactory() as session:
            stored_chunks = (
                replace_source_chunks(
                    session=session,
                    chunks=chunks,
                    embedding_provider=(
                        embedding_provider
                    ),
                )
            )
    except EmbeddingProviderError as exc:
        raise RuntimeError(
            "Embedding generation failed; "
            "the existing source chunk index "
            "was not modified."
        ) from exc
    finally:
        engine.dispose()

    print(
        f"Reindexed {stored_chunks} "
        "source chunks."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())