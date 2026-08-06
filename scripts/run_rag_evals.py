import argparse
import hashlib
import json
import sys
import time

from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from copilot.api.query_service import query_service
from copilot.api.settings import ApiSettings, load_api_settings
from copilot.evals.runner import load_rag_cases, run_rag_evals
from copilot.evals.schemas import RagEvalReport
from copilot.ingestion.manifest import load_chunk_manifest
from copilot.providers.factory import (
    create_embedding_provider,
    create_grounded_answer_generator,
)
from copilot.providers.interfaces import (
    EmbeddingProvider,
    GroundedAnswerGenerator,
)
from copilot.schemas.chunk import SourceChunk
from copilot.schemas.query import QueryRequest
from copilot.storage.database import (
    create_engine_from_url,
    create_session_factory,
)
from copilot.storage.models import SourceChunkRecord


def validate_semantic_index(
    session_factory: sessionmaker[Session],
    chunks: list[SourceChunk],
    embedding_provider: EmbeddingProvider,
) -> None:
    with session_factory() as session:
        rows = session.execute(
            select(
                SourceChunkRecord.chunk_id,
                SourceChunkRecord.embedding_provider,
                SourceChunkRecord.embedding_model,
                SourceChunkRecord.embedding_dimensions,
            )
        ).all()

    expected_ids = {
        chunk.chunk_id
        for chunk in chunks
    }
    indexed_ids = {
        row.chunk_id
        for row in rows
    }
    configuration_matches = all(
        row.embedding_provider
        == embedding_provider.provider_name
        and row.embedding_model
        == embedding_provider.model_name
        and row.embedding_dimensions
        == embedding_provider.dimensions
        for row in rows
    )

    if (
        indexed_ids != expected_ids
        or len(rows) != len(chunks)
        or not configuration_matches
    ):
        raise RuntimeError(
            "The pgvector index does not match the manifest and "
            "configured embedding provider."
        )


def _build_snapshot(
    *,
    manifest_path: Path,
    chunks: list[SourceChunk],
    cases_count: int,
    top_k_values: list[int],
    settings: ApiSettings,
    answer_generator: GroundedAnswerGenerator,
    report: RagEvalReport,
    duration_seconds: float,
    embedding_provider: EmbeddingProvider | None = None,
) -> dict[str, object]:
    result_fields = (
        "case_id",
        "status",
        "schema_valid",
        "retrieval_hit",
        "citations_valid",
        "citation_hit",
        "refusal_correct",
        "first_relevant_rank",
        "relevant_source_recall",
        "answer_terms_present",
        "passed",
    )
    report_payload = report.model_dump(
        exclude={"results"}
    )
    report_payload["results"] = [
        {
            field: getattr(result, field)
            for field in result_fields
        }
        for result in report.results
    ]

    return {
        "metadata": {
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "mode": (
                "openai-pgvector"
                if embedding_provider is not None
                else "deterministic-manifest"
            ),
            "corpus_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "manifest_chunk_count": len(chunks),
            "case_count": cases_count,
            "retrieval_backend": settings.retrieval_backend,
            "embedding_provider": getattr(
                embedding_provider,
                "provider_name",
                None,
            ),
            "embedding_model": getattr(
                embedding_provider,
                "model_name",
                None,
            ),
            "embedding_dimensions": getattr(
                embedding_provider,
                "dimensions",
                None,
            ),
            "answer_provider": answer_generator.provider_name,
            "answer_model": answer_generator.model_name,
            "top_k_values": sorted(set(top_k_values)),
            "duration_seconds": round(duration_seconds, 3),
        },
        "report": report_payload,
    }


def _print_optional_rate(
    label: str,
    value: float | None,
) -> None:
    rendered = (
        "N/A"
        if value is None
        else f"{value:.2%}"
    )
    print(f"{label}: {rendered}")


def _print_report(report: RagEvalReport) -> None:
    print("RAG evaluation")
    print(f"Total: {report.total_cases}")
    print(f"Passed: {report.passed_cases}")
    print(f"Failed: {report.failed_cases}")
    print(
        "Schema validity: "
        f"{report.schema_validity_rate:.2%}"
    )
    print(
        "Retrieval hit rate: "
        f"{report.retrieval_hit_rate:.2%}"
    )
    print(f"Hit rate at 3: {report.hit_rate_at_3:.2%}")
    print(f"Hit rate at 5: {report.hit_rate_at_5:.2%}")
    print(
        "MRR at 5: "
        f"{report.mean_reciprocal_rank_at_5:.3f}"
    )
    print(
        "Mean source recall at 5: "
        f"{report.mean_source_recall_at_5:.2%}"
    )
    print(
        "Citation validity: "
        f"{report.citation_validity_rate:.2%}"
    )
    print(
        "Citation hit rate: "
        f"{report.citation_hit_rate:.2%}"
    )
    print(
        "Refusal accuracy: "
        f"{report.refusal_accuracy:.2%}"
    )
    _print_optional_rate(
        "Refusal precision",
        report.refusal_precision,
    )
    _print_optional_rate(
        "Refusal recall",
        report.refusal_recall,
    )
    _print_optional_rate(
        "Answer-term accuracy",
        report.answer_term_accuracy,
    )
    print(f"Pass rate: {report.pass_rate:.2%}")

    for result in report.results:
        prefix = "PASS" if result.passed else "FAIL"
        print(f"{prefix} {result.case_id}")

        for reason in result.failure_reasons:
            print(f"  - {reason}")


def main(
    argv: list[str] | None = None,
) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RAG retrieval, citations, "
            "schema validity, and refusals."
        )
    )
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("fixture_path", type=Path)
    parser.add_argument(
        "--mode",
        choices=("offline", "semantic"),
        default="offline",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--json",
        action="store_true",
    )
    output_group.add_argument(
        "--output",
        type=Path,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when any evaluation case fails.",
    )

    args = parser.parse_args(argv)
    cases = load_rag_cases(args.fixture_path)
    chunks = load_chunk_manifest(args.manifest_path)
    started = time.monotonic()
    embedding_provider: EmbeddingProvider | None = None
    engine = None

    if args.mode == "offline":
        settings = load_api_settings().model_copy(
            update={
                "retrieval_backend": "manifest",
                "manifest_path": args.manifest_path,
                "ai_provider": "deterministic",
            }
        )
        generator = create_grounded_answer_generator(
            settings
        )

        def execute_query(request: QueryRequest):
            return query_service(
                settings,
                request,
                grounded_answer_generator=generator,
            )

        report = run_rag_evals(cases, execute_query)
    else:
        settings = load_api_settings()

        if (
            settings.retrieval_backend != "pgvector"
            or settings.ai_provider != "openai"
            or settings.database_url is None
        ):
            print(
                "Semantic mode requires OpenAI, pgvector, and "
                "ANOMALYOPS_DATABASE_URL in .env.",
                file=sys.stderr,
            )
            return 2

        try:
            embedding_provider = create_embedding_provider(
                settings
            )
            generator = create_grounded_answer_generator(
                settings
            )
            engine = create_engine_from_url(
                settings.database_url
            )
            session_factory = create_session_factory(
                engine
            )
            validate_semantic_index(
                session_factory,
                chunks,
                embedding_provider,
            )

            def execute_query(request: QueryRequest):
                return query_service(
                    settings,
                    request,
                    session_factory=session_factory,
                    embedding_provider=embedding_provider,
                    grounded_answer_generator=generator,
                )

            report = run_rag_evals(
                cases,
                execute_query,
            )
        except Exception as exc:
            print(
                "Semantic evaluation failed with "
                f"{type(exc).__name__}. Check .env and run "
                "`python scripts/reindex_embeddings.py "
                f"{args.manifest_path}`.",
                file=sys.stderr,
            )
            return 2
        finally:
            if engine is not None:
                engine.dispose()

    duration_seconds = time.monotonic() - started

    if args.json:
        print(report.model_dump_json(indent=2))
    elif args.output is not None:
        snapshot = _build_snapshot(
            manifest_path=args.manifest_path,
            chunks=chunks,
            cases_count=len(cases),
            top_k_values=[case.top_k for case in cases],
            settings=settings,
            answer_generator=generator,
            report=report,
            duration_seconds=duration_seconds,
            embedding_provider=embedding_provider,
        )
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.output.write_text(
            json.dumps(snapshot, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved RAG evaluation to {args.output}.")
    else:
        _print_report(report)

    if args.strict and report.failed_cases > 0:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
