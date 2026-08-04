import argparse
from pathlib import Path

from dotenv import load_dotenv

from copilot.api.query_service import (
    query_service,
)
from copilot.api.settings import (
    load_api_settings,
)
from copilot.evals.runner import (
    load_rag_cases,
    run_rag_evals,
)
from copilot.providers.factory import (
    create_grounded_answer_generator,
)
from copilot.schemas.query import (
    QueryRequest,
)


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
    parser.add_argument(
        "manifest_path",
        type=Path,
    )
    parser.add_argument(
        "fixture_path",
        type=Path,
    )
    parser.add_argument(
        "--json",
        action="store_true",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit with status 1 when any "
            "evaluation case fails."
        ),
    )

    args = parser.parse_args(argv)

    settings = load_api_settings().model_copy(
        update={
            "retrieval_backend": "manifest",
            "manifest_path": args.manifest_path,
        }
    )

    generator = (
        create_grounded_answer_generator(
            settings
        )
    )
    cases = load_rag_cases(
        args.fixture_path
    )

    def execute_query(
        request: QueryRequest,
    ):
        return query_service(
            settings,
            request,
            grounded_answer_generator=generator,
        )

    report = run_rag_evals(
        cases,
        execute_query,
    )

    if args.json:
        print(
            report.model_dump_json(
                indent=2
            )
        )
    else:
        print("RAG evaluation")
        print(
            f"Total: {report.total_cases}"
        )
        print(
            f"Passed: {report.passed_cases}"
        )
        print(
            f"Failed: {report.failed_cases}"
        )
        print(
            "Schema validity: "
            f"{report.schema_validity_rate:.2%}"
        )
        print(
            "Retrieval hit rate: "
            f"{report.retrieval_hit_rate:.2%}"
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
        print(
            f"Pass rate: {report.pass_rate:.2%}"
        )

        for result in report.results:
            prefix = (
                "PASS"
                if result.passed
                else "FAIL"
            )
            print(
                f"{prefix} {result.case_id}"
            )

            for reason in (
                result.failure_reasons
            ):
                print(f"  - {reason}")

    if (
        args.strict
        and report.failed_cases > 0
    ):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())