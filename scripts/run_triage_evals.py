import argparse
from pathlib import Path

from dotenv import load_dotenv

from copilot.api.settings import (
    load_api_settings,
)
from copilot.clients.anomaly_api import (
    AnomalyApiClient,
)
from copilot.evals.runner import (
    load_triage_cases,
    run_triage_evals,
)
from copilot.providers.factory import (
    create_triage_agent,
)
from copilot.tools.anomaly import (
    AnomalyOperationalTools,
)


def format_optional_rate(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"

    return f"{value:.2%}"


def main(
    argv: list[str] | None = None,
) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate triage schema validity, "
            "grounding, consistency, and "
            "expected findings."
        )
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

    settings = load_api_settings()

    if (
        settings.anomaly_api_base_url is None
        or not settings.anomaly_api_base_url.strip()
    ):
        raise RuntimeError(
            "ANOMALYOPS_ANOMALY_API_BASE_URL "
            "is required for triage evaluation."
        )

    cases = load_triage_cases(
        args.fixture_path
    )

    with AnomalyApiClient(
        settings.anomaly_api_base_url
    ) as anomaly_client:
        tools = AnomalyOperationalTools(
            anomaly_client
        )
        agent = create_triage_agent(
            settings,
            tools,
        )

        report = run_triage_evals(
            cases,
            agent.triage,
        )

    if args.json:
        print(
            report.model_dump_json(
                indent=2
            )
        )
    else:
        print("Triage evaluation")
        print(
            f"Provider: {agent.provider_name}"
        )
        print(
            f"Model: {agent.model_name}"
        )
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
            "Evidence validity: "
            f"{report.evidence_validity_rate:.2%}"
        )
        print(
            "Run consistency: "
            f"{report.run_consistency_rate:.2%}"
        )
        print(
            "Max-events compliance: "
            f"{report.max_events_compliance_rate:.2%}"
        )
        print(
            "Status semantics: "
            f"{report.status_semantics_rate:.2%}"
        )
        print(
            "Status accuracy: "
            f"{format_optional_rate(report.status_accuracy)}"
        )
        print(
            "Expected findings accuracy: "
            f"{format_optional_rate(report.expected_findings_accuracy)}"
        )
        print(
            "Finding count accuracy: "
            f"{format_optional_rate(report.finding_count_accuracy)}"
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