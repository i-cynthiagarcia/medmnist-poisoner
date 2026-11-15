"""Validate a completed baseline sweep and print or save its Markdown summary."""

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


DATASETS = ("organamnist", "organcmnist", "organsmnist")
DATASET_NAMES = {
    "organamnist": "OrganAMNIST",
    "organcmnist": "OrganCMNIST",
    "organsmnist": "OrganSMNIST",
}
CONFIGURATIONS = (
    (3, 1, 0.001),
    (3, 1, 0.01),
    (3, 1, 0.1),
    (3, 3, 0.001),
    (5, 1, 0.001),
    (5, 1, 0.1),
    (5, 3, 0.1),
)
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


@dataclass(frozen=True)
class Aggregate:
    dataset: str
    rounds: int
    local_epochs: int
    learning_rate: float
    mean_accuracy: float
    accuracy_stddev: float
    mean_loss: float


@dataclass(frozen=True)
class SweepSummary:
    generation_id: str
    protocol_id: str
    execution_id: str
    seeds: tuple[int, ...]
    federated: tuple[Aggregate, ...]
    centralized: tuple[Aggregate, ...] | None
    centralized_records: int
    centralized_issues: tuple[str, ...]
    partition_count: int


def _validate_version_id(value: str, prefix: str) -> str:
    if not re.fullmatch(rf"{prefix}-v\d+", value):
        raise ValueError(f"{prefix}-id must use the form '{prefix}-vN'.")
    return value


def resolve_execution_id(
    results_root: Path,
    generation_id: str,
    protocol_id: str,
    configured_id: str,
) -> str:
    """Resolve an explicit run ID or select the latest numbered execution."""
    if configured_id != "auto":
        if not re.fullmatch(r"run-\d+", configured_id):
            raise ValueError("execution-id must be 'auto' or use the form 'run-N'.")
        return configured_id

    protocol_dir = results_root / generation_id / protocol_id
    if not protocol_dir.is_dir():
        raise FileNotFoundError(
            f"No sweep executions found under {protocol_dir}."
        )
    run_numbers = [
        int(match.group(1))
        for path in protocol_dir.iterdir()
        if (match := re.fullmatch(r"run-(\d+)", path.name))
        and path.is_dir()
    ]
    if not run_numbers:
        raise FileNotFoundError(
            f"No run-N execution directories found under {protocol_dir}."
        )
    return f"run-{max(run_numbers)}"


def _load_records(results_dir: Path) -> list[dict]:
    result_paths = sorted(results_dir.rglob("seed-*.json"))
    if not result_paths:
        raise FileNotFoundError(
            f"No run JSON records found under {results_dir}."
        )

    records = []
    for path in result_paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON run record: {path}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Run record must contain a JSON object: {path}")
        records.append(record)
    return records


def _record_configuration(record: dict, mode: str) -> tuple[int, int, float]:
    configuration = record.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError(f"Missing configuration in {record.get('experiment')}.")

    if mode == "federated":
        fields = (
            "federated_rounds",
            "local_epochs",
            "learning_rate",
        )
    else:
        fields = (
            "federated_round_equivalents",
            "local_epochs_per_round_equivalent",
            "learning_rate",
        )
    try:
        return (
            int(configuration[fields[0]]),
            int(configuration[fields[1]]),
            float(configuration[fields[2]]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Incomplete {mode} configuration in {record.get('experiment')}."
        ) from error


def _validate_metrics(record: dict, rounds: int) -> tuple[float, float]:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing metrics in {record.get('experiment')}.")
    validation = metrics.get("validation_by_round")
    expected_keys = {str(round_number) for round_number in range(rounds + 1)}
    if not isinstance(validation, dict) or set(validation) != expected_keys:
        raise ValueError(
            f"Incomplete per-round validation metrics for "
            f"{record.get('dataset')}/{record.get('experiment')}/"
            f"seed-{record.get('seed')}."
        )

    final_metrics = validation[str(rounds)]
    try:
        accuracy = float(final_metrics["accuracy"])
        loss = float(final_metrics["loss"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Missing final accuracy or loss for "
            f"{record.get('dataset')}/{record.get('experiment')}/"
            f"seed-{record.get('seed')}."
        ) from error
    if not math.isfinite(accuracy) or not math.isfinite(loss):
        raise ValueError(
            f"Non-finite final validation metrics for "
            f"{record.get('dataset')}/{record.get('experiment')}/"
            f"seed-{record.get('seed')}."
        )
    if not 0 <= accuracy <= 1 or loss < 0:
        raise ValueError(
            f"Invalid final validation metrics for "
            f"{record.get('dataset')}/{record.get('experiment')}/"
            f"seed-{record.get('seed')}."
        )
    if record.get("loss_aggregation") != "per_image":
        raise ValueError(
            f"Expected per-image loss aggregation in "
            f"{record.get('dataset')}/{record.get('experiment')}/"
            f"seed-{record.get('seed')}."
        )
    if metrics.get("test_final") is not None:
        raise ValueError(
            "The baseline sweep must not use test-set metrics for selection: "
            f"{record.get('dataset')}/{record.get('experiment')}/"
            f"seed-{record.get('seed')}."
        )
    return accuracy, loss


def _validate_partition(
    record: dict, project_root: Path, seen_paths: set[Path]
) -> None:
    manifest_ref = record.get("partition_manifest")
    if not isinstance(manifest_ref, str) or not manifest_ref:
        raise ValueError(
            f"Missing partition manifest for "
            f"{record.get('dataset')}/{record.get('experiment')}/"
            f"seed-{record.get('seed')}."
        )
    manifest_path = Path(manifest_ref)
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    manifest_path = manifest_path.resolve()
    if manifest_path in seen_paths:
        raise ValueError(f"Partition manifest is reused: {manifest_path}")
    seen_paths.add(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing partition manifest: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid partition manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Partition manifest must be a JSON object: {manifest_path}")
    if (
        manifest.get("generation_id") != record.get("generation_id")
        or manifest.get("protocol_id") != record.get("protocol_id")
        or manifest.get("execution_id") != record.get("execution_id")
        or manifest.get("dataset") != record.get("dataset")
        or manifest.get("seed") != record.get("seed")
        or manifest.get("num_clients") != 20
    ):
        raise ValueError(f"Partition manifest does not match run: {manifest_path}")
    verification = manifest.get("verification", {})
    if any(
        verification.get(key) is not True
        for key in (
            "partitions_are_disjoint",
            "all_training_examples_assigned",
            "each_training_example_assigned_once",
        )
    ):
        raise ValueError(f"Partition verification failed: {manifest_path}")


def _aggregate(
    dataset: str,
    configuration: tuple[int, int, float],
    values: list[tuple[float, float]],
) -> Aggregate:
    accuracies = [accuracy for accuracy, _ in values]
    losses = [loss for _, loss in values]
    return Aggregate(
        dataset=dataset,
        rounds=configuration[0],
        local_epochs=configuration[1],
        learning_rate=configuration[2],
        mean_accuracy=statistics.fmean(accuracies),
        accuracy_stddev=statistics.pstdev(accuracies),
        mean_loss=statistics.fmean(losses),
    )


def summarize_records(
    records: list[dict],
    project_root: Path,
    generation_id: str,
    protocol_id: str,
    execution_id: str,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> SweepSummary:
    """Validate a full baseline matrix and aggregate its final-round metrics."""
    _validate_version_id(generation_id, "generation")
    _validate_version_id(protocol_id, "protocol")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Expected seeds must be a non-empty, unique list.")
    if any(seed < 0 for seed in seeds):
        raise ValueError("Expected seeds must be non-negative integers.")

    expected_keys = {
        (dataset, configuration, seed)
        for dataset in DATASETS
        for configuration in CONFIGURATIONS
        for seed in seeds
    }
    found: dict[str, dict[tuple[str, tuple[int, int, float], int], tuple[float, float]]] = {
        "federated": {},
        "centralized": {},
    }
    seen: dict[str, set[tuple[str, tuple[int, int, float], int]]] = {
        "federated": set(),
        "centralized": set(),
    }
    centralized_issues = []
    partition_paths: set[Path] = set()

    for record in records:
        mode = record.get("training_mode")
        if mode not in found:
            raise ValueError(f"Unexpected training mode: {mode!r}")
        if (
            record.get("generation_id") != generation_id
            or record.get("protocol_id") != protocol_id
            or record.get("execution_id") != execution_id
        ):
            raise ValueError(
                f"Run identifiers do not match {generation_id}/"
                f"{protocol_id}/{execution_id}."
            )
        dataset = record.get("dataset")
        if dataset not in DATASETS:
            raise ValueError(f"Unexpected baseline dataset: {dataset!r}")
        try:
            seed = int(record["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Run record is missing a valid seed.") from error
        configuration = _record_configuration(record, mode)
        if configuration not in CONFIGURATIONS:
            raise ValueError(
                f"Unexpected sweep configuration: "
                f"{dataset}/{configuration}."
            )
        key = (dataset, configuration, seed)
        if key not in expected_keys:
            raise ValueError(
                f"Unexpected dataset/configuration/seed combination: {key}."
            )
        if key in seen[mode]:
            raise ValueError(f"Duplicate {mode} run record: {key}.")
        seen[mode].add(key)
        if mode == "federated" and record.get("status") != "completed":
            raise ValueError(
                f"Run did not complete: {dataset}/{record.get('experiment')}/"
                f"seed-{seed} (status={record.get('status')!r})."
            )

        if mode == "federated":
            training_results = record.get("training_results")
            if (
                not isinstance(training_results, dict)
                or training_results.get("valid_run") is not True
            ):
                raise ValueError(f"Federated run failed validity checks: {key}.")
            rounds = configuration[0]
            expected_rounds = {str(number) for number in range(1, rounds + 1)}
            received = training_results.get("received_results_by_round", {})
            failures = training_results.get("client_failures_by_round", {})
            if not isinstance(received, dict) or set(received) != expected_rounds or any(
                received[round_number] != 20 for round_number in expected_rounds
            ):
                raise ValueError(
                    f"Expected 20 client results in every round: {key}."
                )
            if not isinstance(failures, dict) or set(failures) != expected_rounds or any(
                failures[round_number] != 0 for round_number in expected_rounds
            ):
                raise ValueError(f"Client failures found in federated run: {key}.")
            if record.get("configuration", {}).get("clients") != 20:
                raise ValueError(f"Expected 20 clients in federated run: {key}.")
            _validate_partition(record, project_root, partition_paths)
        else:
            centralized_config = record.get("configuration", {})
            try:
                if record.get("status") != "completed":
                    raise ValueError(
                        f"status={record.get('status')!r}, expected 'completed'"
                    )
                if not isinstance(centralized_config, dict):
                    raise ValueError("missing centralized configuration")
                if (
                    centralized_config.get("epochs")
                    != configuration[0] * configuration[1]
                    or centralized_config.get("training_data")
                    != "full_train_split"
                ):
                    raise ValueError(
                        "centralized control does not match the federated "
                        "epoch budget"
                    )
                metric_values = _validate_metrics(record, configuration[0])
            except ValueError as error:
                centralized_issues.append(
                    f"{dataset}/{record.get('experiment')}/seed-{seed}: {error}"
                )
                continue
            found[mode][key] = metric_values
            continue

        found[mode][key] = _validate_metrics(record, configuration[0])

    missing_federated = expected_keys - found["federated"].keys()
    if missing_federated:
        raise ValueError(
            "Federated sweep is incomplete; missing records: "
            + ", ".join(map(str, sorted(missing_federated)[:10]))
        )

    missing_centralized = expected_keys - seen["centralized"]
    if missing_centralized:
        centralized_issues.append(
            "missing centralized records: "
            + ", ".join(map(str, sorted(missing_centralized)[:10]))
        )

    def make_aggregates(
        mode: str,
    ) -> tuple[Aggregate, ...]:
        values = found[mode]
        return tuple(
            _aggregate(
                dataset,
                configuration,
                [
                    values[(dataset, configuration, seed)]
                    for seed in seeds
                ],
            )
            for dataset in DATASETS
            for configuration in CONFIGURATIONS
        )

    return SweepSummary(
        generation_id=generation_id,
        protocol_id=protocol_id,
        execution_id=execution_id,
        seeds=tuple(sorted(seeds)),
        federated=make_aggregates("federated"),
        centralized=(
            make_aggregates("centralized")
            if seen["centralized"] == expected_keys
            and not centralized_issues
            else None
        ),
        centralized_records=len(seen["centralized"]),
        centralized_issues=tuple(centralized_issues),
        partition_count=len(partition_paths),
    )


def _format_learning_rate(value: float) -> str:
    return f"{value:g}"


def _render_table(
    aggregates: tuple[Aggregate, ...],
    centered: bool = False,
) -> list[str]:
    if centered:
        lines = [
            "| Dataset | Round equivalents | Epochs per round equivalent | Learning rate | Final validation accuracy | Mean final loss |",
            "|---------|------------------:|---------------------------:|--------------:|---------------------------:|----------------:|",
        ]
    else:
        lines = [
            "| Dataset | Rounds | Local epochs | Learning rate | Final validation accuracy | Mean final loss |",
            "|---------|-------:|-------------:|--------------:|---------------------------:|----------------:|",
        ]

    best_by_dataset = {}
    if not centered:
        for dataset in DATASETS:
            best_by_dataset[dataset] = max(
                (item for item in aggregates if item.dataset == dataset),
                key=lambda item: item.mean_accuracy,
            )

    for item in aggregates:
        accuracy = (
            f"{item.mean_accuracy * 100:.2f}% ± "
            f"{item.accuracy_stddev * 100:.2f} pp"
        )
        cells = [
            DATASET_NAMES[item.dataset],
            str(item.rounds),
            str(item.local_epochs),
            _format_learning_rate(item.learning_rate),
            accuracy,
            f"{item.mean_loss:.3f}",
        ]
        if not centered and item == best_by_dataset[item.dataset]:
            cells = [f"**{cell}**" for cell in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_summary(
    summary: SweepSummary,
    results_root: str = "results/baseline",
    models_root: str = "models/baseline",
) -> str:
    """Render a Markdown report for a validated baseline sweep."""
    seed_range = ", ".join(map(str, summary.seeds))
    federated_count = len(summary.federated) * len(summary.seeds)
    centralized_count = summary.centralized_records
    lines = [
        f"## Current Reference: Complete Clean Baseline "
        f"({summary.generation_id} / {summary.protocol_id})",
        "",
        f"**Status:** The `{summary.execution_id}` execution contains "
        f"{federated_count} valid federated runs"
        + f", with {len(summary.seeds)} seeds ({seed_range}) for every "
        "tested configuration and dataset. No failed or incomplete federated "
        "runs need to be redone.",
        "",
        "**Use:** Use these federated results as the current clean reference "
        "for poisoning comparisons. Centralized full-train runs, when present, "
        "are reported separately as comparators, not as federated results.",
        "",
        "### Protocol and run integrity",
        "",
        "- Datasets: OrganAMNIST, OrganCMNIST, and OrganSMNIST.",
        f"- Seeds: {seed_range}.",
        "- Configurations: the seven combinations shown in the tables below, "
        "for each dataset.",
        "- Federated training: 20 clients with FedAvg; every client returned "
        "a result in every round. All runs completed, passed the recorded "
        "validity check, and had zero client failures.",
        (
            "- Centralized comparisons: all records completed with finite "
            "per-round validation metrics."
            if summary.centralized is not None
            else (
                "- Centralized comparisons: none were included in this execution."
                if centralized_count == 0
                else (
                    (
                        f"- Centralized comparisons: found {centralized_count} "
                        f"records; {len(summary.centralized_issues)} contained "
                        "invalid metrics, so no aggregate table was generated."
                    )
                    if centralized_count == federated_count
                    else (
                        f"- Centralized comparisons: found {centralized_count} "
                        f"of {federated_count} expected records; no aggregate "
                        "table was generated."
                    )
                )
            )
        ),
        f"- Partition manifests: all {summary.partition_count} verify disjoint "
        "partitions and assignment of every training example exactly once.",
        "- Metrics: validation accuracy and per-image cross-entropy loss are "
        "recorded at initialization and each training round. Test evaluation "
        "was disabled; `test_final` is null throughout.",
        f"- Run-specific JSON records and model checkpoints are in "
        f"`{results_root}/{summary.generation_id}/{summary.protocol_id}/"
        f"{summary.execution_id}/` and "
        f"`{models_root}/{summary.generation_id}/{summary.protocol_id}/"
        f"{summary.execution_id}/`.",
        "",
        "### Federated sweep results",
        "",
        "Accuracy is the mean final-round validation accuracy across "
        f"{len(summary.seeds)} seeds; “±” is the population standard deviation "
        "in percentage points. Mean final loss is the per-image-weighted "
        "validation cross-entropy. The best observed configuration by mean "
        "validation accuracy for each dataset is bolded.",
        "",
        *_render_table(summary.federated),
    ]
    if summary.centralized is not None:
        lines.extend(
            [
                "",
                "### Centralized full-train comparison results",
                "",
                "These runs train centrally on the full `train` split for the "
                "equivalent number of epochs. Accuracy is mean final-round-"
                "equivalent validation accuracy ± population standard "
                f"deviation across {len(summary.seeds)} seeds; loss is the "
                "mean per-image validation cross-entropy.",
                "",
                *_render_table(summary.centralized, centered=True),
            ]
        )
    elif centralized_count:
        lines.extend(
            [
                "",
                "### Centralized comparison data requiring attention",
                "",
                "The centralized controls are not summarized because their "
                "records are incomplete or contain invalid metrics. Resolve "
                "these issues before using centralized comparisons:",
                "",
                *[f"- {issue}" for issue in summary.centralized_issues],
            ]
        )

    lines.extend(
        [
            "",
            "### Candidate settings for poisoning comparisons",
            "",
            "| Dataset | Rounds | Local epochs | Learning rate | Seeds |",
            "|---------|-------:|-------------:|--------------:|-------|",
        ]
    )
    for dataset in DATASETS:
        best = max(
            (item for item in summary.federated if item.dataset == dataset),
            key=lambda item: item.mean_accuracy,
        )
        lines.append(
            f"| {DATASET_NAMES[dataset]} | {best.rounds} | "
            f"{best.local_epochs} | {_format_learning_rate(best.learning_rate)} "
            f"| {seed_range} |"
        )
    lines.extend(
        [
            "",
            "These are the best mean validation-accuracy settings among the "
            "seven tested configurations per dataset, not estimates of "
            "globally optimal settings. Keep each dataset's configuration "
            "fixed across its clean and poisoned runs; do not retune settings "
            "per attack condition. The shared validation set informed this "
            "selection, so reserve the test split for final evaluation after "
            "the protocol is locked.",
        ]
    )
    return "\n".join(lines)


def _project_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def write_report(report: str, output_path: Path) -> None:
    """Write a Markdown report, creating its parent directory when needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.rstrip() + "\n", encoding="utf-8")


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the latest full baseline sweep, summarize its final "
            "validation metrics, and print or save a Markdown report."
        )
    )
    parser.add_argument(
        "--generation-id",
        default="generation-v3",
        help="Generation directory (default: generation-v3).",
    )
    parser.add_argument(
        "--protocol-id",
        default="protocol-v1",
        help="Protocol directory (default: protocol-v1).",
    )
    parser.add_argument(
        "--execution-id",
        default="auto",
        help="Execution run-N, or auto for the highest run number (default: auto).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/baseline"),
        help="Results root (default: results/baseline).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models/baseline"),
        help="Models root shown in the generated report (default: models/baseline).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Expected seed set (default: 42 43 44 45 46).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Write the Markdown report to this path (relative paths are "
            "relative to quickstart-pytorch). By default, print it to stdout."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    results_root = _project_path(args.results_dir, project_root)
    models_root = _project_path(args.models_dir, project_root)
    try:
        generation_id = _validate_version_id(args.generation_id, "generation")
        protocol_id = _validate_version_id(args.protocol_id, "protocol")
        execution_id = resolve_execution_id(
            results_root,
            generation_id,
            protocol_id,
            args.execution_id,
        )
        results_dir = (
            results_root / generation_id / protocol_id / execution_id
        )
        records = _load_records(results_dir)
        summary = summarize_records(
            records,
            project_root,
            generation_id,
            protocol_id,
            execution_id,
            tuple(args.seeds),
        )
        generated_section = render_summary(
            summary,
            _display_path(results_root, project_root),
            _display_path(models_root, project_root),
        )
        if args.output is None:
            print(generated_section)
            return

        output_path = _project_path(args.output, project_root)
        write_report(generated_section, output_path)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Saved Markdown summary to {_display_path(output_path, project_root)}")


if __name__ == "__main__":
    main()
