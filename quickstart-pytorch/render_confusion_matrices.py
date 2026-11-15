"""Render and index confusion matrices stored in run-result JSON files."""

import argparse
import json
from pathlib import Path

from pytorchexample.artifacts import run_artifact_stem
from pytorchexample.visualization import save_confusion_matrix_plot


def plan_run_plots(
    result_path: Path, record: dict
) -> tuple[dict, list[tuple[Path, list[list[int]], str]]]:
    """Build per-run plot paths from saved validation and test matrices."""
    metrics = record["metrics"]
    plot_dir = result_path.parent / "plots" / run_artifact_stem(
        record["seed"], record["run_id"]
    )
    plot_refs: dict[str, dict[str, str] | str | None] = {
        "validation_by_round": {},
        "validation_final_recomputed": None,
        "test_final": None,
    }
    jobs = []
    round_metrics = metrics.get("validation_by_round", {})
    has_round_matrices = bool(round_metrics) and all(
        "confusion_matrix" in round_record
        for round_record in round_metrics.values()
    )

    if has_round_matrices:
        round_plot_refs = plot_refs["validation_by_round"]
        for round_number, round_record in sorted(
            round_metrics.items(), key=lambda item: int(item[0])
        ):
            plot_path = plot_dir / f"validation-round-{int(round_number):03d}.png"
            round_plot_refs[round_number] = plot_path.as_posix()
            jobs.append(
                (
                    plot_path,
                    round_record["confusion_matrix"],
                    f"Validation confusion matrix - round {round_number}",
                )
            )
    elif "validation_final_recomputed" in metrics:
        final_record = metrics["validation_final_recomputed"]
        plot_path = plot_dir / "validation-final-recomputed.png"
        plot_refs["validation_final_recomputed"] = plot_path.as_posix()
        jobs.append(
            (
                plot_path,
                final_record["confusion_matrix"],
                "Validation confusion matrix - final saved checkpoint",
            )
        )
    else:
        raise ValueError(
            f"No saved validation confusion matrix in {result_path}."
        )

    test_metrics = metrics.get("test_final")
    if test_metrics is not None and "confusion_matrix" in test_metrics:
        plot_path = plot_dir / "test-final.png"
        plot_refs["test_final"] = plot_path.as_posix()
        jobs.append(
            (
                plot_path,
                test_metrics["confusion_matrix"],
                "Test confusion matrix - final model",
            )
        )

    return plot_refs, jobs


def parse_args() -> argparse.Namespace:
    """Parse input result directory."""
    parser = argparse.ArgumentParser(
        description="Render stored confusion matrices as per-run PNG plots."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/baseline/validation-v2"),
        help=(
            "Run-result directory "
            "(default: results/baseline/validation-v2)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Render plots for saved matrices without replacing existing artifacts."""
    args = parse_args()
    result_paths = sorted(args.results_dir.rglob("seed-*.json"))
    if not result_paths:
        raise FileNotFoundError(
            f"No run JSON records found under {args.results_dir}."
        )

    plans = []
    for result_path in result_paths:
        record = json.loads(result_path.read_text())
        plot_refs, jobs = plan_run_plots(result_path, record)
        existing_refs = record["metrics"].get("confusion_matrix_plots")
        if existing_refs is not None:
            if existing_refs != plot_refs or not all(
                path.is_file() for path, _, _ in jobs
            ):
                raise ValueError(
                    f"Existing plot references are incomplete or inconsistent: "
                    f"{result_path}"
                )
            continue
        existing_plots = [path for path, _, _ in jobs if path.exists()]
        if existing_plots:
            raise FileExistsError(
                "Refusing to replace existing plots: "
                + ", ".join(str(path) for path in existing_plots)
            )
        record["metrics"]["confusion_matrix_plots"] = plot_refs
        plans.append(
            (result_path, record, jobs, record["metrics"]["class_names"])
        )

    planned_paths = [
        path
        for _, _, jobs, _ in plans
        for path, _, _ in jobs
    ]
    if len(planned_paths) != len(set(planned_paths)):
        raise FileExistsError("Multiple run records resolve to the same plot path.")

    for _, _, jobs, class_names in plans:
        for plot_path, matrix, title in jobs:
            save_confusion_matrix_plot(matrix, class_names, plot_path, title)

    for result_path, record, _, _ in plans:
        result_path.write_text(json.dumps(record, indent=2) + "\n")

    rendered = sum(len(jobs) for _, _, jobs, _ in plans)
    print(
        f"Rendered {rendered} confusion-matrix plots for "
        f"{len(plans)} runs; existing plots were not overwritten."
    )


if __name__ == "__main__":
    main()
