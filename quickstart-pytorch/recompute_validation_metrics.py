"""Recompute final validation metrics from saved global model checkpoints."""

import argparse
import json
import math
from pathlib import Path

import torch
from flwr.app import MetricRecord

from pytorchexample.shared.metrics import (
    serialize_evaluation_metrics,
)
from pytorchexample.shared.artifacts import run_artifact_stem
from pytorchexample.shared.data import (
    get_class_names,
    load_centralized_dataset,
)
from pytorchexample.shared.model import Net
from pytorchexample.shared.training import evaluate


def metrics_match(expected, actual, tolerance: float = 1e-9) -> bool:
    """Compare nested JSON metric values, allowing tiny float differences."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            metrics_match(expected[key], actual[key], tolerance)
            for key in expected
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            metrics_match(expected_value, actual_value, tolerance)
            for expected_value, actual_value in zip(expected, actual)
        )
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        return math.isclose(
            expected, actual, rel_tol=tolerance, abs_tol=tolerance
        )
    return expected == actual


def recompute_final_metrics(
    results_dir: Path, models_dir: Path
) -> list[tuple[Path, dict, dict, dict | None]]:
    """Evaluate every saved final model and return updated run records."""
    result_paths = sorted(results_dir.rglob("seed-*.json"))
    if not result_paths:
        raise FileNotFoundError(f"No run JSON records found under {results_dir}.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaders = {}
    updates = []

    for result_path in result_paths:
        record = json.loads(result_path.read_text())
        stored_metrics = record["metrics"].get("validation_final_recomputed")
        dataset = record["dataset"]
        experiment = record["experiment"]
        seed = record["seed"]
        generation_id = record.get("generation_id", "generation-v2")
        model_dir = models_dir / generation_id
        protocol_id = record.get("protocol_id")
        if protocol_id is not None:
            model_dir /= protocol_id
        model_dir = model_dir / dataset / experiment
        run_id = record.get("run_id")
        model_path = (
            model_dir / f"{run_artifact_stem(seed, run_id)}.pt"
            if run_id is not None
            else model_dir / f"seed-{seed}.pt"
        )
        if not model_path.is_file():
            legacy_model_path = model_dir / f"seed-{seed}.pt"
            if legacy_model_path.is_file():
                model_path = legacy_model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing final checkpoint: {model_path}")

        if dataset not in loaders:
            loaders[dataset] = load_centralized_dataset("val", dataset)

        model = Net()
        state_dict = torch.load(
            model_path, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)
        metrics = evaluate(model, loaders[dataset], device, dataset)
        class_names = get_class_names(dataset)
        final_metrics = serialize_evaluation_metrics(
            MetricRecord(metrics), class_names
        )
        final_metrics.update(
            {
                "evaluation_split": "val",
                "evaluation_device": str(device),
                "loss_aggregation": "per_image",
                "source_checkpoint": model_path.as_posix(),
            }
        )

        updates.append((result_path, record, final_metrics, stored_metrics))

    return updates


def parse_args() -> argparse.Namespace:
    """Parse recomputation and output options."""
    parser = argparse.ArgumentParser(
        description=(
            "Recompute per-image final validation metrics using saved global "
            "model checkpoints."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/baseline/generation-v2"),
        help=(
            "Input run-record directory "
            "(default: results/baseline/generation-v2)."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models/baseline"),
        help="Final model directory (default: models/baseline).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--verify",
        action="store_true",
        help="Recompute and compare against existing final metrics without writing.",
    )
    mode.add_argument(
        "--in-place",
        action="store_true",
        help="Add or replace recomputed metrics in the input run records.",
    )
    mode.add_argument(
        "--output-dir",
        type=Path,
        help="Write updated copies of all run records to a new directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Recompute, verify, or write final validation metrics."""
    args = parse_args()
    if args.output_dir is not None and args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}"
        )

    updates = recompute_final_metrics(args.results_dir, args.models_dir)

    if args.verify:
        for result_path, _, final_metrics, stored_metrics in updates:
            if stored_metrics is None or not metrics_match(
                stored_metrics, final_metrics
            ):
                raise ValueError(
                    f"Stored final metrics do not match recomputed values: "
                    f"{result_path}"
                )
        print(f"Verified recomputed final metrics for {len(updates)} runs.")
        return

    if args.in_place:
        output_paths = [
            (result_path, record, final_metrics)
            for result_path, record, final_metrics, _ in updates
        ]
    else:
        if args.output_dir is None:
            raise ValueError("Select --verify, --in-place, or --output-dir.")
        args.output_dir.mkdir(parents=True)
        output_paths = [
            (
                args.output_dir / result_path.relative_to(args.results_dir),
                record,
                final_metrics,
            )
            for result_path, record, final_metrics, _ in updates
        ]

    for output_path, record, final_metrics in output_paths:
        dataset = record["dataset"]
        record["metrics"]["class_names"] = get_class_names(dataset)
        record["metrics"]["confusion_matrix_axes"] = {
            "rows": "true_class",
            "columns": "predicted_class",
        }
        record["metrics"]["validation_final_recomputed"] = final_metrics
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Wrote recomputed final metrics for {len(output_paths)} runs.")


if __name__ == "__main__":
    main()
