"""Run the frozen, fresh OrganCMNIST and OrganSMNIST defense comparisons."""

import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

from run_organ_cs_defense_pilot import (
    AGGREGATIONS,
    DATASETS,
    ROOT,
    archive_checksum,
    capture_environment,
    prepare_datasets,
)
from run_organa_defense_sweep import validate_record

RESULTS = ROOT / "results/defense/generation-v1/protocol-v8"
MODELS = ROOT / "models/defense/generation-v1/protocol-v8"
SEEDS = tuple(range(42, 52))
FLOORS = {
    "organcmnist": {"accuracy": 0.80, "kidney_left_recall": 0.50},
    "organsmnist": {"accuracy": 0.55, "kidney_left_recall": 0.25},
}
RUNNER = Path(__file__).resolve()


def comparison_environment(check_committed=False):
    environment = capture_environment(check_committed=check_committed)
    contents = RUNNER.read_bytes()
    if check_committed:
        committed = subprocess.check_output(
            ["git", "show", "HEAD:quickstart-pytorch/run_organ_cs_defense_comparison.py"],
            cwd=ROOT,
        )
        if contents != committed:
            raise ValueError("Commit the comparison runner before starting")
    environment["comparison_runner_sha256"] = hashlib.sha256(contents).hexdigest()
    return environment


def verify_environment(environment, checksums):
    if comparison_environment() != environment:
        raise ValueError("Source, revision or environment changed during the comparison")
    for dataset, checksum in checksums.items():
        if archive_checksum(dataset) != checksum:
            raise ValueError(f"{dataset} archive changed during the comparison")


def next_execution_id():
    numbers = []
    for root in (RESULTS, MODELS):
        for path in root.glob("run-*"):
            suffix = path.name.removeprefix("run-")
            if suffix.isdigit():
                numbers.append(int(suffix))
    return f"run-{max(numbers, default=0) + 1}"


def make_command(dataset, execution, seed, aggregation, count):
    fraction = "1.0" if count else "0.0"
    config = (
        f'experiment-type="label-flipping" aggregation="{aggregation}" '
        f'aggregation-subdirectories=true require-complete-round=true '
        f'dataset="{dataset}" seed={seed} num-clients=20 num-server-rounds=5 '
        f'local-epochs=3 learning-rate=0.01 batch-size=32 '
        f'poison-num-clients={count} poison-fraction={fraction} '
        f'poison-source-class-id=4 poison-target-class-id=5 evaluate-test=false '
        f'generation-id="generation-v1" protocol-id="protocol-v8" '
        f'execution-id="{execution}" results-dir="results/defense" '
        f'models-dir="models/defense"'
    )
    return [
        "flwr", "run", ".", "--federation-config", "num-supernodes=20",
        "--run-config", config, "--stream",
    ]


def validate_condition(dataset, execution, seed, aggregation, count, partitions):
    from pytorchexample.experiments.poisoning.label_flipping import select_poisoned_clients
    from pytorchexample.shared.data import get_class_names

    prefix = "median_" if aggregation == "median" else ""
    fraction = "1p000" if count else "0p000"
    experiment = f"{prefix}label_flip_r5_e3_lr0p010_clients{count}_f{fraction}_s4_t5"
    paths = list(
        (RESULTS / execution / aggregation / dataset / experiment).glob(f"seed-{seed}_run-*.json")
    )
    if len(paths) != 1:
        raise ValueError(f"Expected one result; found {len(paths)}")
    record = json.loads(paths[0].read_text())
    if (
        record["generation_id"] != "generation-v1"
        or record["protocol_id"] != "protocol-v8"
        or record["execution_id"] != execution
        or record["artifact_layout"] != "aggregation_subdirectories"
    ):
        raise ValueError("Incorrect execution metadata")
    clients = validate_record(
        record, seed, aggregation, count, float(count == 4),
        dataset=dataset, local_epochs=3,
    )
    partition = json.loads((ROOT / record["partition_manifest"]).read_text())
    if (partition["dataset"], partition["seed"], partition["num_clients"]) != (dataset, seed, 20):
        raise ValueError("Incorrect partition metadata")
    metrics = record["metrics"]
    names = get_class_names(dataset)
    validation = metrics["validation_by_round"]
    targeted = metrics["targeted_source_to_target_misclassification"]
    if (
        metrics["class_names"] != names
        or set(validation) != {str(i) for i in range(6)}
        or set(targeted["validation_by_round"]) != {str(i) for i in range(6)}
        or any(
            set(round_metrics["per_class"]) != set(names)
            or not math.isfinite(round_metrics["accuracy"])
            or not math.isfinite(round_metrics["loss"])
            or any(not math.isfinite(round_metrics["per_class"][name]["recall"]) for name in names)
            for round_metrics in validation.values()
        )
        or any(
            not math.isfinite(round_metrics["rate"]) or round_metrics["support"] <= 0
            for round_metrics in targeted["validation_by_round"].values()
        )
        or targeted["test_final"] is not None
    ):
        raise ValueError("Missing or invalid validation metrics")
    if seed in partitions and partitions[seed] != clients:
        raise ValueError("Partition summaries differ across matched conditions")
    attack = record["configuration"]["label_flipping"]
    selected = select_poisoned_clients(20, count, seed)
    if attack["poisoned_client_ids"] != selected:
        raise ValueError("Attacker selection differs from the seeded selection")
    if count:
        expected_flips = {
            str(client["partition_id"]): client["class_counts"]["kidney-left"]
            for client in clients if client["partition_id"] in selected
        }
        if (
            len(expected_flips) != count
            or attack["flipped_source_examples_by_client"] != expected_flips
            or attack["total_flipped_source_examples"] != sum(expected_flips.values())
        ):
            raise ValueError("Recorded full-flip counts differ from the partition")
    partitions[seed] = clients
    return record


def clean_summary(dataset, records):
    summary = {}
    for aggregation in AGGREGATIONS:
        available = [records.get((seed, aggregation, 0)) for seed in SEEDS]
        if any(record is None for record in available):
            summary[aggregation] = {"complete": False}
            continue
        accuracy = statistics.fmean(
            record["metrics"]["validation_by_round"]["5"]["accuracy"] for record in available
        )
        recall = statistics.fmean(
            record["metrics"]["validation_by_round"]["5"]["per_class"]["kidney-left"]["recall"]
            for record in available
        )
        summary[aggregation] = {
            "complete": True,
            "mean_accuracy": accuracy,
            "mean_kidney_left_recall": recall,
            "accuracy_floor_met": accuracy >= FLOORS[dataset]["accuracy"],
            "kidney_left_recall_floor_met": recall >= FLOORS[dataset]["kidney_left_recall"],
        }
    return summary


def run_dataset(dataset, environment, checksums):
    execution = next_execution_id()
    results_dir = RESULTS / execution
    models_dir = MODELS / execution
    results_dir.mkdir(parents=True, exist_ok=False)
    models_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "description": "Fresh matched validation-only defense comparison",
        "dataset": dataset,
        "dataset_archive_md5": checksums[dataset],
        "seeds": list(SEEDS),
        "aggregations": list(AGGREGATIONS),
        "clients": 20,
        "rounds": 5,
        "local_epochs": 3,
        "learning_rate": 0.01,
        "batch_size": 32,
        "attack": {"poisoned_clients": 4, "flip_fraction": 1.0, "source": 4, "target": 5},
        "clean_floors": FLOORS[dataset],
        "primary_metric": "(median attacked targeted error - median clean targeted error) - (fedavg attacked targeted error - fedavg clean targeted error)",
        "evaluate_test": False,
        "automatic_retries": False,
        **environment,
    }
    (results_dir / "comparison-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    partitions = {}
    records = {}
    failures = []
    attempted = 0
    stopped = False
    in_progress = None

    def save_outcome():
        outcome = {
            "dataset": dataset,
            "attempted": attempted,
            "expected": len(SEEDS) * len(AGGREGATIONS) * 2,
            "failures": failures,
            "in_progress": in_progress,
            "complete": attempted == len(SEEDS) * len(AGGREGATIONS) * 2 and not failures,
            "clean_summary": clean_summary(dataset, records),
        }
        (results_dir / "comparison-outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")

    save_outcome()
    print(f"\nStarting {dataset} {execution}: 40 fresh conditions", flush=True)
    for seed in SEEDS:
        for aggregation in AGGREGATIONS:
            for count in (0, 4):
                condition = f"dataset={dataset} seed={seed} aggregation={aggregation} clients={count}"
                print(f"Running {condition}", flush=True)
                in_progress = condition
                save_outcome()
                try:
                    result = subprocess.run(
                        make_command(dataset, execution, seed, aggregation, count), cwd=ROOT
                    )
                    if result.returncode:
                        raise ValueError(f"Flower exited {result.returncode}")
                    record = validate_condition(dataset, execution, seed, aggregation, count, partitions)
                    records[seed, aggregation, count] = record
                except (OSError, ValueError, KeyError, TypeError) as error:
                    failures.append(f"{condition}: {error}")
                    print(f"FAILED: {failures[-1]}", file=sys.stderr, flush=True)
                attempted += 1
                try:
                    verify_environment(environment, checksums)
                except (OSError, ValueError, subprocess.CalledProcessError) as error:
                    records.pop((seed, aggregation, count), None)
                    failures.append(f"{condition}: provenance check failed: {error}; remaining conditions not attempted")
                    print(f"FAILED: {failures[-1]}", file=sys.stderr, flush=True)
                    stopped = True
                in_progress = None
                save_outcome()
                if stopped:
                    break
            if stopped:
                break
        if stopped:
            break
    print(f"Finished {dataset} {execution}: {len(failures)} failures", flush=True)
    for aggregation, summary in clean_summary(dataset, records).items():
        print(f"{dataset} {aggregation} clean summary: {json.dumps(summary)}", flush=True)
    return execution, failures, stopped


def main():
    try:
        environment = comparison_environment(check_committed=True)
        checksums = prepare_datasets()
        verify_environment(environment, checksums)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Comparison preflight failed: {error}", file=sys.stderr)
        return 1
    failures = []
    for dataset in DATASETS:
        try:
            verify_environment(environment, checksums)
            _, failed, stopped = run_dataset(dataset, environment, checksums)
            failures.extend(failed)
            if stopped:
                break
        except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
            failures.append(f"dataset={dataset}: {error}; remaining conditions not attempted")
            print(f"FAILED: {failures[-1]}", file=sys.stderr, flush=True)
            break
    if failures:
        print("\nFailed or incomplete conditions:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("\nAll 80 conditions completed and passed record verification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
