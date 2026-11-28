"""Run the matched OrganCMNIST and OrganSMNIST defense matrices."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import run_organ_cs_defense_comparison as comparison
from run_organ_cs_defense_pilot import DATASETS, ROOT, prepare_datasets
from run_organa_defense_sweep import (
    AGGREGATIONS,
    SEEDS,
    load_records,
    matrix_conditions,
    normalized_dependencies,
    validate_record,
)

RESULTS = ROOT / "results/defense/generation-v1/protocol-v9"
MODELS = ROOT / "models/defense/generation-v1/protocol-v9"
REFERENCES = {"organcmnist": "run-1", "organsmnist": "run-2"}
RUNNER = Path(__file__).resolve()
ENVIRONMENT_KEYS = (
    "source_sha256", "comparison_runner_sha256", "python", "platform",
    "torch_cuda_available", "cuda_version", "cuda_devices",
    "visible_cuda_devices", "omp_num_threads", "mkl_num_threads",
)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_snapshot(root, records):
    paths = {root / "comparison-manifest.json", root / "comparison-outcome.json"}
    paths.update(root.glob("**/seed-*.json"))
    for record in records.values():
        paths.add(ROOT / record["partition_manifest"])
        paths.add(ROOT / record["model_artifact"])
        paths.update(ROOT / path for path in record["round_checkpoints"].values())
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(paths)}


def validate_reference(dataset, environment, checksum):
    execution = REFERENCES[dataset]
    root = comparison.RESULTS / execution
    manifest = json.loads((root / "comparison-manifest.json").read_text())
    outcome = json.loads((root / "comparison-outcome.json").read_text())
    if (
        manifest["dataset"] != dataset
        or manifest["dataset_archive_md5"] != checksum
        or manifest["seeds"] != list(SEEDS)
        or manifest["aggregations"] != list(AGGREGATIONS)
        or manifest["clients"] != 20
        or manifest["rounds"] != 5
        or manifest["local_epochs"] != 3
        or manifest["learning_rate"] != 0.01
        or manifest["batch_size"] != 32
        or manifest["attack"] != {"poisoned_clients": 4, "flip_fraction": 1.0, "source": 4, "target": 5}
        or manifest["clean_floors"] != comparison.FLOORS[dataset]
        or manifest["evaluate_test"]
        or manifest["automatic_retries"]
        or outcome["dataset"] != dataset
        or outcome["attempted"] != 40
        or outcome["expected"] != 40
        or not outcome["complete"]
        or outcome["failures"]
        or outcome["in_progress"] is not None
    ):
        raise ValueError(f"{dataset}: incomplete or incompatible reference comparison")
    mismatches = [key for key in ENVIRONMENT_KEYS if manifest[key] != environment[key]]
    if normalized_dependencies(manifest["dependencies"]) != normalized_dependencies(environment["dependencies"]):
        mismatches.append("dependencies")
    if mismatches:
        raise ValueError(f"{dataset}: reference environment differs: {', '.join(mismatches)}")

    records = load_records(root)
    expected = {(seed, agg, count) for seed in SEEDS for agg in AGGREGATIONS for count in (0, 4)}
    if set(records) != expected:
        raise ValueError(f"{dataset}: reference condition set differs")
    partitions = {}
    for seed, aggregation, count in sorted(expected):
        comparison.validate_condition(dataset, execution, seed, aggregation, count, partitions)
    summary = comparison.clean_summary(dataset, records)
    if summary != outcome["clean_summary"]:
        raise ValueError(f"{dataset}: reference clean summary differs from records")
    if any(not summary[agg]["complete"] or not summary[agg]["accuracy_floor_met"]
           or not summary[agg]["kidney_left_recall_floor_met"] for agg in AGGREGATIONS):
        raise ValueError(f"{dataset}: reference clean floors were not met")
    return {
        "execution": execution,
        "manifest_sha256": file_hash(root / "comparison-manifest.json"),
        "partitions": partitions,
        "snapshot": reference_snapshot(root, records),
    }


def sweep_environment(check_committed=False):
    environment = comparison.comparison_environment(check_committed=check_committed)
    contents = RUNNER.read_bytes()
    if check_committed:
        committed = subprocess.check_output(
            ["git", "show", "HEAD:quickstart-pytorch/run_organ_cs_defense_sweep.py"], cwd=ROOT
        )
        if contents != committed:
            raise ValueError("Commit the OrganC/S sweep runner before starting")
    environment["sweep_runner_sha256"] = hashlib.sha256(contents).hexdigest()
    return environment


def preflight():
    environment = sweep_environment(check_committed=True)
    checksums = prepare_datasets()
    comparison.verify_environment(
        {key: value for key, value in environment.items() if key != "sweep_runner_sha256"},
        checksums,
    )
    references = {
        dataset: validate_reference(dataset, environment, checksums[dataset])
        for dataset in DATASETS
    }
    return environment, checksums, references


def verify_environment(environment, checksums):
    if sweep_environment() != environment:
        raise ValueError("Source, revision or environment changed during the sweep")
    for dataset, checksum in checksums.items():
        if comparison.archive_checksum(dataset) != checksum:
            raise ValueError(f"{dataset} archive changed during the sweep")


def verify_reference(dataset, reference):
    root = comparison.RESULTS / reference["execution"]
    records = load_records(root)
    if reference_snapshot(root, records) != reference["snapshot"]:
        raise ValueError(f"{dataset}: reference records or artifacts changed during the sweep")


def next_execution_id():
    numbers = []
    for root in (RESULTS, MODELS):
        for path in root.glob("run-*"):
            suffix = path.name.removeprefix("run-")
            if suffix.isdigit():
                numbers.append(int(suffix))
    return f"run-{max(numbers, default=0) + 1}"


def make_command(dataset, execution, seed, aggregation, count, fraction):
    config = (
        f'experiment-type="label-flipping" aggregation="{aggregation}" '
        f'aggregation-subdirectories=true require-complete-round=true '
        f'dataset="{dataset}" seed={seed} num-clients=20 num-server-rounds=5 '
        f'local-epochs=3 learning-rate=0.01 batch-size=32 '
        f'poison-num-clients={count} poison-fraction={fraction} '
        f'poison-source-class-id=4 poison-target-class-id=5 evaluate-test=false '
        f'generation-id="generation-v1" protocol-id="protocol-v9" '
        f'execution-id="{execution}" results-dir="results/defense" '
        f'models-dir="models/defense"'
    )
    return ["flwr", "run", ".", "--federation-config", "num-supernodes=20",
            "--run-config", config, "--stream"]


def validate_condition(dataset, execution, seed, aggregation, count, fraction, reference_partitions):
    from pytorchexample.experiments.poisoning.label_flipping import select_poisoned_clients
    from pytorchexample.shared.data import get_class_names

    prefix = "median_" if aggregation == "median" else ""
    tag = f"{float(fraction):.3f}".replace(".", "p")
    experiment = f"{prefix}label_flip_r5_e3_lr0p010_clients{count}_f{tag}_s4_t5"
    paths = list((RESULTS / execution / aggregation / dataset / experiment).glob(f"seed-{seed}_run-*.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one result; found {len(paths)}")
    record = json.loads(paths[0].read_text())
    if (
        record["generation_id"] != "generation-v1"
        or record["protocol_id"] != "protocol-v9"
        or record["execution_id"] != execution
        or record["artifact_layout"] != "aggregation_subdirectories"
    ):
        raise ValueError("Incorrect execution metadata")
    clients = validate_record(record, seed, aggregation, count, float(fraction), dataset=dataset, local_epochs=3)
    partition = json.loads((ROOT / record["partition_manifest"]).read_text())
    if (partition["dataset"], partition["seed"], partition["num_clients"]) != (dataset, seed, 20):
        raise ValueError("Incorrect partition metadata")
    if clients != reference_partitions[seed]:
        raise ValueError("Partition summary differs from the matched reference")
    metrics = record["metrics"]
    names = get_class_names(dataset)
    validation = metrics["validation_by_round"]
    targeted = metrics["targeted_source_to_target_misclassification"]
    if (
        metrics["class_names"] != names
        or set(validation) != {str(i) for i in range(6)}
        or set(targeted["validation_by_round"]) != {str(i) for i in range(6)}
        or any(
            set(item["per_class"]) != set(names)
            or not math.isfinite(item["accuracy"])
            or not math.isfinite(item["loss"])
            or any(not math.isfinite(item["per_class"][name]["recall"]) for name in names)
            for item in validation.values()
        )
        or any(not math.isfinite(item["rate"]) or item["support"] <= 0
               for item in targeted["validation_by_round"].values())
        or targeted["test_final"] is not None
    ):
        raise ValueError("Missing or invalid validation metrics")
    attack = record["configuration"]["label_flipping"]
    selected = select_poisoned_clients(20, count, seed)
    expected_flips = {
        str(client["partition_id"]): round(client["class_counts"]["kidney-left"] * float(fraction))
        for client in clients if client["partition_id"] in selected
    }
    if attack["poisoned_client_ids"] != selected:
        raise ValueError("Attacker selection differs from the seeded selection")
    if (
        len(expected_flips) != count
        or attack["flipped_source_examples_by_client"] != expected_flips
        or attack["total_flipped_source_examples"] != sum(expected_flips.values())
    ):
        raise ValueError("Recorded flip counts differ from the partition and fraction")
    return record


def run_dataset(dataset, environment, checksums, reference):
    execution = next_execution_id()
    results_dir = RESULTS / execution
    models_dir = MODELS / execution
    results_dir.mkdir(parents=True, exist_ok=False)
    models_dir.mkdir(parents=True, exist_ok=False)
    conditions = [(seed, agg, count, fraction) for count, fraction in matrix_conditions()
                  for seed in SEEDS for agg in AGGREGATIONS]
    manifest = {
        "description": "Matched validation-only Organ defense matrix",
        "dataset": dataset,
        "dataset_archive_md5": checksums[dataset],
        "reference_execution": str((comparison.RESULTS / reference["execution"]).relative_to(ROOT)),
        "reference_manifest_sha256": reference["manifest_sha256"],
        "reference_artifact_sha256": reference["snapshot"],
        "seeds": list(SEEDS),
        "aggregations": list(AGGREGATIONS),
        "attack_settings": [{"poisoned_clients": count, "flip_fraction": float(fraction)}
                            for count, fraction in matrix_conditions()],
        "clients": 20, "rounds": 5, "local_epochs": 3, "learning_rate": 0.01,
        "batch_size": 32, "source_class_id": 4, "target_class_id": 5,
        "evaluate_test": False, "automatic_retries": False,
        **environment,
    }
    (results_dir / "sweep-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    failures = []
    attempted = 0
    in_progress = None
    stopped = False
    reference_checked = False

    def save_outcome():
        outcome = {
            "dataset": dataset, "expected": len(conditions), "attempted": attempted,
            "in_progress": in_progress, "failures": failures,
            "complete": (attempted == len(conditions) and not failures
                         and in_progress is None and reference_checked),
        }
        (results_dir / "sweep-outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")

    save_outcome()
    print(f"\nStarting {dataset} {execution}: {len(conditions)} new conditions", flush=True)
    for index, (seed, aggregation, count, fraction) in enumerate(conditions):
        condition = (f"dataset={dataset} seed={seed} aggregation={aggregation} "
                     f"clients={count} fraction={fraction}")
        try:
            verify_environment(environment, checksums)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            failures.append(f"{condition}: provenance check failed: {error}; remaining conditions not attempted")
            stopped = True
            save_outcome()
            break
        print(f"Running {condition}", flush=True)
        in_progress = condition
        save_outcome()
        try:
            result = subprocess.run(make_command(dataset, execution, seed, aggregation, count, fraction), cwd=ROOT)
            if result.returncode:
                raise ValueError(f"Flower exited {result.returncode}")
            validate_condition(dataset, execution, seed, aggregation, count, fraction, reference["partitions"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(f"{condition}: {error}")
            print(f"FAILED: {failures[-1]}", file=sys.stderr, flush=True)
        attempted += 1
        try:
            verify_environment(environment, checksums)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            failures.append(f"{condition}: provenance check failed: {error}; remaining conditions not attempted")
            stopped = True
        if not stopped and (index + 1) % (len(SEEDS) * len(AGGREGATIONS)) == 0:
            try:
                verify_reference(dataset, reference)
            except (OSError, ValueError, KeyError, TypeError) as error:
                failures.append(f"{condition}: reference check failed: {error}; remaining conditions not attempted")
                stopped = True
        in_progress = None
        save_outcome()
        if stopped:
            break
    if not stopped:
        reference_checked = True
        save_outcome()
    print(f"Finished {dataset} {execution}: {len(failures)} failures", flush=True)
    return failures, stopped


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run 600 new matched OrganC/S defense conditions sequentially."
    )
    parser.parse_args(argv)
    try:
        environment, checksums, references = preflight()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Sweep preflight failed: {error}", file=sys.stderr)
        return 1
    failures = []
    for dataset in DATASETS:
        try:
            verify_environment(environment, checksums)
            verify_reference(dataset, references[dataset])
            failed, stopped = run_dataset(dataset, environment, checksums, references[dataset])
            failures.extend(failed)
            if stopped:
                break
        except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
            failures.append(f"dataset={dataset}: {error}; remaining datasets not attempted")
            break
    if failures:
        print("\nFailed or incomplete conditions:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("\nAll 600 new OrganC/S conditions completed and passed verification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
