"""Run the matched OrganAMNIST defense matrix using the completed controls."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "results/defense/generation-v1/protocol-v3/run-2"
RESULTS = ROOT / "results/defense/generation-v1/protocol-v4"
MODELS = ROOT / "models/defense/generation-v1/protocol-v4"
SEEDS = range(42, 52)
AGGREGATIONS = ("fedavg", "median")
COUNTS = (1, 2, 4, 8)
FRACTIONS = ("0.25", "0.5", "0.75", "1.0")


def matrix_conditions():
    return [
        (count, fraction)
        for count in COUNTS
        for fraction in FRACTIONS
        if (count, fraction) != (4, "1.0")
    ]


def load_records(root):
    records = {}
    for path in root.glob("**/seed-*.json"):
        record = json.loads(path.read_text())
        key = (
            record["seed"],
            record["aggregation_id"],
            record["configuration"]["label_flipping"]["num_poisoned_clients"],
        )
        if key in records:
            raise ValueError(f"Duplicate condition {key} in {root}")
        records[key] = record
    return records


def validate_record(record, seed, aggregation, count, fraction, dataset="organamnist"):
    config = record["configuration"]
    attack = config["label_flipping"]
    training = record["training_results"]
    if (
        record["status"] != "completed"
        or record["dataset"] != dataset
        or record["seed"] != seed
        or record["aggregation_id"] != aggregation
        or record["aggregation"] != (
            "FedAvg" if aggregation == "fedavg" else "coordinate-wise median"
        )
        or not training["valid_run"]
        or not training["metrics_finite"]
        or not training["model_parameters_finite"]
        or training["received_results_by_round"] != {str(i): 20 for i in range(1, 6)}
        or training["client_failures_by_round"] != {str(i): 0 for i in range(1, 6)}
        or record["metrics"]["test_final"] is not None
        or config["clients"] != 20
        or config["federated_rounds"] != 5
        or config["local_epochs"] != 1
        or config["learning_rate"] != 0.01
        or config["batch_size"] != 32
        or not config["require_complete_round"]
        or attack["num_poisoned_clients"] != count
        or attack["fraction_of_source_labels_flipped_per_poisoned_client"] != fraction
        or attack["source_class_id"] != 4
        or attack["target_class_id"] != 5
    ):
        raise ValueError(f"Invalid condition: seed={seed}, aggregation={aggregation}, clients={count}, fraction={fraction}")
    partition = json.loads((ROOT / record["partition_manifest"]).read_text())
    verification = partition["verification"]
    if not all(
        verification[key]
        for key in ("partitions_are_disjoint", "all_training_examples_assigned", "each_training_example_assigned_once")
    ):
        raise ValueError(f"Invalid partition for seed={seed}, aggregation={aggregation}, clients={count}")
    if (
        set(record["round_checkpoints"]) != {str(i) for i in range(1, 6)}
        or not all((ROOT / path).is_file() for path in record["round_checkpoints"].values())
        or not (ROOT / record["model_artifact"]).is_file()
    ):
        raise ValueError(f"Missing checkpoint or model for seed={seed}, aggregation={aggregation}, clients={count}")
    return partition["clients"]


def validate_reference():
    manifest = json.loads((REFERENCE / "pilot-manifest.json").read_text())
    if (
        manifest["seeds"] != list(SEEDS)
        or manifest["learning_rate"] != 0.01
        or manifest["aggregations"] != list(AGGREGATIONS)
        or not manifest["includes_clean_controls"]
        or manifest["poison_num_clients"] != 4
        or manifest["poison_fraction"] != 1.0
        or manifest["evaluate_test"]
    ):
        raise ValueError("Reference manifest does not describe the frozen 40-run comparison")

    records = load_records(REFERENCE)
    expected = {(seed, agg, count) for seed in SEEDS for agg in AGGREGATIONS for count in (0, 4)}
    if set(records) != expected:
        raise ValueError(f"Reference conditions mismatch: missing={sorted(expected - set(records))}, unexpected={sorted(set(records) - expected)}")
    partitions = {}
    for seed in SEEDS:
        selected = []
        for aggregation in AGGREGATIONS:
            for count in (0, 4):
                record = records[seed, aggregation, count]
                clients = validate_record(record, seed, aggregation, count, float(count == 4))
                if seed in partitions and partitions[seed] != clients:
                    raise ValueError(f"Reference partitions differ for seed {seed}")
                partitions[seed] = clients
                if count == 4:
                    selected.append(record["configuration"]["label_flipping"]["poisoned_client_ids"])
        if selected[0] != selected[1] or len(selected[0]) != 4:
            raise ValueError(f"Reference attacker selections differ for seed {seed}")
    return manifest, partitions


def validate_environment(manifest):
    import torch
    from medmnist import INFO

    python = shutil.which("python")
    flwr = shutil.which("flwr")
    if (
        not python
        or not flwr
        or Path(python).absolute() != Path(sys.executable).absolute()
        or Path(flwr).absolute().parent != Path(sys.executable).absolute().parent
    ):
        raise ValueError("Activate the same environment for python and flwr before starting")
    paths = sorted(ROOT.glob("pytorchexample/**/*.py"))
    paths += [ROOT / "pyproject.toml", ROOT / "run_median_defense_pilot.sh"]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    if hashes != manifest["source_sha256"]:
        raise ValueError("Training source or pilot runner differs from the reference manifest")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("Commit tracked changes before starting the sweep")
    committed_runner = subprocess.check_output(
        ["git", "show", "HEAD:quickstart-pytorch/run_organa_defense_sweep.py"], cwd=ROOT
    )
    if committed_runner != Path(__file__).read_bytes():
        raise ValueError("Commit the sweep runner before starting")
    environment = {
        "dataset_archive_md5": INFO["organamnist"]["MD5"],
        "python": sys.version,
        "platform": platform.platform(),
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }
    mismatches = [key for key, value in environment.items() if manifest[key] != value]
    if mismatches:
        raise ValueError(f"Environment differs from the reference manifest: {', '.join(mismatches)}")
    archive = Path.home() / ".medmnist/organamnist.npz"
    digest = hashlib.md5(usedforsecurity=False)
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest["dataset_archive_md5"]:
        raise ValueError(f"Dataset archive checksum differs from the reference: {archive}")
    dependencies = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], cwd=ROOT, text=True
    ).splitlines()
    if normalized_dependencies(dependencies) != normalized_dependencies(manifest["dependencies"]):
        raise ValueError("Installed dependencies differ from the reference manifest")


def normalized_dependencies(lines):
    return sorted(
        line for line in lines
        if not (line.startswith("-e git+") and "#egg=quickstart_pytorch" in line)
    )


def next_execution_id():
    numbers = []
    for root in (RESULTS, MODELS):
        for path in root.glob("run-*"):
            suffix = path.name.removeprefix("run-")
            if suffix.isdigit():
                numbers.append(int(suffix))
    return f"run-{max(numbers, default=0) + 1}"


def validate_new_results(execution, count, fraction, reference_partitions):
    from pytorchexample.experiments.poisoning.label_flipping import select_poisoned_clients

    records = load_records(RESULTS / execution)
    expected = {(seed, agg, count) for seed in SEEDS for agg in AGGREGATIONS}
    if set(records) != expected:
        raise ValueError(f"{execution}: missing={sorted(expected - set(records))}, unexpected={sorted(set(records) - expected)}")
    for seed in SEEDS:
        attack_details = []
        for aggregation in AGGREGATIONS:
            record = records[seed, aggregation, count]
            clients = validate_record(record, seed, aggregation, count, float(fraction))
            if clients != reference_partitions[seed]:
                raise ValueError(f"{execution}: partition differs for seed={seed}, aggregation={aggregation}")
            selected = record["configuration"]["label_flipping"]["poisoned_client_ids"]
            if selected != select_poisoned_clients(20, count, seed):
                raise ValueError(f"{execution}: attacker selection differs for seed={seed}, aggregation={aggregation}")
            attack = record["configuration"]["label_flipping"]
            attack_details.append(
                (attack["flipped_source_examples_by_client"], attack["total_flipped_source_examples"])
            )
        if attack_details[0] != attack_details[1]:
            raise ValueError(f"{execution}: flipped-example counts differ between aggregations for seed={seed}")


def record_sweep_provenance(execution, count, fraction, reference, revision):
    path = RESULTS / execution / "pilot-manifest.json"
    manifest = json.loads(path.read_text())
    keys = (
        "source_sha256", "dataset_archive_md5", "python", "platform",
        "torch_cuda_available", "cuda_version", "cuda_devices",
        "visible_cuda_devices", "omp_num_threads", "mkl_num_threads",
    )
    mismatches = [key for key in keys if manifest[key] != reference[key]]
    if (
        mismatches
        or normalized_dependencies(manifest["dependencies"]) != normalized_dependencies(reference["dependencies"])
        or manifest["git_revision"] != revision
        or any(not item.startswith("?? ") for item in manifest["git_status"])
        or manifest["seeds"] != list(SEEDS)
        or manifest["aggregations"] != list(AGGREGATIONS)
        or manifest["includes_clean_controls"]
        or manifest["learning_rate"] != 0.01
        or manifest["poison_num_clients"] != count
        or manifest["poison_fraction"] != float(fraction)
        or not manifest["continue_on_failure"]
        or manifest["evaluate_test"]
    ):
        raise ValueError(f"{execution}: manifest does not match the frozen sweep; differences: {mismatches}")
    manifest["sweep"] = {
        "reference_execution": str(REFERENCE.relative_to(ROOT)),
        "reference_manifest_sha256": hashlib.sha256(
            (REFERENCE / "pilot-manifest.json").read_bytes()
        ).hexdigest(),
        "sweep_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "poison_num_clients": count,
        "poison_fraction": float(fraction),
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def check_sweep_environment(reference, revision):
    validate_environment(reference)
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current != revision:
        raise ValueError("Git revision changed during the sweep")


def run_matrix(reference, reference_partitions, revision):
    failed = []
    for count, fraction in matrix_conditions():
        try:
            check_sweep_environment(reference, revision)
        except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
            print(f"Sweep aborted due to environment or source change: {error}", file=sys.stderr, flush=True)
            return 1
        execution = next_execution_id()
        print(f"\nStarting {execution}: clients={count}, fraction={fraction} (20 conditions)", flush=True)
        env = os.environ.copy()
        env.update({
            "SEEDS": " ".join(map(str, SEEDS)),
            "AGGREGATIONS": " ".join(AGGREGATIONS),
            "RUN_CLEAN_CONTROLS": "false",
            "POISON_NUM_CLIENTS": str(count),
            "POISON_FRACTION": fraction,
            "LEARNING_RATE": "0.01",
            "CONTINUE_ON_FAILURE": "true",
            "GENERATION_ID": "generation-v1",
            "PROTOCOL_ID": "protocol-v4",
            "EXECUTION_ID": execution,
            "RESULTS_DIR": "results/defense",
            "MODELS_DIR": "models/defense",
        })
        result = subprocess.run(["bash", "run_median_defense_pilot.sh"], cwd=ROOT, env=env)
        try:
            check_sweep_environment(reference, revision)
        except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
            print(f"Sweep aborted due to environment or source change: {error}", file=sys.stderr, flush=True)
            return 1
        try:
            record_sweep_provenance(execution, count, fraction, reference, revision)
        except (ValueError, KeyError, TypeError) as error:
            print(f"Sweep aborted due to invalid execution manifest: {error}", file=sys.stderr, flush=True)
            return 1
        except OSError as error:
            print(f"Execution manifest unavailable: {error}", file=sys.stderr, flush=True)
            failed.append((execution, count, fraction))
            continue
        try:
            validate_new_results(execution, count, fraction, reference_partitions)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"Result verification failed: {error}", file=sys.stderr, flush=True)
            failed.append((execution, count, fraction))
            continue
        if result.returncode:
            print(f"{execution} exited with status {result.returncode}", file=sys.stderr, flush=True)
            failed.append((execution, count, fraction))
    if failed:
        print("\nFailed or incomplete matrix settings:", file=sys.stderr)
        for execution, count, fraction in failed:
            print(f"  {execution}: clients={count}, fraction={fraction}", file=sys.stderr)
        return 1
    print("\nAll 300 new conditions completed and passed verification.")
    return 0


def main():
    try:
        manifest, partitions = validate_reference()
        validate_environment(manifest)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f"Sweep preflight failed: {error}", file=sys.stderr)
        return 1
    print(f"Reusing validated clean and 4-client/100% records from {REFERENCE}", flush=True)
    return run_matrix(manifest, partitions, revision)


if __name__ == "__main__":
    sys.exit(main())
