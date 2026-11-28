"""Run matched, validation-only OrganCMNIST and OrganSMNIST defense pilots."""

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from run_organa_defense_sweep import validate_record

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/defense/generation-v1/protocol-v5"
MODELS = ROOT / "models/defense/generation-v1/protocol-v5"
DATASETS = ("organcmnist", "organsmnist")
SEEDS = (42, 49, 50)
AGGREGATIONS = ("fedavg", "median")


def source_paths():
    return sorted(ROOT.glob("pytorchexample/**/*.py")) + [
        ROOT / "pyproject.toml",
        ROOT / "run_median_defense_pilot.sh",
        ROOT / "run_organa_defense_sweep.py",
        ROOT / "run_organ_cs_defense_pilot.py",
    ]


def archive_checksum(dataset):
    digest = hashlib.md5(usedforsecurity=False)
    with (Path.home() / ".medmnist" / f"{dataset}.npz").open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_datasets():
    from medmnist import INFO
    from pytorchexample.shared.data import get_class_names, get_dataset

    checksums = {}
    for dataset in DATASETS:
        if get_class_names(dataset)[4:6] != ["kidney-left", "kidney-right"]:
            raise ValueError(f"{dataset}: unexpected attack class mapping")
        print(f"Preparing {dataset}; MedMNIST may download the dataset if missing.", flush=True)
        get_dataset("train", dataset)
        checksum = archive_checksum(dataset)
        if checksum != INFO[dataset]["MD5"]:
            raise ValueError(f"{dataset}: archive checksum does not match MedMNIST metadata")
        checksums[dataset] = checksum
    return checksums


def capture_environment(check_committed=False):
    import torch

    python = shutil.which("python")
    flwr = shutil.which("flwr")
    if (
        not python
        or not flwr
        or Path(python).absolute() != Path(sys.executable).absolute()
        or Path(flwr).absolute().parent != Path(sys.executable).absolute().parent
    ):
        raise ValueError("Activate the same environment for python and flwr")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("Commit tracked changes before starting the pilots")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = source_paths()
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        contents = path.read_bytes()
        if check_committed:
            committed = subprocess.check_output(
                ["git", "show", f"HEAD:quickstart-pytorch/{relative}"], cwd=ROOT
            )
            if committed != contents:
                raise ValueError(f"Commit the source file before starting: {relative}")
        hashes[str(relative)] = hashlib.sha256(contents).hexdigest()
    return {
        "git_revision": revision,
        "source_sha256": hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "dependencies": subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], cwd=ROOT, text=True
        ).splitlines(),
    }


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
        f'local-epochs=1 learning-rate=0.01 batch-size=32 '
        f'poison-num-clients={count} poison-fraction={fraction} '
        f'poison-source-class-id=4 poison-target-class-id=5 evaluate-test=false '
        f'generation-id="generation-v1" protocol-id="protocol-v5" '
        f'execution-id="{execution}" results-dir="results/defense" '
        f'models-dir="models/defense"'
    )
    return ["flwr", "run", ".", "--federation-config", "num-supernodes=20",
            "--run-config", config, "--stream"]


def validate_condition(dataset, execution, seed, aggregation, count, partitions, attacks):
    from pytorchexample.experiments.poisoning.label_flipping import select_poisoned_clients
    from pytorchexample.shared.data import get_class_names

    key = (seed, aggregation, count)
    prefix = "median_" if aggregation == "median" else ""
    fraction = "1p000" if count else "0p000"
    experiment = f"{prefix}label_flip_r5_e1_lr0p010_clients{count}_f{fraction}_s4_t5"
    paths = list((RESULTS / execution / aggregation / dataset / experiment).glob(f"seed-{seed}_run-*.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one result for {key}, found {len(paths)}")
    record = json.loads(paths[0].read_text())
    if record["generation_id"] != "generation-v1" or record["protocol_id"] != "protocol-v5" or record["execution_id"] != execution:
        raise ValueError(f"Incorrect execution metadata for {key}")
    clients = validate_record(record, seed, aggregation, count, float(count == 4), dataset=dataset)
    metrics = record["metrics"]
    names = get_class_names(dataset)
    final = metrics["validation_by_round"]["5"]
    targeted = metrics["targeted_source_to_target_misclassification"]
    if (
        metrics["class_names"] != names
        or set(final["per_class"]) != set(names)
        or not math.isfinite(final["accuracy"])
        or not math.isfinite(final["loss"])
        or any(not math.isfinite(final["per_class"][name]["recall"]) for name in names)
        or not math.isfinite(targeted["validation_by_round"]["5"]["rate"])
        or targeted["validation_by_round"]["5"]["support"] <= 0
        or targeted["test_final"] is not None
    ):
        raise ValueError(f"Missing or invalid final validation metrics for {key}")
    if seed in partitions and partitions[seed] != clients:
        raise ValueError(f"Partition summaries differ for {key}")
    partitions[seed] = clients
    if count == 4:
        attack = record["configuration"]["label_flipping"]
        if attack["poisoned_client_ids"] != select_poisoned_clients(20, 4, seed):
            raise ValueError(f"Attacker selection differs for {key}")
        details = (attack["flipped_source_examples_by_client"], attack["total_flipped_source_examples"])
        if seed in attacks and attacks[seed] != details:
            raise ValueError(f"Flipped-example counts differ between aggregations for seed {seed}")
        attacks[seed] = details


def verify_environment(dataset, checksum, environment):
    if capture_environment() != environment or archive_checksum(dataset) != checksum:
        raise ValueError("Source, environment or dataset changed during the pilots")


def run_dataset(dataset, checksum, environment):
    execution = next_execution_id()
    results_dir = RESULTS / execution
    models_dir = MODELS / execution
    results_dir.mkdir(parents=True, exist_ok=False)
    models_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "description": "Matched validation-only Organ defense stability pilot",
        "dataset": dataset,
        "dataset_archive_md5": checksum,
        "seeds": list(SEEDS),
        "learning_rate": 0.01,
        "aggregations": list(AGGREGATIONS),
        "includes_clean_controls": True,
        "poison_num_clients": 4,
        "poison_fraction": 1.0,
        "continue_on_failure": True,
        "evaluate_test": False,
        **environment,
    }
    (results_dir / "pilot-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    failures = []
    partitions = {}
    attacks = {}
    attempted = 0
    stopped = False
    print(f"\nStarting {dataset} {execution}: 12 conditions", flush=True)
    for seed in SEEDS:
        for aggregation in AGGREGATIONS:
            for count in (0, 4):
                condition = f"dataset={dataset} seed={seed} aggregation={aggregation} clients={count}"
                print(f"Running {condition}", flush=True)
                result = subprocess.run(make_command(dataset, execution, seed, aggregation, count), cwd=ROOT)
                attempted += 1
                if result.returncode:
                    failures.append(f"{condition}: Flower exited {result.returncode}")
                    print(f"FAILED: {failures[-1]}", file=sys.stderr, flush=True)
                else:
                    try:
                        validate_condition(dataset, execution, seed, aggregation, count, partitions, attacks)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        failures.append(f"{condition}: {error}")
                        print(f"FAILED: {failures[-1]}", file=sys.stderr, flush=True)
                try:
                    verify_environment(dataset, checksum, environment)
                except (OSError, ValueError, subprocess.CalledProcessError) as error:
                    failures.append(f"{condition}: provenance check failed: {error}; remaining conditions not attempted")
                    print(f"FAILED: {failures[-1]}", file=sys.stderr, flush=True)
                    stopped = True
                    break
            if stopped:
                break
        if stopped:
            break
    outcome = {"attempted": attempted, "expected": 12, "failures": failures, "complete": attempted == 12 and not failures}
    (results_dir / "pilot-outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
    return execution, failures, stopped


def main():
    try:
        environment = capture_environment(check_committed=True)
        checksums = prepare_datasets()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Pilot preflight failed: {error}", file=sys.stderr)
        return 1
    failures = []
    for dataset in DATASETS:
        try:
            verify_environment(dataset, checksums[dataset], environment)
            execution, failed, stopped = run_dataset(dataset, checksums[dataset], environment)
            failures.extend(failed)
            print(f"Finished {dataset} {execution}: {len(failed)} failed conditions", flush=True)
            if stopped:
                break
        except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
            failures.append(f"dataset={dataset}: {error}")
            print(f"FAILED: {failures[-1]}", file=sys.stderr, flush=True)
            break
    if failures:
        print("\nFailed or incomplete conditions:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("\nAll 24 pilot conditions completed and passed verification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
