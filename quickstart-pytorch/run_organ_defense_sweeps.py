"""Run and resume the three expanded Organ defense matrices."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import run_organ_cs_defense_sweep as organ_cs
import run_organa_defense_sweep as organ_a

ROOT = Path(__file__).resolve().parent
RUNNER = Path(__file__).resolve()
STATE_ROOT = ROOT / "results/defense/generation-v1/resumable-sweeps"
MIN_FREE_BYTES = 1024 ** 3
DATASETS = ("organamnist", *organ_cs.DATASETS)


def conditions():
    return [
        {"dataset": dataset, "seed": seed, "aggregation": aggregation,
         "count": count, "fraction": fraction, "attempts": []}
        for dataset in DATASETS
        for count, fraction in organ_a.matrix_conditions()
        for seed in organ_a.SEEDS
        for aggregation in organ_a.AGGREGATIONS
    ]


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".sweep-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def reference_a_snapshot():
    records = organ_a.load_records(organ_a.REFERENCE)
    paths = {organ_a.REFERENCE / "pilot-manifest.json"}
    paths.update(organ_a.REFERENCE.glob("**/seed-*.json"))
    for record in records.values():
        paths.add(ROOT / record["partition_manifest"])
        paths.add(ROOT / record["model_artifact"])
        paths.update(ROOT / path for path in record["round_checkpoints"].values())
    return {str(path.relative_to(ROOT)): organ_cs.file_hash(path) for path in sorted(paths)}


def preflight():
    committed = subprocess.check_output(
        ["git", "show", "HEAD:quickstart-pytorch/run_organ_defense_sweeps.py"], cwd=ROOT
    )
    if committed != RUNNER.read_bytes():
        raise ValueError("Commit the all-dataset launcher before starting")
    reference, partitions = organ_a.validate_reference()
    organ_a.validate_environment(reference)
    environment, checksums, references = organ_cs.preflight()
    provenance = {
        "revision": environment["git_revision"],
        "launcher_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
        "environment": environment,
        "checksums": checksums,
        "organ_a_reference": reference_a_snapshot(),
        "organ_cs_references": {
            dataset: references[dataset]["snapshot"] for dataset in organ_cs.DATASETS
        },
    }
    return provenance, reference, partitions, environment, checksums, references


def validate_a(condition, execution, reference, partitions, revision):
    from pytorchexample.experiments.poisoning.label_flipping import select_poisoned_clients

    seed, aggregation, count, fraction = (
        condition["seed"], condition["aggregation"], condition["count"], condition["fraction"]
    )
    root = organ_a.RESULTS / execution
    manifest = json.loads((root / "pilot-manifest.json").read_text())
    keys = (
        "source_sha256", "dataset_archive_md5", "python", "platform",
        "torch_cuda_available", "cuda_version", "cuda_devices",
        "visible_cuda_devices", "omp_num_threads", "mkl_num_threads",
    )
    if (
        any(manifest[key] != reference[key] for key in keys)
        or organ_a.normalized_dependencies(manifest["dependencies"])
        != organ_a.normalized_dependencies(reference["dependencies"])
        or manifest["git_revision"] != revision
        or any(not item.startswith("?? ") for item in manifest["git_status"])
        or manifest["seeds"] != [seed]
        or manifest["aggregations"] != [aggregation]
        or manifest["includes_clean_controls"]
        or manifest["learning_rate"] != 0.01
        or manifest["poison_num_clients"] != count
        or manifest["poison_fraction"] != float(fraction)
        or not manifest["continue_on_failure"]
        or manifest["evaluate_test"]
    ):
        raise ValueError(f"{execution}: incompatible pilot manifest")
    records = organ_a.load_records(root)
    key = (seed, aggregation, count)
    if set(records) != {key}:
        raise ValueError(f"{execution}: expected exactly one matching result")
    record = records[key]
    if (
        record["generation_id"] != "generation-v1"
        or record["protocol_id"] != "protocol-v4"
        or record["execution_id"] != execution
        or record["artifact_layout"] != "aggregation_subdirectories"
    ):
        raise ValueError(f"{execution}: incorrect execution metadata")
    clients = organ_a.validate_record(record, seed, aggregation, count, float(fraction))
    if clients != partitions[seed]:
        raise ValueError(f"{execution}: partition differs from reference")
    selected = select_poisoned_clients(20, count, seed)
    attack = record["configuration"]["label_flipping"]
    flips = {
        str(client["partition_id"]): round(client["class_counts"]["kidney-left"] * float(fraction))
        for client in clients if client["partition_id"] in selected
    }
    if (
        attack["poisoned_client_ids"] != selected
        or len(flips) != count
        or attack["flipped_source_examples_by_client"] != flips
        or attack["total_flipped_source_examples"] != sum(flips.values())
    ):
        raise ValueError(f"{execution}: attacker selection or flip counts differ")


def validate_attempt(condition, execution, context):
    _, reference, partitions, environment, _, references = context
    if condition["dataset"] == "organamnist":
        validate_a(condition, execution, reference, partitions, environment["git_revision"])
    else:
        manifest = json.loads((organ_cs.RESULTS / execution / "attempt-manifest.json").read_text())
        expected = attempt_manifest(condition, execution, context[0])
        if manifest != expected:
            raise ValueError(f"{execution}: attempt manifest differs from sweep provenance")
        organ_cs.validate_condition(
            condition["dataset"], execution, condition["seed"], condition["aggregation"],
            condition["count"], condition["fraction"],
            references[condition["dataset"]]["partitions"],
        )


def attempt_manifest(condition, execution, provenance):
    digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    return {
        "condition": {key: value for key, value in condition.items() if key != "attempts"},
        "execution": execution,
        "git_revision": provenance["revision"],
        "sweep_provenance_sha256": digest,
    }


def run_attempt(condition, execution, lock_fd):
    if condition["dataset"] == "organamnist":
        env = os.environ.copy()
        env.update({
            "SEEDS": str(condition["seed"]),
            "AGGREGATIONS": condition["aggregation"],
            "RUN_CLEAN_CONTROLS": "false",
            "POISON_NUM_CLIENTS": str(condition["count"]),
            "POISON_FRACTION": condition["fraction"],
            "LEARNING_RATE": "0.01",
            "CONTINUE_ON_FAILURE": "true",
            "GENERATION_ID": "generation-v1",
            "PROTOCOL_ID": "protocol-v4",
            "EXECUTION_ID": execution,
            "RESULTS_DIR": "results/defense",
            "MODELS_DIR": "models/defense",
        })
        command = ["bash", "run_median_defense_pilot.sh"]
    else:
        env = None
        command = organ_cs.make_command(
            condition["dataset"], execution, condition["seed"], condition["aggregation"],
            condition["count"], condition["fraction"],
        )
    return subprocess.run(command, cwd=ROOT, env=env, pass_fds=(lock_fd,)).returncode


def verify_context(context):
    provenance, reference, _, environment, checksums, references = context
    organ_a.check_sweep_environment(reference, provenance["revision"])
    organ_cs.verify_environment(environment, checksums)
    if reference_a_snapshot() != provenance["organ_a_reference"]:
        raise ValueError("OrganA reference artifacts changed")
    for dataset in organ_cs.DATASETS:
        organ_cs.verify_reference(dataset, references[dataset])


def verify_environment(context):
    provenance, reference, _, environment, checksums, _ = context
    organ_a.check_sweep_environment(reference, provenance["revision"])
    organ_cs.verify_environment(environment, checksums)


def reserve_execution(dataset):
    is_a = dataset == "organamnist"
    root = organ_a.RESULTS if is_a else organ_cs.RESULTS
    next_id = organ_a.next_execution_id if is_a else organ_cs.next_execution_id
    reservations = root / ".sweep-reservations"
    reservations.mkdir(parents=True, exist_ok=True)
    number = int(next_id().removeprefix("run-"))
    while True:
        execution = f"run-{number}"
        try:
            with (reservations / execution).open("x"):
                return execution
        except FileExistsError:
            number += 1


def run_sweep(path, state, context, retry_reason, lock_fd):
    if state["provenance"] != context[0]:
        raise ValueError("Sweep provenance differs from the current environment or references")
    expected = [{k: v for k, v in item.items() if k != "attempts"} for item in conditions()]
    actual = [{k: v for k, v in item.items() if k != "attempts"} for item in state["conditions"]]
    if actual != expected:
        raise ValueError("Saved sweep conditions differ from the frozen matrix")
    verify_context(context)
    for item in state["conditions"]:
        attempts = item["attempts"]
        if attempts and attempts[-1]["status"] == "completed":
            validate_attempt(item, attempts[-1]["execution"], context)

    if retry_reason and not any(
        item["attempts"] and len(item["attempts"]) < 2
        and item["attempts"][-1]["status"] in ("failed", "in_progress")
        for item in state["conditions"]
    ):
        raise ValueError("No failed or interrupted condition is eligible for a retry")
    incomplete = []
    for index, item in enumerate(state["conditions"]):
        attempts = item["attempts"]
        if attempts and attempts[-1]["status"] == "completed":
            continue
        if attempts:
            if len(attempts) >= 2 or not retry_reason:
                incomplete.append(item)
                continue
        if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
            print("Stopping before the next condition: less than 1 GiB free on the workspace filesystem.",
                  file=sys.stderr, flush=True)
            return 1
        if index % 20 == 0:
            verify_context(context)
        verify_environment(context)
        execution = reserve_execution(item["dataset"])
        attempt = {"execution": execution, "status": "in_progress",
                   "retry_reason": retry_reason.strip() if attempts else None}
        attempts.append(attempt)
        retry_reason = None
        atomic_write(path, state)
        print(f"Running {item['dataset']} seed={item['seed']} aggregation={item['aggregation']} "
              f"clients={item['count']} fraction={item['fraction']} execution={execution}", flush=True)
        try:
            if item["dataset"] != "organamnist":
                result_dir = organ_cs.RESULTS / execution
                result_dir.mkdir(parents=True, exist_ok=False)
                atomic_write(result_dir / "attempt-manifest.json",
                             attempt_manifest(item, execution, context[0]))
            code = run_attempt(item, execution, lock_fd)
            if code:
                raise ValueError(f"Flower/pilot exited {code}")
            validate_attempt(item, execution, context)
            verify_environment(context)
        except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
            attempt["status"] = "failed"
            attempt["error"] = str(error)
            atomic_write(path, state)
            print(f"Condition failed; preserving attempt and continuing: {error}",
                  file=sys.stderr, flush=True)
            incomplete.append(item)
            continue
        attempt["status"] = "completed"
        atomic_write(path, state)
    verify_context(context)
    if incomplete:
        print(f"{len(incomplete)} conditions remain incomplete. Investigate each failure; "
              "only confirmed infrastructure failures may be retried once with "
              "--retry-infrastructure REASON.", file=sys.stderr, flush=True)
        for item in incomplete:
            print(f"  {item}", file=sys.stderr, flush=True)
        return 1
    print("All 900 Organ defense conditions completed and passed verification.", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run or resume all three expanded Organ defense sweeps.")
    parser.add_argument("--resume", type=Path, help="Path to the saved sweep state JSON")
    parser.add_argument("--retry-infrastructure", metavar="REASON",
                        help="Retry the first failed/interrupted condition once, with this reason")
    args = parser.parse_args(argv)
    if args.retry_infrastructure is not None and (
        args.resume is None or not args.retry_infrastructure.strip()
    ):
        parser.error("--retry-infrastructure requires --resume and a non-empty reason")
    path = args.resume.resolve() if args.resume else STATE_ROOT / f"sweep-{uuid.uuid4().hex}.json"
    try:
        if args.resume and not path.is_file():
            raise ValueError(f"Sweep state does not exist: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(path.suffix + ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(f"Sweep is already running: {path}") from error
            context = preflight()
            if args.resume:
                state = json.loads(path.read_text())
                if state["version"] != 1:
                    raise ValueError("Unsupported sweep state version")
            else:
                state = {"version": 1, "provenance": context[0], "conditions": conditions()}
                atomic_write(path, state)
            print(f"Sweep state: {path}", flush=True)
            return run_sweep(path, state, context, args.retry_infrastructure, lock.fileno())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError,
            subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"Sweep stopped: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
