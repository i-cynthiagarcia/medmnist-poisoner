#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if (($# != 0)); then
  echo "Usage: $0 (optionally set SEEDS, AGGREGATIONS, RUN_CLEAN_CONTROLS, POISON_NUM_CLIENTS, POISON_FRACTION, LEARNING_RATE, CONTINUE_ON_FAILURE, GENERATION_ID, PROTOCOL_ID, EXECUTION_ID, RESULTS_DIR, MODELS_DIR)" >&2
  exit 2
fi

read -r -a seeds <<< "${SEEDS:-42 43 44 45 46}"
if ((${#seeds[@]} == 0)); then
  echo "SEEDS must contain at least one seed." >&2
  exit 2
fi
declare -A seen_seeds
for seed in "${seeds[@]}"; do
  if [[ ! "$seed" =~ ^(0|[1-9][0-9]*)$ ]] || [[ -v "seen_seeds[$seed]" ]]; then
    echo "SEEDS must contain distinct non-negative integers without leading zeros." >&2
    exit 2
  fi
  seen_seeds["$seed"]=1
done

read -r -a aggregations <<< "${AGGREGATIONS:-fedavg median}"
if ((${#aggregations[@]} == 0)); then
  echo "AGGREGATIONS must contain at least one aggregation." >&2
  exit 2
fi
declare -A seen_aggregations
for aggregation in "${aggregations[@]}"; do
  if [[ "$aggregation" != "fedavg" && "$aggregation" != "median" ]] \
    || [[ -v "seen_aggregations[$aggregation]" ]]; then
    echo "AGGREGATIONS must contain distinct values from: fedavg median." >&2
    exit 2
  fi
  seen_aggregations["$aggregation"]=1
done

run_clean_controls="${RUN_CLEAN_CONTROLS:-true}"
if [[ "$run_clean_controls" != "true" && "$run_clean_controls" != "false" ]]; then
  echo "RUN_CLEAN_CONTROLS must be true or false." >&2
  exit 2
fi
continue_on_failure="${CONTINUE_ON_FAILURE:-false}"
if [[ "$continue_on_failure" != "true" && "$continue_on_failure" != "false" ]]; then
  echo "CONTINUE_ON_FAILURE must be true or false." >&2
  exit 2
fi
poison_num_clients="${POISON_NUM_CLIENTS:-1}"
if [[ ! "$poison_num_clients" =~ ^[1-9][0-9]*$ ]] \
  || ((poison_num_clients > 20)); then
  echo "POISON_NUM_CLIENTS must be an integer from 1 through 20." >&2
  exit 2
fi
poison_fraction="${POISON_FRACTION:-1.0}"
case "$poison_fraction" in
  0.25|0.5|0.75|1.0) ;;
  *)
    echo "POISON_FRACTION must be one of: 0.25, 0.5, 0.75, 1.0." >&2
    exit 2
    ;;
esac

learning_rate="${LEARNING_RATE:-0.1}"
case "$learning_rate" in
  0.1|0.01) ;;
  *)
    echo "LEARNING_RATE must be 0.1 or 0.01." >&2
    exit 2
    ;;
esac

results_root="${RESULTS_DIR:-results/defense}"
models_root="${MODELS_DIR:-models/defense}"
generation_id="${GENERATION_ID:-generation-v1}"
protocol_id="${PROTOCOL_ID:-protocol-v1}"
if [[ ! "$generation_id" =~ ^generation-v[0-9]+$ || ! "$protocol_id" =~ ^protocol-v[0-9]+$ ]]; then
  echo "GENERATION_ID and PROTOCOL_ID must use generation-vN and protocol-vN." >&2
  exit 2
fi
if [[ "$generation_id" == "generation-v1" ]] \
  && { [[ "$learning_rate" == "0.01" && ( "$protocol_id" == "protocol-v1" || "$protocol_id" == "protocol-v2" ) ]] \
       || [[ "$learning_rate" == "0.1" && "$protocol_id" == "protocol-v3" ]]; }; then
  echo "Use a separate protocol ID for each learning rate; generation-v1/protocol-v3 is reserved for 0.01." >&2
  exit 2
fi
execution_id="${EXECUTION_ID:-}"
if [[ -z "$execution_id" ]]; then
  highest_run=0
  for run_path in \
    "$results_root/$generation_id/$protocol_id"/run-* \
    "$models_root/$generation_id/$protocol_id"/run-*; do
    [[ -e "$run_path" ]] || continue
    run_name="${run_path##*/}"
    if [[ "$run_name" =~ ^run-([0-9]+)$ ]]; then
      run_number=$((10#${BASH_REMATCH[1]}))
      if ((run_number > highest_run)); then
        highest_run="$run_number"
      fi
    fi
  done
  execution_id="run-$((highest_run + 1))"
elif [[ ! "$execution_id" =~ ^run-[0-9]+$ ]]; then
  echo "EXECUTION_ID must use the form run-N." >&2
  exit 2
fi

results_dir="$results_root/$generation_id/$protocol_id/$execution_id"
models_dir="$models_root/$generation_id/$protocol_id/$execution_id"
if [[ -e "$results_dir" || -e "$models_dir" ]]; then
  echo "Protocol execution output path already exists: $protocol_id/$execution_id" >&2
  exit 1
fi
if ! command -v flwr >/dev/null 2>&1 || ! command -v python >/dev/null 2>&1; then
  echo "Python and the Flower CLI are required; activate the project environment first." >&2
  exit 127
fi

python -c '
from pytorchexample.shared.data import get_class_names
names = get_class_names("organamnist")
if names[4:6] != ["kidney-left", "kidney-right"]:
    raise SystemExit(f"Unexpected OrganAMNIST class mapping: {names[4:6]}")
'
mkdir -p "$results_dir" "$models_dir"
python - "$results_dir/pilot-manifest.json" "$learning_rate" "${seeds[@]}" <<'PY'
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
from medmnist import INFO

paths = sorted(Path("pytorchexample").rglob("*.py"))
paths += [Path("pyproject.toml"), Path("run_median_defense_pilot.sh")]
record = {
    "description": "OrganAMNIST matched FedAvg and median label-flipping comparison",
    "dataset": "organamnist",
    "dataset_archive_md5": INFO["organamnist"]["MD5"],
    "seeds": [int(seed) for seed in sys.argv[3:]],
    "learning_rate": float(sys.argv[2]),
    "aggregations": os.environ.get("AGGREGATIONS", "fedavg median").split(),
    "includes_clean_controls": os.environ.get("RUN_CLEAN_CONTROLS", "true") == "true",
    "poison_num_clients": int(os.environ.get("POISON_NUM_CLIENTS", "1")),
    "poison_fraction": float(os.environ.get("POISON_FRACTION", "1.0")),
    "continue_on_failure": os.environ.get("CONTINUE_ON_FAILURE", "false") == "true",
    "git_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "git_status": subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=normal"], text=True
    ).splitlines(),
    "source_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    },
    "python": sys.version,
    "platform": platform.platform(),
    "torch_cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "cuda_devices": [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ],
    "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    "dependencies": subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    ).splitlines(),
    "evaluate_test": False,
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
PY

run_condition() {
  local seed=$1 aggregation=$2 poisoned_clients=$3 fraction=$4
  echo "Running seed=$seed aggregation=$aggregation poisoned_clients=$poisoned_clients fraction=$fraction"
  if ! flwr run . \
    --federation-config "num-supernodes=20" \
    --run-config "experiment-type=\"label-flipping\" aggregation=\"$aggregation\" aggregation-subdirectories=true require-complete-round=true dataset=\"organamnist\" seed=$seed num-clients=20 num-server-rounds=5 local-epochs=1 learning-rate=$learning_rate batch-size=32 poison-num-clients=$poisoned_clients poison-fraction=$fraction poison-source-class-id=4 poison-target-class-id=5 evaluate-test=false generation-id=\"$generation_id\" protocol-id=\"$protocol_id\" execution-id=\"$execution_id\" results-dir=\"$results_root\" models-dir=\"$models_root\"" \
    --stream; then
    echo "Flower run failed for seed=$seed aggregation=$aggregation poisoned_clients=$poisoned_clients." >&2
    return 1
  fi

  local fraction_tag
  fraction_tag="$(LC_NUMERIC=C printf "%.3f" "$fraction" | tr '.' 'p')"
  if ! python - "$results_dir" "$seed" "$aggregation" "$poisoned_clients" "$fraction_tag" "$fraction" "$learning_rate" <<'PY'
import json
import sys
from pathlib import Path

root, seed, aggregation, count, fraction_tag, fraction, learning_rate = sys.argv[1:]
prefix = "median_" if aggregation == "median" else ""
rate_tag = f"{float(learning_rate):.3f}".replace(".", "p")
experiment = f"{prefix}label_flip_r5_e1_lr{rate_tag}_clients{count}_f{fraction_tag}_s4_t5"
records = list((Path(root) / aggregation / "organamnist" / experiment).glob(f"seed-{seed}_run-*.json"))
if len(records) != 1:
    raise SystemExit(f"Expected one result for {aggregation}, seed {seed}, clients {count}; found {len(records)}.")
record = json.loads(records[0].read_text())
training = record["training_results"]
configuration = record["configuration"]
label_flipping = configuration["label_flipping"]
if (
    record.get("status") != "completed"
    or not training["valid_run"]
    or not training["metrics_finite"]
    or not training["model_parameters_finite"]
    or training["received_results_by_round"] != {str(i): 20 for i in range(1, 6)}
    or training["client_failures_by_round"] != {str(i): 0 for i in range(1, 6)}
    or record["metrics"]["test_final"] is not None
    or record["aggregation"] != ("FedAvg" if aggregation == "fedavg" else "coordinate-wise median")
    or record.get("artifact_layout") != "aggregation_subdirectories"
    or record.get("aggregation_id") != aggregation
    or record["seed"] != int(seed)
    or configuration["clients"] != 20
    or configuration["federated_rounds"] != 5
    or configuration["local_epochs"] != 1
    or configuration["learning_rate"] != float(learning_rate)
    or configuration["batch_size"] != 32
    or not configuration["require_complete_round"]
    or label_flipping["num_poisoned_clients"] != int(count)
    or label_flipping["fraction_of_source_labels_flipped_per_poisoned_client"] != float(fraction)
    or label_flipping["source_class_id"] != 4
    or label_flipping["target_class_id"] != 5
):
    raise SystemExit(f"Invalid or unexpected pilot result: {records[0]}")
PY
  then
    echo "Result validation failed for seed=$seed aggregation=$aggregation poisoned_clients=$poisoned_clients." >&2
    return 1
  fi
}

failed_conditions=()
runs_per_aggregation=1
if [[ "$run_clean_controls" == "true" ]]; then
  runs_per_aggregation=2
fi
echo "Starting $((${#seeds[@]} * ${#aggregations[@]} * runs_per_aggregation)) validation-only runs under $generation_id/$protocol_id/$execution_id"
for seed in "${seeds[@]}"; do
  for aggregation in "${aggregations[@]}"; do
    if [[ "$run_clean_controls" == "true" ]]; then
      if ! run_condition "$seed" "$aggregation" 0 0.0; then
        if [[ "$continue_on_failure" != "true" ]]; then exit 1; fi
        failed_conditions+=("seed=$seed aggregation=$aggregation clients=0 fraction=0.0")
        echo "Continuing after failed condition: ${failed_conditions[-1]}" >&2
      fi
    fi
    if ! run_condition "$seed" "$aggregation" "$poison_num_clients" "$poison_fraction"; then
      if [[ "$continue_on_failure" != "true" ]]; then exit 1; fi
      failed_conditions+=("seed=$seed aggregation=$aggregation clients=$poison_num_clients fraction=$poison_fraction")
      echo "Continuing after failed condition: ${failed_conditions[-1]}" >&2
    fi
  done
done

if ((${#failed_conditions[@]} > 0)); then
  printf 'Failed conditions:\n' >&2
  printf '  %s\n' "${failed_conditions[@]}" >&2
  exit 1
fi
