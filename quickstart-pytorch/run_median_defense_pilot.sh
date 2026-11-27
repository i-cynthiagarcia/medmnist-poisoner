#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if (($# != 0)); then
  echo "Usage: $0 (optionally set SEEDS, GENERATION_ID, PROTOCOL_ID, EXECUTION_ID, RESULTS_DIR, MODELS_DIR)" >&2
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

results_root="${RESULTS_DIR:-results/defense}"
models_root="${MODELS_DIR:-models/defense}"
generation_id="${GENERATION_ID:-generation-v1}"
protocol_id="${PROTOCOL_ID:-protocol-v1}"
if [[ ! "$generation_id" =~ ^generation-v[0-9]+$ || ! "$protocol_id" =~ ^protocol-v[0-9]+$ ]]; then
  echo "GENERATION_ID and PROTOCOL_ID must use generation-vN and protocol-vN." >&2
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
python - "$results_dir/pilot-manifest.json" "${seeds[@]}" <<'PY'
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
    "description": "OrganAMNIST coordinate-wise median label-flipping pilot",
    "dataset": "organamnist",
    "dataset_archive_md5": INFO["organamnist"]["MD5"],
    "seeds": [int(seed) for seed in sys.argv[2:]],
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

echo "Starting $((${#seeds[@]} * 4)) validation-only pilot runs under $generation_id/$protocol_id/$execution_id"
for seed in "${seeds[@]}"; do
  for aggregation in fedavg median; do
    for poisoned_clients in 0 1; do
      fraction=0.0
      if ((poisoned_clients == 1)); then fraction=1.0; fi
      flwr run . \
        --federation-config "num-supernodes=20" \
        --run-config "experiment-type=\"label-flipping\" aggregation=\"$aggregation\" aggregation-subdirectories=true require-complete-round=true dataset=\"organamnist\" seed=$seed num-clients=20 num-server-rounds=5 local-epochs=1 learning-rate=0.1 batch-size=32 poison-num-clients=$poisoned_clients poison-fraction=$fraction poison-source-class-id=4 poison-target-class-id=5 evaluate-test=false generation-id=\"$generation_id\" protocol-id=\"$protocol_id\" execution-id=\"$execution_id\" results-dir=\"$results_root\" models-dir=\"$models_root\"" \
        --stream
      python - "$results_dir" "$seed" "$aggregation" "$poisoned_clients" <<'PY'
import json
import sys
from pathlib import Path

root, seed, aggregation, count = sys.argv[1:]
fraction = "0p000" if count == "0" else "1p000"
prefix = "median_" if aggregation == "median" else ""
experiment = f"{prefix}label_flip_r5_e1_lr0p100_clients{count}_f{fraction}_s4_t5"
records = list((Path(root) / aggregation / "organamnist" / experiment).glob(f"seed-{seed}_run-*.json"))
if len(records) != 1:
    raise SystemExit(f"Expected one result for {aggregation}, seed {seed}, clients {count}; found {len(records)}.")
record = json.loads(records[0].read_text())
if (
    not record["training_results"]["valid_run"]
    or record["metrics"]["test_final"] is not None
    or record["aggregation"] != ("FedAvg" if aggregation == "fedavg" else "coordinate-wise median")
    or record.get("artifact_layout") != "aggregation_subdirectories"
    or record.get("aggregation_id") != aggregation
    or not record["configuration"]["require_complete_round"]
):
    raise SystemExit(f"Invalid or unexpected pilot result: {records[0]}")
PY
    done
  done
done
