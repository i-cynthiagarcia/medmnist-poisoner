#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if (($# != 3)); then
  echo "Usage: $0 <organ-dataset> <source-class-id> <target-class-id>" >&2
  echo "List current class IDs with the Python snippet in README.md." >&2
  exit 2
fi

dataset="$1"
source_class_id="$2"
target_class_id="$3"
case "$dataset" in
  organamnist|organcmnist|organsmnist) ;;
  *)
    echo "Unsupported dataset: $dataset" >&2
    echo "Choose organamnist, organcmnist, or organsmnist." >&2
    exit 2
    ;;
esac
for class_id in "$source_class_id" "$target_class_id"; do
  if [[ ! "$class_id" =~ ^[0-9]+$ ]]; then
    echo "Class IDs must be non-negative integers." >&2
    exit 2
  fi
done
if ! command -v python >/dev/null 2>&1; then
  echo "Python is not available in PATH." >&2
  exit 127
fi

python -c '
import sys
from pytorchexample.shared.data import get_class_names

dataset, source, target = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
names = get_class_names(dataset)
if source >= len(names) or target >= len(names):
    raise SystemExit(f"Class IDs must be in the range 0..{len(names) - 1}.")
if source == target:
    raise SystemExit("Source and target class IDs must differ.")
print(f"Dataset: {dataset}")
print(f"Targeted flip: {source} ({names[source]}) -> {target} ({names[target]})")
' "$dataset" "$source_class_id" "$target_class_id"

if ! command -v flwr >/dev/null 2>&1; then
  echo "Flower CLI 'flwr' is not available in PATH; activate the project environment first." >&2
  exit 127
fi

num_clients="${NUM_CLIENTS:-20}"
num_rounds="${NUM_SERVER_ROUNDS:-5}"
local_epochs="${LOCAL_EPOCHS:-1}"
learning_rate="${LEARNING_RATE:-0.1}"
batch_size="${BATCH_SIZE:-32}"
read -r -a seeds <<< "${SEEDS:-42 43 44 45 46}"
read -r -a poisoned_client_counts <<< "${POISONED_CLIENT_COUNTS:-1 2 4 8}"
read -r -a flip_fractions <<< "${POISON_FRACTIONS:-0.1 0.25 0.5 1.0}"

if [[ ! "$num_clients" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_CLIENTS must be a positive integer." >&2
  exit 2
fi
if [[ ! "$num_rounds" =~ ^[1-9][0-9]*$ || ! "$local_epochs" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SERVER_ROUNDS and LOCAL_EPOCHS must be positive integers." >&2
  exit 2
fi
for count in "${poisoned_client_counts[@]}"; do
  if [[ ! "$count" =~ ^[1-9][0-9]*$ ]] || ((count > num_clients)); then
    echo "Each POISONED_CLIENT_COUNTS value must be in 1..NUM_CLIENTS." >&2
    exit 2
  fi
done
for seed in "${seeds[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "SEEDS values must be non-negative integers." >&2
    exit 2
  fi
done
python - "${flip_fractions[@]}" <<'PY'
import math
import sys

for value in sys.argv[1:]:
    try:
        fraction = float(value)
    except ValueError as error:
        raise SystemExit(f"Invalid poison fraction: {value}") from error
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise SystemExit("POISON_FRACTIONS values must be greater than 0 and at most 1.")
PY

results_root="${RESULTS_DIR:-results/poisoning}"
models_root="${MODELS_DIR:-models/poisoning}"
generation_id="${GENERATION_ID:-generation-v3}"
protocol_id="${PROTOCOL_ID:-protocol-v1}"
if [[ ! "$generation_id" =~ ^generation-v[0-9]+$ ]]; then
  echo "GENERATION_ID must use the form generation-vN." >&2
  exit 2
fi
if [[ ! "$protocol_id" =~ ^protocol-v[0-9]+$ ]]; then
  echo "PROTOCOL_ID must use the form protocol-vN." >&2
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
  echo "Choose a new EXECUTION_ID or different output roots." >&2
  exit 1
fi

mkdir -p "$results_dir" "$models_dir"
experiment_count=$((${#seeds[@]} * (1 + ${#poisoned_client_counts[@]} * ${#flip_fractions[@]})))
echo "Starting $experiment_count label-flipping runs under $generation_id/$protocol_id/$execution_id"

run_experiment() {
  local seed="$1"
  local poisoned_clients="$2"
  local fraction="$3"
  flwr run . \
    --federation-config "num-supernodes=$num_clients" \
    --run-config "experiment-type=\"label-flipping\" dataset=\"$dataset\" seed=$seed num-clients=$num_clients num-server-rounds=$num_rounds local-epochs=$local_epochs learning-rate=$learning_rate batch-size=$batch_size poison-num-clients=$poisoned_clients poison-fraction=$fraction poison-source-class-id=$source_class_id poison-target-class-id=$target_class_id evaluate-test=false generation-id=\"$generation_id\" protocol-id=\"$protocol_id\" execution-id=\"$execution_id\" results-dir=\"$results_root\" models-dir=\"$models_root\"" \
    --stream
}

for seed in "${seeds[@]}"; do
  run_experiment "$seed" 0 0
  for poisoned_clients in "${poisoned_client_counts[@]}"; do
    for fraction in "${flip_fractions[@]}"; do
      run_experiment "$seed" "$poisoned_clients" "$fraction"
    done
  done
done
