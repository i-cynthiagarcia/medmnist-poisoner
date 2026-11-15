#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

datasets=("$@")
if ((${#datasets[@]} == 0)); then
  datasets=(organamnist organcmnist organsmnist)
fi

for dataset in "${datasets[@]}"; do
  case "$dataset" in
    organamnist|organcmnist|organsmnist) ;;
    *)
      echo "Unsupported dataset: $dataset" >&2
      echo "Choose organamnist, organcmnist, or organsmnist." >&2
      exit 2
      ;;
  esac
done

read -r -a seeds <<< "${SEEDS:-42 43 44 45 46}"
if ((${#seeds[@]} == 0)); then
  echo "SEEDS must contain at least one seed." >&2
  exit 2
fi

settings=(
  "3 1 0.001 0p001"
  "3 1 0.01 0p010"
  "3 1 0.1 0p100"
  "3 3 0.001 0p001"
  "5 1 0.001 0p001"
  "5 1 0.1 0p100"
  "5 3 0.1 0p100"
)

results_root="${RESULTS_DIR:-results/baseline}"
models_root="${MODELS_DIR:-models/baseline}"
protocol_id="protocol_$(date -u +%Y%m%dT%H%M%SZ)"
results_dir="$results_root/$protocol_id"
models_dir="$models_root/$protocol_id"

if [[ -e "$results_dir" || -e "$models_dir" ]]; then
  echo "Protocol output path already exists: $protocol_id" >&2
  echo "Wait for a new timestamp or choose different output roots." >&2
  exit 1
fi

if ! command -v flwr >/dev/null 2>&1; then
  echo "Flower CLI 'flwr' is not available in PATH; activate the project environment first." >&2
  exit 127
fi

mkdir -p "$results_dir" "$models_dir"
echo "Starting protocol $protocol_id"

for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    for setting in "${settings[@]}"; do
      read -r rounds epochs lr _ <<< "$setting"
      flwr run . \
        --federation-config "num-supernodes=20" \
        --run-config "dataset=\"$dataset\" seed=$seed num-server-rounds=$rounds local-epochs=$epochs learning-rate=$lr evaluate-test=false protocol-id=\"$protocol_id\" results-dir=\"$results_root\" models-dir=\"$models_root\"" \
        --stream
    done
  done
done
