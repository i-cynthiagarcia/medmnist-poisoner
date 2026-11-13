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

results_dir="${RESULTS_DIR:-results/validation-v2}"
models_dir="${MODELS_DIR:-models/validation-v2}"

for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    for setting in "${settings[@]}"; do
      read -r rounds epochs lr lr_label <<< "$setting"
      experiment="baseline_r${rounds}_e${epochs}_lr${lr_label}"
      result_file="$results_dir/$dataset/$experiment/seed-$seed.json"
      model_file="$models_dir/$dataset/$experiment/seed-$seed.pt"
      if [[ -e "$result_file" || -e "$model_file" ]]; then
        echo "Refusing to overwrite an existing output for $dataset, seed $seed, $experiment." >&2
        echo "Use new RESULTS_DIR and MODELS_DIR values to start another sweep." >&2
        exit 1
      fi
    done
  done
done

if ! command -v flwr >/dev/null 2>&1; then
  echo "Flower CLI 'flwr' is not available in PATH; activate the project environment first." >&2
  exit 127
fi

for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    for setting in "${settings[@]}"; do
      read -r rounds epochs lr _ <<< "$setting"
      flwr run . \
        --federation-config "num-supernodes=20" \
        --run-config "dataset=\"$dataset\" seed=$seed num-server-rounds=$rounds local-epochs=$epochs learning-rate=$lr evaluate-test=false results-dir=\"$results_dir\" models-dir=\"$models_dir\"" \
        --stream
    done
  done
done
