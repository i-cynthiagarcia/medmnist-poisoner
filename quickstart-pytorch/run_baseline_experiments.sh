#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mode="single_round"
run_centralized=false
datasets=()
while (($#)); do
  case "$1" in
    --full-sweep)
      if [[ "$mode" == "full_sweep" ]]; then
        echo "--full-sweep may be specified only once." >&2
        exit 2
      fi
      mode="full_sweep"
      ;;
    --centralized)
      if [[ "$run_centralized" == true ]]; then
        echo "--centralized may be specified only once." >&2
        exit 2
      fi
      run_centralized=true
      ;;
    --help|-h)
      cat <<'HELP'
Usage: run_baseline_experiments.sh [--full-sweep] [--centralized] [dataset ...]

Without --full-sweep, run one round for each Organ dataset (seed 42,
one local epoch, learning rate 0.001).
With --full-sweep, run the seven-setting, five-seed, three-dataset matrix
(105 Flower runs; each run uses 3 or 5 federated rounds).
With --centralized, also train a centralized full-training-set model for each
selected dataset, seed, and setting. By default, only Flower runs are made.

Set SEEDS to override seeds, GENERATION_ID to select a generation, and
RESULTS_DIR/MODELS_DIR to select category roots.
HELP
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      echo "Use --help for usage." >&2
      exit 2
      ;;
    *)
      datasets+=("$1")
      ;;
  esac
  shift
done

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

if [[ "$mode" == "full_sweep" ]]; then
  default_seeds="42 43 44 45 46"
  settings=(
    "3 1 0.001 0p001"
    "3 1 0.01 0p010"
    "3 1 0.1 0p100"
    "3 3 0.001 0p001"
    "5 1 0.001 0p001"
    "5 1 0.1 0p100"
    "5 3 0.1 0p100"
  )
else
  default_seeds="42"
  settings=("1 1 0.001 0p001")
fi

read -r -a seeds <<< "${SEEDS:-$default_seeds}"
if ((${#seeds[@]} == 0)); then
  echo "SEEDS must contain at least one seed." >&2
  exit 2
fi

results_root="${RESULTS_DIR:-results/baseline}"
models_root="${MODELS_DIR:-models/baseline}"
generation_id="${GENERATION_ID:-generation-v3}"
if [[ ! "$generation_id" =~ ^generation-v[0-9]+$ ]]; then
  echo "GENERATION_ID must use the form generation-vN." >&2
  exit 2
fi
protocol_id="protocol_$(date -u +%Y%m%dT%H%M%SZ)"
results_dir="$results_root/$generation_id/$protocol_id"
models_dir="$models_root/$generation_id/$protocol_id"

if [[ -e "$results_dir" || -e "$models_dir" ]]; then
  echo "Protocol output path already exists: $protocol_id" >&2
  echo "Wait for a new timestamp or choose different output roots." >&2
  exit 1
fi

if ! command -v flwr >/dev/null 2>&1; then
  echo "Flower CLI 'flwr' is not available in PATH; activate the project environment first." >&2
  exit 127
fi
if [[ "$run_centralized" == true ]] && ! command -v python >/dev/null 2>&1; then
  echo "Python is not available in PATH; activate the project environment first." >&2
  exit 127
fi

mkdir -p "$results_dir" "$models_dir"
run_count=$((${#datasets[@]} * ${#seeds[@]} * ${#settings[@]}))
echo "Starting $run_count federated $mode run(s) under $generation_id/$protocol_id"
if [[ "$run_centralized" == true ]]; then
  echo "Also running $run_count centralized control(s)."
fi

for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    for setting in "${settings[@]}"; do
      read -r rounds epochs lr _ <<< "$setting"
      flwr run . \
        --federation-config "num-supernodes=20" \
        --run-config "dataset=\"$dataset\" seed=$seed num-server-rounds=$rounds local-epochs=$epochs learning-rate=$lr evaluate-test=false generation-id=\"$generation_id\" protocol-id=\"$protocol_id\" results-dir=\"$results_root\" models-dir=\"$models_root\"" \
        --stream
      if [[ "$run_centralized" == true ]]; then
        python -m pytorchexample.experiments.baseline.centralized \
          --dataset "$dataset" \
          --seed "$seed" \
          --num-rounds "$rounds" \
          --local-epochs "$epochs" \
          --learning-rate "$lr" \
          --batch-size 32 \
          --generation-id "$generation_id" \
          --protocol-id "$protocol_id" \
          --results-dir "$results_root" \
          --models-dir "$models_root"
      fi
    done
  done
done
