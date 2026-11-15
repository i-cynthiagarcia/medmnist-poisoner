#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mode="single_round"
run_centralized=false
datasets=()
while (($#)); do
  case "$1" in
    --full-sweep)
      if [[ "$mode" != "single_round" ]]; then
        echo "--full-sweep and --selected-settings are mutually exclusive." >&2
        exit 2
      fi
      mode="full_sweep"
      ;;
    --selected-settings)
      if [[ "$mode" != "single_round" ]]; then
        echo "--full-sweep and --selected-settings are mutually exclusive." >&2
        exit 2
      fi
      mode="selected_settings"
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
Usage: run_baseline_experiments.sh [--full-sweep | --selected-settings] [--centralized] [dataset ...]

Without --full-sweep, run one round for each Organ dataset (seed 42,
one local epoch, learning rate 0.001).
With --full-sweep, run the seven-setting, five-seed, three-dataset matrix
(105 Flower runs; each run uses 3 or 5 federated rounds).
With --selected-settings, run the current candidate control setting for each
selected dataset across seeds 42-46 (15 Flower runs for all datasets).
The settings are OrganAMNIST: 5 rounds/1 epoch/0.1; OrganCMNIST and
OrganSMNIST: 5 rounds/3 epochs/0.1.
With --centralized, also train a centralized full-training-set model for each
selected dataset, seed, and setting. By default, only Flower runs are made.

Set SEEDS to override seeds, GENERATION_ID/PROTOCOL_ID to select the result
versions, EXECUTION_ID to choose a run number, and RESULTS_DIR/MODELS_DIR to
select category roots. By default, the next run number is chosen automatically.
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
  setting_count="${#settings[@]}"
elif [[ "$mode" == "selected_settings" ]]; then
  default_seeds="42 43 44 45 46"
  settings=()
  setting_count=1
else
  default_seeds="42"
  settings=("1 1 0.001 0p001")
  setting_count="${#settings[@]}"
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
protocol_id="${PROTOCOL_ID:-protocol-v1}"
if [[ ! "$protocol_id" =~ ^protocol-v[0-9]+$ ]]; then
  echo "PROTOCOL_ID must use the form protocol-vN." >&2
  exit 2
fi
results_protocol_dir="$results_root/$generation_id/$protocol_id"
models_protocol_dir="$models_root/$generation_id/$protocol_id"
execution_id="${EXECUTION_ID:-}"
if [[ -z "$execution_id" ]]; then
  highest_run=0
  for run_path in "$results_protocol_dir"/run-* "$models_protocol_dir"/run-*; do
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
results_dir="$results_protocol_dir/$execution_id"
models_dir="$models_protocol_dir/$execution_id"

if [[ -e "$results_dir" || -e "$models_dir" ]]; then
  echo "Protocol execution output path already exists: $protocol_id/$execution_id" >&2
  echo "Choose a new EXECUTION_ID or different output roots." >&2
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
run_count=$((${#datasets[@]} * ${#seeds[@]} * setting_count))
mode_label="${mode//_/-}"
echo "Starting $run_count federated $mode_label run(s) under $generation_id/$protocol_id/$execution_id"
if [[ "$run_centralized" == true ]]; then
  echo "Also running $run_count centralized control(s)."
fi

for dataset in "${datasets[@]}"; do
  if [[ "$mode" == "selected_settings" ]]; then
    case "$dataset" in
      organamnist) dataset_settings=("5 1 0.1 0p100") ;;
      organcmnist|organsmnist) dataset_settings=("5 3 0.1 0p100") ;;
    esac
  else
    dataset_settings=("${settings[@]}")
  fi
  for seed in "${seeds[@]}"; do
    for setting in "${dataset_settings[@]}"; do
      read -r rounds epochs lr _ <<< "$setting"
      flwr run . \
        --federation-config "num-supernodes=20" \
        --run-config "dataset=\"$dataset\" seed=$seed num-server-rounds=$rounds local-epochs=$epochs learning-rate=$lr evaluate-test=false generation-id=\"$generation_id\" protocol-id=\"$protocol_id\" execution-id=\"$execution_id\" results-dir=\"$results_root\" models-dir=\"$models_root\"" \
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
          --execution-id "$execution_id" \
          --results-dir "$results_root" \
          --models-dir "$models_root"
      fi
    done
  done
done
