---
tags: [quickstart, vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Federated Learning with PyTorch and Flower (Quickstart Example)

This introductory example to Flower uses PyTorch, but deep knowledge of PyTorch is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case. Running this example in itself is quite easy. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the CIFAR-10 dataset.

## Set up the project

The dependencies and Python version range are declared in `pyproject.toml`;
there is no separate `requirements.txt` to maintain. This project requires
Python `>=3.10,<3.14`; Python 3.12 and 3.13 meet this requirement, while Python
3.14 is not currently supported. From the `quickstart-pytorch/` directory,
create and activate a virtual environment with a supported interpreter, then
install the project and its dependencies in editable mode. The
`flwr[simulation]` dependency provides the `flwr` command used below.
The project-local `.gitignore` excludes `.venv/`, and Flower app bundles
include only the application source, README, and license, so generated
experiment results do not inflate subsequent FABs.

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install -e .
```

On Debian/Ubuntu, if environment creation reports that `ensurepip` is not
available, install the matching venv package first:

```bash
sudo apt update
sudo apt install python3.12-venv
```

Then rerun the `python3.12 -m venv .venv` command.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -e .
```

If you already created a virtual environment, check its version with
`python --version`. If it is Python 3.14, create a new environment with Python
3.12 or 3.13 rather than trying to reuse it. Run
`python -m pip install -e .` from `quickstart-pytorch/`, and use the same
activated environment in the terminal where you run Flower.

The Python package keeps reusable components in
`pytorchexample/shared/` (data, partitions, model, training, metrics, and
artifacts) and experiment entry points in
`pytorchexample/experiments/<experiment>/`. The baseline Flower client/server
apps and centralized runner are grouped in
`pytorchexample/experiments/baseline/`.
The label-flipping Flower apps live in
`pytorchexample/experiments/poisoning/`; reusable Flower handlers are in
`pytorchexample/experiments/shared/`. The project-level Flower app dispatches
between baseline and poisoning using the `experiment-type` run-config value.

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!TIP]
> This example runs faster when the `ClientApp`s have access to a GPU. Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
# Run OrganAMNIST with 20 simulated clients (CPU only)
flwr run . \
  --federation-config "num-supernodes=20" \
  --run-config "dataset=\"organamnist\" seed=42" \
  --stream
```

The supported Organ datasets are `organamnist`, `organcmnist`, and `organsmnist`. Override the dataset and seed without editing source files:

```bash
# OrganCMNIST
flwr run . \
  --federation-config "num-supernodes=20" \
  --run-config "dataset=\"organcmnist\" seed=42" \
  --stream

# OrganSMNIST
flwr run . \
  --federation-config "num-supernodes=20" \
  --run-config "dataset=\"organsmnist\" seed=42" \
  --stream
```

You can also override training settings defined in `pyproject.toml`:

```bash
flwr run . \
  --federation-config "num-supernodes=20" \
  --run-config "dataset=\"organamnist\" seed=42 num-server-rounds=5 local-epochs=3 learning-rate=0.01 batch-size=32" \
  --stream
```

To run one dataset across seeds 42, 43, and 44, use this Bash loop in the terminal:

```bash
for seed in 42 43 44; do
  flwr run . \
    --federation-config "num-supernodes=20" \
    --run-config "dataset=\"organcmnist\" seed=$seed" \
    --stream
done
```

The loop syntax works in Bash and generally in zsh. It stops or continues between separate Flower commands. Each command should be recorded as a separate JSON result using its seed and run ID.

For a quick one-round run on each Organ dataset (seed 42, one local epoch,
learning rate 0.001), run the baseline script without flags:

```bash
bash quickstart-pytorch/run_baseline_experiments.sh
```

To run the full 105-run matrix across all three datasets, explicitly pass
`--full-sweep`:

```bash
bash quickstart-pytorch/run_baseline_experiments.sh --full-sweep
```

The full matrix is seven settings × five seeds (42–46) × three datasets. Each
run trains for 3 or 5 federated rounds, so the matrix has 105 Flower runs and
405 federated training rounds total—not 105 total server rounds. To select a
dataset, append its name, for example
`bash quickstart-pytorch/run_baseline_experiments.sh --full-sweep organcmnist`.
Set `SEEDS` to override the seed list in either mode.
With `--centralized`, the full matrix also runs 105 centralized controls.

To reproduce the current candidate control settings across seeds 42–46, use
`--selected-settings`:

```bash
bash quickstart-pytorch/run_baseline_experiments.sh --selected-settings
```

This runs 15 federated experiments: OrganAMNIST uses 5 rounds, 1 local epoch,
and learning rate 0.1; OrganCMNIST and OrganSMNIST use 5 rounds, 3 local
epochs, and learning rate 0.1. Add `--centralized` to run a matching set of
centralized controls. These are the current candidate settings in
[baseline_results.md](baseline_results.md); update the script's selected
settings when the final controls are confirmed. `SEEDS` and dataset arguments
can also be used to select a subset.

By default, the script runs only federated training: each client model trains
on its uniform partition of the MedMNIST `train` split, and FedAvg produces
the federated aggregate model. Add `--centralized` to also train a separate
centralized control on the complete `train` split for every selected
dataset, seed, and setting:

```bash
bash quickstart-pytorch/run_baseline_experiments.sh --centralized
bash quickstart-pytorch/run_baseline_experiments.sh --full-sweep --centralized
```

The centralized control matches the federated epoch budget
(`federated rounds × local epochs`) and evaluates on `val` at initialization
and after each equivalent round. Test-set evaluation remains disabled. Its
run record identifies it as `centralized_full_train`; federated records
identify the server output as `federated_aggregate` and note that client-local
weights are not persisted. Centralized and federated artifacts have separate
experiment directories. The centralized runner can also be invoked directly
from `quickstart-pytorch/`, for example:

```bash
python -m pytorchexample.experiments.baseline.centralized \
  --dataset organamnist \
  --seed 42 \
  --num-rounds 5 \
  --local-epochs 1 \
  --learning-rate 0.1
```

Per-round centralized metrics use the `val` split; test-set evaluation is
disabled by default.

Each invocation shares a protocol version and execution ID across its runs.
The default protocol version is `protocol-v1`; it identifies the protocol
structure, while `run-1`, `run-2`, and so on distinguish executions of that
same structure. The default generation is `generation-v3`; set `GENERATION_ID`
to select another `generation-vN`. Results are organized by category,
generation, protocol, and execution:
`results/<category>/generation-vN/protocol-vN/run-N/...`. The supported
categories are `baseline`, `poisoning`, and `defense`; model outputs mirror
that layout under `models/`. The default baseline sweep writes beneath
`results/baseline/generation-v3/protocol-v1/run-N/` and
`models/baseline/generation-v3/protocol-v1/run-N/`. The runner automatically
chooses the next run number for that protocol, or you can set `PROTOCOL_ID` and
`EXECUTION_ID` to select them explicitly. Set `RESULTS_DIR` and `MODELS_DIR` to
choose different category roots. A failed or partial execution keeps its
directory; later executions get a new run number rather than replacing those
outputs. Individual files also include the Flower run ID.
Each run records per-round validation loss and accuracy, per-class precision,
recall, F1, support, and a confusion matrix (rows are true classes; columns are
predicted classes). Checkpoints for each aggregated server round are saved under
`models/baseline/generation-v3/protocol-v1/run-N/<dataset>/<experiment>/checkpoints/seed-<seed>_run-<run-id>/`.
Before training, each run verifies that client partitions are disjoint and
assign every training example exactly once. It saves a
`partition-seed-<seed>_run-<run-id>.json` manifest beside the run result with
each client's image count and class counts; the run JSON links to that
manifest.
Annotated PNG confusion matrices are saved under each results experiment's
`plots/seed-<seed>_run-<run-id>/` directory, and their paths are included in the
JSON record. Generation-v2 runs can be plotted from their recomputed final
validation matrices, without requiring training checkpoints:

```bash
python render_confusion_matrices.py \
  --results-dir results/baseline/generation-v2
```

The renderer refuses to replace existing plots and records their paths in each
JSON file. Re-running it is safe when all referenced plots are present.
After the full sweep finishes, validate and summarize its records. By default,
the Markdown report is printed to the terminal:

```bash
python summarize_baseline_sweep.py
```

By default, the summarizer uses the highest `run-N` under
`results/baseline/generation-v3/protocol-v1/`. It expects a complete sweep, so
if the newest execution is a targeted rerun, pass `--execution-id run-N` for
the complete sweep you want to summarize. Pass `--seeds` if the sweep used a
custom seed list. Save the report under the execution's results directory with
`--output`:

```bash
python summarize_baseline_sweep.py \
  --execution-id run-4 \
  --output results/baseline/generation-v3/protocol-v1/run-4/baseline_summary.md
```

The summarizer requires the complete seven-setting matrix across all three
datasets. It refuses to summarize if federated runs failed, records are
missing, client failures occurred, or partition manifests did not pass
verification. If the sweep included `--centralized`, its comparison table is
generated only when all centralized records have finite metrics; otherwise the
table is withheld and the affected runs are listed in the report. The script
prints to stdout or writes the requested report file; it never edits
`baseline_results.md`.

To reproducibly recompute final validation metrics from saved final model
checkpoints, run `python recompute_validation_metrics.py --verify` from
`quickstart-pytorch/`. Use `--in-place` to add or refresh the metrics in the
input run records, or `--output-dir <new-directory>` to write updated copies
without changing the originals. This only re-evaluates final checkpoints; it
cannot recreate per-round metrics when intermediate checkpoints are absent.

### Run Targeted Label-Flipping Experiments

The label-flipping runner varies the number of poisoned clients and the
fraction of each poisoned client's source-class training labels changed to the
target class. It includes a matched zero-poison control for each seed. Client
IDs are selected uniformly without replacement from the run seed; examples
are selected deterministically within each selected client's source-class
partition. Fractions are rounded to the nearest whole training example.
Validation and test labels are never changed.

First get the class IDs from the installed MedMNIST metadata. This avoids
assuming a fixed numeric mapping:

```bash
python -c 'from pytorchexample.shared.data import get_class_names; [print(f"{i}: {name}") for i, name in enumerate(get_class_names("organamnist"))]'
```

Use the displayed IDs for the source and target classes in the experiment. For
the initial trial recorded in
[poisoning_results.md](poisoning_results.md), these are kidney-left (4) and
kidney-right (5). The runner checks and prints the resolved names before
training:

```bash
bash run_label_flipping_experiments.sh organamnist "$SOURCE_CLASS_ID" "$TARGET_CLASS_ID"
```

Set `SOURCE_CLASS_ID` and `TARGET_CLASS_ID` to the IDs printed above.

By default this runs one clean control plus 16 label-flipping settings
(`1, 2, 4, 8` poisoned clients × `0.1, 0.25, 0.5, 1.0` source-label fractions)
for each seed `42–46`, using 5 rounds, 1 local epoch, and learning rate 0.1.
Override the matrix and training settings with environment variables, for
example:

```bash
SEEDS="42 43" POISONED_CLIENT_COUNTS="1 4" POISON_FRACTIONS="0.25 1.0" \
NUM_SERVER_ROUNDS=5 LOCAL_EPOCHS=1 LEARNING_RATE=0.1 \
bash run_label_flipping_experiments.sh organamnist "$SOURCE_CLASS_ID" "$TARGET_CLASS_ID"
```

The runner writes into `results/poisoning/` and `models/poisoning/` by default.
Set `RESULTS_DIR`, `MODELS_DIR`, `GENERATION_ID`, and `PROTOCOL_ID` to change
the output locations/version. Each run records the selected client IDs,
source/target class IDs and resolved names, poisoning fraction, and targeted
source-to-target misclassification rate on validation (and test, if explicitly
enabled in a final evaluation run). It also records the exact number of
source-class training examples flipped for each selected client. The targeted
rate is the fraction of clean source-class evaluation
images predicted as the target class; overall accuracy and the full confusion
matrix remain available alongside it. Compare poisoning runs against the
clean baseline with the same seed, client count, partition, and training
settings. Treat monotonic increases in targeted error as a hypothesis, not an
assumption.

The server evaluates on the centralized validation split at initialization and
after each round. Test-set evaluation is disabled by default while selecting
training settings. Once the protocol is fixed, set `evaluate-test=true` to
evaluate the final model once on the official test split; those metrics are
recorded separately from per-round validation metrics. Do not use test metrics
to select training settings.

By default, a direct Flower run uses `generation-v3/protocol-v1` and
automatically selects the next available `run-N` beneath `results/baseline/`
and `models/baseline/`. Reuse the protocol version when the protocol structure
is unchanged; select a new `protocol-vN` when it changes. A final-test run can
use the same protocol version because its execution ID keeps it separate:

```bash
flwr run . \
  --federation-config "num-supernodes=20" \
  --run-config "dataset=\"organamnist\" seed=42 evaluate-test=true generation-id=\"generation-v3\" protocol-id=\"protocol-v1\" execution-id=\"auto\" results-dir=\"results/baseline\" models-dir=\"models/baseline\"" \
  --stream
```

### Check Client Data Distribution

Use `check_distribution.py` to inspect the uniform client partition independently of a Flower run. It uses the same seeded permutation and interleaved partitioning as `load_data()` in `pytorchexample/shared/data.py`.

```bash
python check_distribution.py \
  --dataset organcmnist \
  --num-clients 20 \
  --seed 42 \
  --size 28
```

Supported datasets are `organamnist`, `organcmnist`, and `organsmnist`. The defaults are dataset `organamnist`, 20 clients, seed 42, and image size 28. View all options with:

```bash
python check_distribution.py --help
```

> [!TIP]
> For a more detailed walk-through check our [quickstart PyTorch tutorial](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
