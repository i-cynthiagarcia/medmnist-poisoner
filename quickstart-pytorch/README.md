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

The Python package keeps reusable data, model, training, metrics, and artifact
components in `pytorchexample/shared/`. The Flower components configured in
`pyproject.toml` are the client and server dispatchers directly in
`pytorchexample/experiments/`; they select baseline or label-flipping behavior
using `experiment-type`. Reusable Flower handlers and FedAvg reply validation
live in `pytorchexample/experiments/shared/`. Label-flipping logic lives in
`pytorchexample/experiments/poisoning/`, and median aggregation lives in
`pytorchexample/experiments/defense/`. The `aggregation` setting selects the
server strategy independently of attack type. The baseline directory contains
the centralized runner.

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
that layout under `models/`. Poisoning results and models additionally use
an attack-family directory (`flipping` or `overlay`) before the generation.
Generation, protocol, and run IDs are scoped to that family. The default
baseline sweep writes beneath
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

The runner writes into `results/poisoning/flipping/` and
`models/poisoning/flipping/` by default.
Set `RESULTS_DIR`, `MODELS_DIR`, `GENERATION_ID`, and `PROTOCOL_ID` to change
the output locations/version. These variables are complete family roots; do
not append the family a second time. When invoking Flower directly with
`experiment-type="label-flipping"`, explicitly set `results-dir` and
`models-dir` to the flipping roots: the project-wide Flower defaults remain
the baseline roots. Existing records were migrated to these directories
without changing their scientific protocol; see the
[results index](results/README.md). The overlay directory is reserved for
future experiments and does not imply an implemented overlay protocol.
Each run records the selected client IDs,
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

### OrganAMNIST median defense pilot

From `quickstart-pytorch/`, run the validation-only pilot with:

```bash
bash run_median_defense_pilot.sh
```

The pilot runs seeds 42–46, each with four matched conditions: clean and
targeted label-flipping under FedAvg and under coordinate-wise median. It
uses 20 clients, five rounds, one local epoch, learning rate 0.1, and batch
size 32. The attack flips all kidney-left (class 4) labels on one
seed-selected client to kidney-right (class 5), in every round. Seed, model
initialization, partition, and attack selection are matched across conditions.
`SEEDS="42 43"` selects a smaller pilot. Results and models default to
`results/defense/generation-v1/protocol-v1/run-N/` and the matching `models/`
directory; `GENERATION_ID`, `PROTOCOL_ID`, `EXECUTION_ID`, `RESULTS_DIR`, and
`MODELS_DIR` override those values. An existing execution is never overwritten.
Within each execution, `fedavg/` and `median/` contain their respective
matched condition outputs, under each dataset and experiment. Direct Flower
runs can opt into this layout with `aggregation-subdirectories=true`; the
default remains the existing ungrouped layout for baseline and poisoning runs.
The execution manifest records source hashes, Git state, and the Python and
dependency environment. The run records retain partitions, checkpoints,
validation metrics, client counts, and attack details.

For an exploratory four-client, 100%-flip median comparison across seeds
42–51, reuse the existing valid FedAvg attack/control pairs and median clean
controls, and run only the new median poisoned conditions:

```bash
PROTOCOL_ID=protocol-v2 \
SEEDS="42 43 44 45 46 47 48 49 50 51" \
AGGREGATIONS=median \
RUN_CLEAN_CONTROLS=false \
POISON_NUM_CLIENTS=4 \
POISON_FRACTION=1.0 \
CONTINUE_ON_FAILURE=true \
bash run_median_defense_pilot.sh
```

This writes a new execution under
`results/defense/generation-v1/protocol-v2/run-N/` and the matching `models/`
directory. The reused FedAvg four-client/100%-flip records and their clean
controls are in
`results/poisoning/flipping/generation-v3/protocol-v3/run-1/organamnist/`;
their recorded settings and validity checks match, but their source and
environment were not captured, so treat cross-protocol findings as
exploratory. Existing median clean controls are valid for seeds 42–48 and
50–51; seed 49 has no valid median-clean result and must be excluded from
matched summaries; see the [seed-49 failure investigation and disposition](defense_results.md#seed-49-median-clean-failure-and-disposition).
The initial median pilot's seeds 43–46 also predate the current source hashes.
`CONTINUE_ON_FAILURE=true` attempts later seeds after a failed condition and
exits nonzero with the failed conditions listed.

The three invalid OrganAMNIST protocol-v3 cells at four poisoned clients and
25–50% flips remain documented and excluded from completed-pair analyses; see
[poisoning_results.md](poisoning_results.md).

The protocol-v2 median attack runs completed for all ten seeds. On the nine
seeds with median-clean controls, the mean poison-minus-clean changes were:
targeted error +5.05 pp, kidney-left recall -14.55 pp, overall accuracy
-2.06 pp, and validation loss +0.0738. The corresponding FedAvg changes were
+23.26 pp, -52.17 pp, -2.80 pp, and +0.0220. These are validation-only
exploratory results: seed 49 lacks a completed median-clean control, seed 42's
two valid median-clean records have identical round-5 metrics and are counted
once, and the per-seed outcomes vary. In particular, do not infer general
robustness from averages alone. Median-clean minus FedAvg-clean mean
differences were +0.33 pp targeted error, +1.08 pp kidney-left recall,
+1.77 pp accuracy, and -0.0663 loss. The detailed per-seed clean differences,
within-method attack changes, and median-minus-FedAvg attack-effect
differences are in [defense_results.md](defense_results.md).

#### 0.01 OrganAMNIST comparison

The 12-run validation-only pilot at learning rate 0.01 completed for seeds 42,
49, and 50 under all four matched conditions. Seed 49's median-clean run
completed, but three selected seeds do not prove stability. The results and
integrity checks are in [defense_results.md](defense_results.md#001-learning-rate-pilot);
they do not replace the earlier nine-seed evidence at 0.1.

The [frozen OrganAMNIST comparison protocol](defense_results.md#frozen-organamnist-comparison-protocol)
uses learning rate 0.01 and seeds 42–51 across all four conditions, with a
75% mean clean validation-accuracy floor per aggregation and paired targeted
error as the primary metric. It specifies failure reporting and infrastructure
retries. The same seed set is proposed for OrganCMNIST and OrganSMNIST, which
need separate stability and clean-performance checks before adopting settings.

Before any new execution, commit the intended source and check
`git status --short` and `git rev-parse HEAD`. Use the same committed source
and environment for every condition; confirm `command -v python` and
`command -v flwr` point to the same activated environment, and do not edit the
source or dependencies during an execution. The manifest records the revision,
working-tree status, source hashes and
`pip freeze`; inspect these alongside the results. The script creates its
output directories before recording status, so new output directories may
appear as untracked. The completed pilot was run with the following command;
it is retained for reproducibility, not as a request to repeat the pilot.
Capture the console log for any new execution, including failures:

```bash
set -o pipefail
log=$(mktemp "$HOME/organa-lr001-pilot.XXXXXX.log")
GENERATION_ID=generation-v1 PROTOCOL_ID=protocol-v3 EXECUTION_ID="" \
RESULTS_DIR=results/defense MODELS_DIR=models/defense \
SEEDS="42 49 50" AGGREGATIONS="fedavg median" RUN_CLEAN_CONTROLS=true \
LEARNING_RATE=0.01 POISON_NUM_CLIENTS=4 POISON_FRACTION=1.0 \
CONTINUE_ON_FAILURE=true bash run_median_defense_pilot.sh 2>&1 | tee "$log"
status=$?
echo "Pilot log: $log"
echo "Pilot exit status: $status"
```

Protocol-v3 is separate from the 0.1-rate protocols; its results must not be
pooled with them. `LEARNING_RATE` accepts `0.1` (default) or `0.01`. Within
generation-v1, the runner rejects 0.01 under protocol-v1/v2 and reserves
protocol-v3 for 0.01. Each condition must complete five rounds with all 20
finite replies, no client failures and finite validation results; the runner
reports failures and exits nonzero after attempting the remaining conditions.
Inspect the saved per-seed validation metrics and checkpoints too. A successful
12-run pilot is only preliminary evidence of stability, not proof.

After committing the intended source and documentation, the following
validation comparison starts a new execution and runs all 40 conditions from
scratch, rather than reusing pilot or historical runs. Verify the committed source and
environment again, retain its manifest and log, and report any failed
conditions rather than treating them as pairs. Compare the final manifest's
revision, source hashes, dependency versions and environment settings with
the pilot manifest; a documentation-only commit changes the revision without
changing the recorded training-source hashes.

```bash
set -o pipefail
log=$(mktemp "$HOME/organa-lr001-final.XXXXXX.log")
GENERATION_ID=generation-v1 PROTOCOL_ID=protocol-v3 EXECUTION_ID="" \
RESULTS_DIR=results/defense MODELS_DIR=models/defense \
SEEDS="42 43 44 45 46 47 48 49 50 51" \
AGGREGATIONS="fedavg median" RUN_CLEAN_CONTROLS=true \
LEARNING_RATE=0.01 POISON_NUM_CLIENTS=4 POISON_FRACTION=1.0 \
CONTINUE_ON_FAILURE=true bash run_median_defense_pilot.sh 2>&1 | tee "$log"
status=$?
echo "Final log: $log"
echo "Final exit status: $status"
```

If a different setting is chosen, do not run this command as written;
evaluate the revised candidate under a new protocol before freezing it. Keep
test evaluation disabled until a separate test-evaluation decision is made.

Select `aggregation="median"` in a direct Flower run to use the defense with
either baseline or label-flipping training; the default is `fedavg`. For each
floating-point model coordinate, median sorts the client values and returns
the middle value for an odd number of clients or the arithmetic mean of the
two middle values for an even number. Thus for 20 clients it averages the
10th and 11th ordered values. It weights clients equally, unlike FedAvg's
training-example weighting, and is not guaranteed to select an actual client
model. Median requires all configured clients to return finite, compatible
model arrays; failed or missing replies abort the round. The pilot also
requires complete, finite replies for its FedAvg controls. Client training
metrics remain example-weighted for both strategies; this does not change
the equal-client weighting of median model aggregation.

Compare the poison-minus-clean change within each aggregation rule, along
with the clean performance cost of median. Inspect targeted validation error,
source-class recall, accuracy, loss, and per-seed values. These few seeds and
reused validation data support exploratory conclusions only; neither the
pilot nor its configuration selects or evaluates against the test split.
The completed five-seed pilot and its validation outcomes are summarized in
[defense_results.md](defense_results.md).

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
