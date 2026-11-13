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
The project-local `.gitignore` excludes `.venv/` from Flower app bundles.

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

To rerun the recorded parameter sweep using validation metrics, use the
reusable Bash script from the repository root:

```bash
bash quickstart-pytorch/run_validation_sweep.sh
```

By default, it runs the seven recorded settings for all three Organ datasets
with seeds 42, 43, 44, 45, and 46 (105 runs). To run one dataset at a time,
pass its name, for example
`bash quickstart-pytorch/run_validation_sweep.sh organcmnist`. Set `SEEDS` to
override the seed list, for example
`SEEDS="42 43 44" bash quickstart-pytorch/run_validation_sweep.sh` for a 63-run
three-seed sweep. The output roots default to `results/validation-v2/` and
`models/validation-v2/`; set `RESULTS_DIR` and `MODELS_DIR` to use different
roots. The script checks for existing outputs before starting and refuses to
overwrite them.
The completed five-seed results and the clean training settings selected for
follow-up comparisons are documented in
[baseline_results.md](baseline_results.md).

The server evaluates on the centralized validation split at initialization and
after each round. Test-set evaluation is disabled by default while selecting
training settings. Once the protocol is fixed, set `evaluate-test=true` to
evaluate the final model once on the official test split; those metrics are
recorded separately from per-round validation metrics. Do not use test metrics
to select training settings.

By default, validation-run JSON files are saved under
`results/validation-v2/`, and model checkpoints under `models/validation-v2/`.
For a final-test run, use separate output paths to preserve the validation-only
records, for example:

```bash
flwr run . \
  --federation-config "num-supernodes=20" \
  --run-config "dataset=\"organamnist\" seed=42 evaluate-test=true results-dir=\"results/final-test-v2\" models-dir=\"models/final-test-v2\"" \
  --stream
```

### Check Client Data Distribution

Use `check_distribution.py` to inspect the uniform client partition independently of a Flower run. It uses the same seeded permutation and interleaved partitioning as `load_data()` in `pytorchexample/task.py`.

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
