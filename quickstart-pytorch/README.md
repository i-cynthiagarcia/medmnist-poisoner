---
tags: [quickstart, vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Federated Learning with PyTorch and Flower (Quickstart Example)

This introductory example to Flower uses PyTorch, but deep knowledge of PyTorch is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case. Running this example in itself is quite easy. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the CIFAR-10 dataset.

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/quickstart-pytorch
```

This will create a new directory called `quickstart-pytorch` with the following structure:

```shell
quickstart-pytorch
├── pytorchexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `pytorchexample` package.

```bash
pip install -e .
```

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
