"""pytorchexample: A Flower / PyTorch app."""

import json
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import Net, load_centralized_dataset, set_dataset, set_seed, test

# Create ServerApp
app = ServerApp()


class VerifiedFedAvg(FedAvg):
    """FedAvg variant that records per-round client reply counts."""

    def __init__(self, num_clients: int, **kwargs):
        super().__init__(**kwargs)
        self.num_clients = num_clients
        self.received_by_round = {}
        self.failed_by_round = {}

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        received = sum(1 for reply in replies if not reply.has_error())
        failed = sum(1 for reply in replies if reply.has_error())
        self.received_by_round[server_round] = received
        self.failed_by_round[server_round] = failed
        return super().aggregate_train(server_round, replies)


def json_value(value):
    """Convert Flower metric values to JSON-compatible primitives."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_run_result(context: Context, result, num_rounds: int, strategy: VerifiedFedAvg) -> Path:
    """Save centralized and training metrics for one completed Flower run."""
    dataset = context.run_config["dataset"]
    seed = context.run_config["seed"]
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    lr_label = f"lr{learning_rate:.3f}".replace(".", "p")
    experiment = f"baseline_r{num_rounds}_e{local_epochs}_{lr_label}"

    server_metrics = {}
    for server_round, metric_record in result.evaluate_metrics_serverapp.items():
        server_metrics[str(server_round)] = {
            key: json_value(value) for key, value in metric_record.items()
        }

    client_train_metrics = {}
    for server_round, metric_record in result.train_metrics_clientapp.items():
        client_train_metrics[str(server_round)] = {
            key: json_value(value) for key, value in metric_record.items()
        }

    received_by_round = {
        str(round_num): strategy.received_by_round.get(round_num)
        for round_num in range(1, num_rounds + 1)
    }
    failed_by_round = {
        str(round_num): strategy.failed_by_round.get(round_num, 0)
        for round_num in range(1, num_rounds + 1)
    }
    num_clients = context.run_config["num-clients"]
    valid_run = all(
        strategy.received_by_round.get(round_num) == num_clients
        and strategy.failed_by_round.get(round_num, 0) == 0
        for round_num in range(1, num_rounds + 1)
    )

    record = {
        "run_id": context.run_id,
        "dataset": dataset,
        "experiment": experiment,
        "status": "completed" if valid_run else "completed_with_failures",
        "aggregation": "FedAvg",
        "seed": seed,
        "configuration": {
            "clients": num_clients,
            "federated_rounds": num_rounds,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "batch_size": context.run_config["batch-size"],
            "fraction_evaluate": context.run_config["fraction-evaluate"],
        },
        "metrics": {
            "round": server_metrics,
        },
        "client_train_metrics": client_train_metrics,
        "training_results": {
            "received_results_by_round": received_by_round,
            "client_failures_by_round": failed_by_round,
            "valid_run": valid_run,
        },
    }

    result_dir = Path("results") / dataset / experiment
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"seed-{seed}.json"
    result_path.write_text(json.dumps(record, indent=2) + "\n")
    return result_path


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    seed: int = context.run_config["seed"]
    dataset: str = context.run_config["dataset"]
    num_clients: int = context.run_config["num-clients"]
    set_seed(seed)
    set_dataset(dataset)

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = VerifiedFedAvg(
        num_clients=num_clients,
        fraction_train=1.0,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=num_clients,
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    result_path = save_run_result(context, result, num_rounds, strategy)
    print(f"\nSaving run results to {result_path}...")

    if context.run_config["save-model"]:
        dataset: str = context.run_config["dataset"]
        seed: int = context.run_config["seed"]
        local_epochs: int = context.run_config["local-epochs"]
        learning_rate: float = context.run_config["learning-rate"]
        lr_label = f"lr{learning_rate:.3f}".replace(".", "p")
        model_dir = Path("models") / dataset / "baseline"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / (
            f"seed-{seed}_rounds-{num_rounds}_epochs-{local_epochs}_{lr_label}.pt"
        )
        print(f"\nSaving final model to {model_path}...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, model_path)


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
