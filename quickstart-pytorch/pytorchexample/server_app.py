"""pytorchexample: A Flower / PyTorch app."""

import json
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.artifacts import resolve_protocol_id, run_artifact_stem
from pytorchexample.task import (
    Net,
    evaluate,
    get_class_names,
    load_centralized_dataset,
    set_dataset,
    set_seed,
)
from pytorchexample.visualization import save_confusion_matrix_plot

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


def serialize_evaluation_metrics(
    metric_record: MetricRecord, class_names: list[str]
) -> dict:
    """Convert flat Flower metrics into a readable per-class JSON structure."""
    num_classes = len(class_names)
    per_class_values = {
        key: json_value(metric_record[key])
        for key in (
            "per_class_precision",
            "per_class_recall",
            "per_class_f1",
            "per_class_support",
        )
    }
    if any(len(values) != num_classes for values in per_class_values.values()):
        raise ValueError("Per-class metrics do not match the dataset class count.")

    confusion_values = json_value(metric_record["confusion_matrix"])
    if len(confusion_values) != num_classes * num_classes:
        raise ValueError("Confusion matrix does not match the dataset class count.")

    return {
        "loss": json_value(metric_record["loss"]),
        "accuracy": json_value(metric_record["accuracy"]),
        "per_class": {
            class_name: {
                "precision": per_class_values["per_class_precision"][index],
                "recall": per_class_values["per_class_recall"][index],
                "f1": per_class_values["per_class_f1"][index],
                "support": per_class_values["per_class_support"][index],
            }
            for index, class_name in enumerate(class_names)
        },
        "confusion_matrix": [
            confusion_values[row * num_classes : (row + 1) * num_classes]
            for row in range(num_classes)
        ],
    }


def experiment_name(num_rounds: int, local_epochs: int, learning_rate: float) -> str:
    """Build the configuration-based experiment directory name."""
    return (
        f"baseline_r{num_rounds}_e{local_epochs}_"
        f"lr{learning_rate:.3f}".replace(".", "p")
    )


def ensure_run_outputs_unused(
    context: Context, experiment: str, protocol_id: str
) -> None:
    """Fail before training if any output for this run ID already exists."""
    dataset = context.run_config["dataset"]
    seed = context.run_config["seed"]
    run_id = context.run_id
    artifact_stem = run_artifact_stem(seed, run_id)
    result_dir = (
        Path(context.run_config["results-dir"])
        / protocol_id
        / dataset
        / experiment
    )
    model_dir = (
        Path(context.run_config["models-dir"])
        / protocol_id
        / dataset
        / experiment
    )
    paths = (
        result_dir / f"{artifact_stem}.json",
        model_dir / f"{artifact_stem}.pt",
        model_dir / "checkpoints" / artifact_stem,
        result_dir / "plots" / artifact_stem,
    )
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing run outputs: "
            + ", ".join(str(path) for path in existing)
        )


def save_round_checkpoint(
    context: Context, arrays: ArrayRecord, server_round: int, protocol_id: str
) -> Path:
    """Save aggregated global weights for one completed server round."""
    dataset = context.run_config["dataset"]
    seed = context.run_config["seed"]
    run_id = context.run_id
    num_rounds = context.run_config["num-server-rounds"]
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    experiment = experiment_name(num_rounds, local_epochs, learning_rate)
    artifact_stem = run_artifact_stem(seed, run_id)
    checkpoint_dir = (
        Path(context.run_config["models-dir"])
        / protocol_id
        / dataset
        / experiment
        / "checkpoints"
        / artifact_stem
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"round-{server_round:03d}.pt"
    if checkpoint_path.exists():
        raise FileExistsError(
            f"Refusing to replace existing checkpoint: {checkpoint_path}"
        )
    torch.save(arrays.to_torch_state_dict(), checkpoint_path)
    return checkpoint_path


def save_run_result(
    context: Context,
    result,
    num_rounds: int,
    strategy: VerifiedFedAvg,
    test_metrics: MetricRecord | None,
    round_checkpoint_paths: dict[int, Path],
    protocol_id: str,
) -> Path:
    """Save centralized and training metrics for one completed Flower run."""
    dataset = context.run_config["dataset"]
    seed = context.run_config["seed"]
    run_id = context.run_id
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    experiment = experiment_name(num_rounds, local_epochs, learning_rate)
    artifact_stem = run_artifact_stem(seed, run_id)
    class_names = get_class_names(dataset)
    result_dir = (
        Path(context.run_config["results-dir"])
        / protocol_id
        / dataset
        / experiment
    )
    result_path = result_dir / f"{artifact_stem}.json"
    if result_path.exists():
        raise FileExistsError(
            f"Refusing to replace existing run record: {result_path}"
        )

    server_metrics = {}
    for server_round, metric_record in result.evaluate_metrics_serverapp.items():
        server_metrics[str(server_round)] = serialize_evaluation_metrics(
            metric_record, class_names
        )

    test_final_metrics = (
        serialize_evaluation_metrics(test_metrics, class_names)
        if test_metrics is not None
        else None
    )
    plot_dir = result_dir / "plots" / artifact_stem
    confusion_matrix_plots = {
        "validation_by_round": {},
        "validation_final_recomputed": None,
        "test_final": None,
    }
    for server_round, metrics in sorted(
        server_metrics.items(), key=lambda item: int(item[0])
    ):
        plot_path = plot_dir / f"validation-round-{int(server_round):03d}.png"
        save_confusion_matrix_plot(
            metrics["confusion_matrix"],
            class_names,
            plot_path,
            f"Validation confusion matrix - round {server_round}",
        )
        confusion_matrix_plots["validation_by_round"][server_round] = (
            plot_path.as_posix()
        )
    if test_final_metrics is not None:
        test_plot_path = plot_dir / "test-final.png"
        save_confusion_matrix_plot(
            test_final_metrics["confusion_matrix"],
            class_names,
            test_plot_path,
            "Test confusion matrix - final model",
        )
        confusion_matrix_plots["test_final"] = test_plot_path.as_posix()

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
        "protocol_id": protocol_id,
        "dataset": dataset,
        "experiment": experiment,
        "status": "completed" if valid_run else "completed_with_failures",
        "aggregation": "FedAvg",
        "seed": seed,
        "loss_aggregation": "per_image",
        "configuration": {
            "clients": num_clients,
            "federated_rounds": num_rounds,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "batch_size": context.run_config["batch-size"],
            "fraction_evaluate": context.run_config["fraction-evaluate"],
        },
        "metrics": {
            "class_names": class_names,
            "confusion_matrix_axes": {
                "rows": "true_class",
                "columns": "predicted_class",
            },
            "validation_by_round": server_metrics,
            "test_final": test_final_metrics,
            "confusion_matrix_plots": confusion_matrix_plots,
        },
        "round_checkpoints": {
            str(server_round): path.as_posix()
            for server_round, path in sorted(round_checkpoint_paths.items())
        },
        "client_train_metrics": client_train_metrics,
        "training_results": {
            "received_results_by_round": received_by_round,
            "client_failures_by_round": failed_by_round,
            "valid_run": valid_run,
        },
    }

    result_dir.mkdir(parents=True, exist_ok=True)
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
    experiment = experiment_name(
        num_rounds,
        context.run_config["local-epochs"],
        lr,
    )
    protocol_id = resolve_protocol_id(context.run_config["protocol-id"])
    ensure_run_outputs_unused(context, experiment, protocol_id)
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

    round_checkpoint_paths = {}

    def evaluate_round(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        if server_round > 0 and context.run_config["save-model"]:
            round_checkpoint_paths[server_round] = save_round_checkpoint(
                context, arrays, server_round, protocol_id
            )
        return evaluate_arrays(arrays, "val")

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_round,
    )

    test_metrics = None
    if context.run_config["evaluate-test"]:
        test_metrics = evaluate_arrays(result.arrays, "test")

    result_path = save_run_result(
        context,
        result,
        num_rounds,
        strategy,
        test_metrics,
        round_checkpoint_paths,
        protocol_id,
    )
    print(f"\nSaving run results to {result_path}...")

    if context.run_config["save-model"]:
        model_dir = (
            Path(context.run_config["models-dir"])
            / protocol_id
            / dataset
            / experiment
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        artifact_stem = run_artifact_stem(seed, context.run_id)
        model_path = model_dir / f"{artifact_stem}.pt"
        if model_path.exists():
            raise FileExistsError(
                f"Refusing to replace existing final model: {model_path}"
            )
        print(f"\nSaving final model to {model_path}...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, model_path)


def evaluate_arrays(arrays: ArrayRecord, split: str) -> MetricRecord:
    """Evaluate model arrays on a centralized dataset split."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = load_centralized_dataset(split)

    metrics = evaluate(model, dataloader, device)
    return MetricRecord(metrics)
