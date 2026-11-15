"""pytorchexample: A Flower / PyTorch app."""

import json
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from pytorchexample.shared.config import (
    require_config_float,
    require_config_int,
    require_config_string,
)
from pytorchexample.shared.metrics import json_value, serialize_evaluation_metrics
from pytorchexample.shared.artifacts import (
    resolve_execution_id,
    resolve_protocol_id,
    run_artifact_stem,
    validate_generation_id,
)
from pytorchexample.shared.data import (
    build_partition_manifest,
    get_class_names,
    get_dataset,
    load_centralized_dataset,
)
from pytorchexample.shared.model import Net
from pytorchexample.experiments.shared.label_flipping import (
    select_poisoned_clients,
    validate_label_flip_config,
)
from pytorchexample.shared.training import (
    evaluate,
    set_seed,
)
from pytorchexample.shared.visualization import save_confusion_matrix_plot

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


def experiment_name(
    num_rounds: int,
    local_epochs: int,
    learning_rate: float,
    label_flipping_enabled: bool = False,
    poison_num_clients: int = 0,
    poison_fraction: float = 0.0,
    source_class_id: int = -1,
    target_class_id: int = -1,
) -> str:
    """Build the configuration-based experiment directory name."""
    if label_flipping_enabled:
        prefix = "label_flip"
    else:
        prefix = "baseline"
    name = (
        f"{prefix}_r{num_rounds}_e{local_epochs}_"
        f"lr{learning_rate:.3f}".replace(".", "p")
    )
    if label_flipping_enabled:
        name += (
            f"_clients{poison_num_clients}"
            f"_f{poison_fraction:.3f}_s{source_class_id}_t{target_class_id}"
        ).replace(".", "p")
    return name


def experiment_name_for_context(
    context: Context, label_flipping_enabled: bool = False
) -> str:
    """Build the output name from the full training configuration."""
    config = context.run_config
    arguments = (
        config["num-server-rounds"],
        config["local-epochs"],
        config["learning-rate"],
    )
    if not label_flipping_enabled:
        return experiment_name(*arguments)
    return experiment_name(
        *arguments,
        label_flipping_enabled=True,
        poison_num_clients=config["poison-num-clients"],
        poison_fraction=config["poison-fraction"],
        source_class_id=config["poison-source-class-id"],
        target_class_id=config["poison-target-class-id"],
    )


def targeted_source_to_target_metrics(
    metrics: dict, source_class_id: int, target_class_id: int
) -> dict:
    """Extract the targeted error rate and source-class support."""
    matrix = metrics["confusion_matrix"]
    source_support = sum(matrix[source_class_id])
    return {
        "rate": (
            matrix[source_class_id][target_class_id] / source_support
            if source_support
            else None
        ),
        "support": source_support,
    }


def ensure_run_outputs_unused(
    context: Context,
    experiment: str,
    generation_id: str,
    protocol_id: str,
    execution_id: str,
) -> None:
    """Fail before training if any output for this run ID already exists."""
    dataset = require_config_string(context.run_config["dataset"], "dataset")
    seed = context.run_config["seed"]
    run_id = context.run_id
    artifact_stem = run_artifact_stem(seed, run_id)
    result_dir = (
        Path(context.run_config["results-dir"])
        / generation_id
        / protocol_id
        / execution_id
        / dataset
        / experiment
    )
    model_dir = (
        Path(context.run_config["models-dir"])
        / generation_id
        / protocol_id
        / execution_id
        / dataset
        / experiment
    )
    paths = (
        result_dir / f"{artifact_stem}.json",
        result_dir / f"partition-{artifact_stem}.json",
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


def save_partition_manifest(
    context: Context,
    manifest: dict,
    generation_id: str,
    protocol_id: str,
    execution_id: str,
    label_flipping_enabled: bool = False,
) -> Path:
    """Save verified client partition sizes and class counts before training."""
    dataset = require_config_string(context.run_config["dataset"], "dataset")
    seed = context.run_config["seed"]
    run_id = context.run_id
    result_dir = (
        Path(context.run_config["results-dir"])
        / generation_id
        / protocol_id
        / execution_id
        / dataset
        / experiment_name_for_context(context, label_flipping_enabled)
    )
    artifact_stem = run_artifact_stem(seed, run_id)
    manifest_path = result_dir / f"partition-{artifact_stem}.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to replace existing partition manifest: {manifest_path}"
        )

    record = {
        "run_id": run_id,
        "generation_id": generation_id,
        "protocol_id": protocol_id,
        "execution_id": execution_id,
        **manifest,
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(record, indent=2) + "\n")
    return manifest_path


def save_round_checkpoint(
    context: Context,
    arrays: ArrayRecord,
    server_round: int,
    generation_id: str,
    protocol_id: str,
    execution_id: str,
    label_flipping_enabled: bool = False,
) -> Path:
    """Save aggregated global weights for one completed server round."""
    dataset = require_config_string(context.run_config["dataset"], "dataset")
    seed = context.run_config["seed"]
    run_id = context.run_id
    experiment = experiment_name_for_context(context, label_flipping_enabled)
    artifact_stem = run_artifact_stem(seed, run_id)
    checkpoint_dir = (
        Path(context.run_config["models-dir"])
        / generation_id
        / protocol_id
        / execution_id
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
    partition_manifest_path: Path,
    generation_id: str,
    protocol_id: str,
    execution_id: str,
    label_flipping_enabled: bool = False,
) -> Path:
    """Save federated aggregate and training metrics for one Flower run."""
    dataset = require_config_string(context.run_config["dataset"], "dataset")
    seed = context.run_config["seed"]
    run_id = context.run_id
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    experiment = experiment_name_for_context(context, label_flipping_enabled)
    artifact_stem = run_artifact_stem(seed, run_id)
    class_names = get_class_names(dataset)
    result_dir = (
        Path(context.run_config["results-dir"])
        / generation_id
        / protocol_id
        / execution_id
        / dataset
        / experiment
    )
    result_path = result_dir / f"{artifact_stem}.json"
    model_artifact = (
        Path(context.run_config["models-dir"])
        / generation_id
        / protocol_id
        / execution_id
        / dataset
        / experiment
        / f"{artifact_stem}.pt"
    )
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
    label_flipping = None
    targeted_validation_metrics = None
    targeted_test_metrics = None
    if label_flipping_enabled:
        source_class_id = context.run_config["poison-source-class-id"]
        target_class_id = context.run_config["poison-target-class-id"]
        num_poisoned_clients = context.run_config["poison-num-clients"]
        poisoned_client_ids = select_poisoned_clients(
            context.run_config["num-clients"],
            num_poisoned_clients,
            context.run_config["seed"],
        )
        source_class_name = class_names[source_class_id]
        source_examples_by_client = {
            client["partition_id"]: client["class_counts"][source_class_name]
            for client in json.loads(
                partition_manifest_path.read_text()
            )["clients"]
        }
        flipped_examples_by_client = {
            str(client_id): round(
                source_examples_by_client[client_id]
                * context.run_config["poison-fraction"]
            )
            for client_id in poisoned_client_ids
        }
        label_flipping = {
            "type": "targeted_label_flipping",
            "source_class_id": source_class_id,
            "source_class_name": source_class_name,
            "target_class_id": target_class_id,
            "target_class_name": class_names[target_class_id],
            "fraction_of_source_labels_flipped_per_poisoned_client": (
                context.run_config["poison-fraction"]
            ),
            "num_poisoned_clients": num_poisoned_clients,
            "poisoned_client_ids": poisoned_client_ids,
            "flipped_source_examples_by_client": flipped_examples_by_client,
            "total_flipped_source_examples": sum(
                flipped_examples_by_client.values()
            ),
            "client_selection": "seeded_random_without_replacement",
            "example_selection": "seeded_random_per_client",
        }
        targeted_validation_metrics = {
            server_round: targeted_source_to_target_metrics(
                metrics, source_class_id, target_class_id
            )
            for server_round, metrics in server_metrics.items()
        }
        if test_final_metrics is not None:
            targeted_test_metrics = targeted_source_to_target_metrics(
                test_final_metrics, source_class_id, target_class_id
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
        "generation_id": generation_id,
        "protocol_id": protocol_id,
        "execution_id": execution_id,
        "dataset": dataset,
        "experiment": experiment,
        "status": "completed" if valid_run else "completed_with_failures",
        "training_mode": (
            "federated_label_flipping"
            if label_flipping is not None
            else "federated"
        ),
        "model_type": "federated_aggregate",
        "model_artifact": (
            model_artifact.as_posix()
            if context.run_config["save-model"]
            else None
        ),
        "aggregation": "FedAvg",
        "seed": seed,
        "loss_aggregation": "per_image",
        "configuration": {
            "clients": num_clients,
            "client_models": {
                "count": num_clients,
                "training_data": "uniform_partition",
                "weights_saved": False,
            },
            "federated_rounds": num_rounds,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "batch_size": context.run_config["batch-size"],
            "fraction_evaluate": context.run_config["fraction-evaluate"],
            **(
                {"label_flipping": label_flipping}
                if label_flipping is not None
                else {}
            ),
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
            **(
                {
                    "targeted_source_to_target_misclassification": {
                        "validation_by_round": targeted_validation_metrics,
                        "test_final": targeted_test_metrics,
                    }
                }
                if label_flipping is not None
                else {}
            ),
        },
        "round_checkpoints": {
            str(server_round): path.as_posix()
            for server_round, path in sorted(round_checkpoint_paths.items())
        },
        "partition_manifest": partition_manifest_path.as_posix(),
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


def run_server(
    grid: Grid, context: Context, label_flipping_enabled: bool = False
) -> None:
    """Run the shared federated protocol with optional label flipping."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    seed: int = context.run_config["seed"]
    dataset = require_config_string(context.run_config["dataset"], "dataset")
    num_clients: int = require_config_int(
        context.run_config["num-clients"], "num-clients"
    )
    if label_flipping_enabled:
        num_poisoned_clients = require_config_int(
            context.run_config["poison-num-clients"], "poison-num-clients"
        )
        poison_fraction = require_config_float(
            context.run_config["poison-fraction"], "poison-fraction"
        )
        source_class_id = require_config_int(
            context.run_config["poison-source-class-id"],
            "poison-source-class-id",
        )
        target_class_id = require_config_int(
            context.run_config["poison-target-class-id"],
            "poison-target-class-id",
        )
        validate_label_flip_config(
            num_clients,
            num_poisoned_clients,
            poison_fraction,
            source_class_id,
            target_class_id,
            len(get_class_names(dataset)),
        )
    experiment = experiment_name_for_context(context, label_flipping_enabled)
    generation_id = validate_generation_id(
        context.run_config["generation-id"]
    )
    protocol_id = resolve_protocol_id(context.run_config["protocol-id"])
    execution_id = resolve_execution_id(
        context.run_config["execution-id"],
        context.run_config["results-dir"],
        context.run_config["models-dir"],
        generation_id,
        protocol_id,
    )
    ensure_run_outputs_unused(
        context, experiment, generation_id, protocol_id, execution_id
    )
    set_seed(seed)
    partition_manifest = build_partition_manifest(
        get_dataset("train", dataset), dataset, num_clients, seed
    )
    partition_manifest_path = save_partition_manifest(
        context,
        partition_manifest,
        generation_id,
        protocol_id,
        execution_id,
        label_flipping_enabled,
    )

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
                context,
                arrays,
                server_round,
                generation_id,
                protocol_id,
                execution_id,
                label_flipping_enabled,
            )
        return evaluate_arrays(arrays, "val", dataset)

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
        test_metrics = evaluate_arrays(result.arrays, "test", dataset)

    result_path = save_run_result(
        context,
        result,
        num_rounds,
        strategy,
        test_metrics,
        round_checkpoint_paths,
        partition_manifest_path,
        generation_id,
        protocol_id,
        execution_id,
        label_flipping_enabled,
    )
    print(f"\nSaving run results to {result_path}...")

    if context.run_config["save-model"]:
        model_dir = (
            Path(context.run_config["models-dir"])
            / generation_id
            / protocol_id
            / execution_id
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


def evaluate_arrays(
    arrays: ArrayRecord, split: str, dataset: str
) -> MetricRecord:
    """Evaluate model arrays on a centralized dataset split."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = load_centralized_dataset(split, dataset)

    metrics = evaluate(model, dataloader, device, dataset)
    return MetricRecord(metrics)
