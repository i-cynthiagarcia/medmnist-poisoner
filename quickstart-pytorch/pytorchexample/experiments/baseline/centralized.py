"""Train and save a centralized full-training-set MedMNIST baseline."""

import argparse
import json
from pathlib import Path

import torch
from flwr.app import MetricRecord
from torch.utils.data import DataLoader

from pytorchexample.shared.artifacts import (
    resolve_execution_id,
    resolve_protocol_id,
    run_artifact_stem,
    validate_generation_id,
)
from pytorchexample.shared.metrics import (
    serialize_evaluation_metrics,
)
from pytorchexample.shared.data import (
    SUPPORTED_DATASETS,
    get_class_names,
    get_dataset,
    load_centralized_dataset,
)
from pytorchexample.shared.model import Net
from pytorchexample.shared.training import (
    evaluate,
    set_seed,
    train_epoch,
)
from pytorchexample.shared.visualization import save_confusion_matrix_plot


def experiment_name(
    num_rounds: int, local_epochs: int, learning_rate: float
) -> str:
    """Name an output folder for a centralized run's matched training budget."""
    learning_rate_name = f"{learning_rate:.3f}".replace(".", "p")
    return (
        f"centralized_r{num_rounds}_e{local_epochs}_lr{learning_rate_name}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one model on the full MedMNIST training split. The epoch "
            "budget is num-rounds times local-epochs."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(SUPPORTED_DATASETS),
        default="organamnist",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--generation-id", default="generation-v3")
    parser.add_argument("--protocol-id", default="protocol-v1")
    parser.add_argument("--execution-id", default="auto")
    parser.add_argument("--results-dir", type=Path, default=Path("results/baseline"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/baseline"))
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the final model once on the test split.",
    )
    args = parser.parse_args()
    if args.num_rounds <= 0:
        parser.error("--num-rounds must be greater than zero.")
    if args.local_epochs <= 0:
        parser.error("--local-epochs must be greater than zero.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than zero.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")
    return args


def save_checkpoint(
    model: Net,
    checkpoint_dir: Path,
    round_number: int,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"round-{round_number:03d}.pt"
    if checkpoint_path.exists():
        raise FileExistsError(
            f"Refusing to replace existing checkpoint: {checkpoint_path}"
        )
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def main() -> None:
    args = parse_args()
    generation_id = validate_generation_id(args.generation_id)
    protocol_id = resolve_protocol_id(args.protocol_id)
    execution_id = resolve_execution_id(
        args.execution_id,
        args.results_dir,
        args.models_dir,
        generation_id,
        protocol_id,
    )
    run_id = "centralized"
    artifact_stem = run_artifact_stem(args.seed, run_id)
    experiment = experiment_name(
        args.num_rounds, args.local_epochs, args.learning_rate
    )
    result_dir = (
        args.results_dir
        / generation_id
        / protocol_id
        / execution_id
        / args.dataset
        / experiment
    )
    model_dir = (
        args.models_dir
        / generation_id
        / protocol_id
        / execution_id
        / args.dataset
        / experiment
    )
    result_path = result_dir / f"{artifact_stem}.json"
    model_path = model_dir / f"{artifact_stem}.pt"
    checkpoint_dir = model_dir / "checkpoints" / artifact_stem
    plot_dir = result_dir / "plots" / artifact_stem
    existing = [
        path
        for path in (result_path, model_path, checkpoint_dir, plot_dir)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing centralized outputs: "
            + ", ".join(str(path) for path in existing)
        )

    set_seed(args.seed)
    train_dataset = get_dataset("train", args.dataset)
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )
    validation_loader = load_centralized_dataset("val", args.dataset)
    class_names = get_class_names(args.dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9
    )

    validation_by_round = {
        "0": serialize_evaluation_metrics(
            MetricRecord(evaluate(model, validation_loader, device, args.dataset)),
            class_names,
        )
    }
    training_loss_by_epoch = {}
    round_checkpoints = {}
    total_epochs = args.num_rounds * args.local_epochs
    completed_epochs = 0
    for round_number in range(1, args.num_rounds + 1):
        for _ in range(args.local_epochs):
            completed_epochs += 1
            training_loss_by_epoch[str(completed_epochs)] = train_epoch(
                model, trainloader, optimizer, device
            )
        validation_by_round[str(round_number)] = serialize_evaluation_metrics(
            MetricRecord(evaluate(model, validation_loader, device, args.dataset)),
            class_names,
        )
        checkpoint_path = save_checkpoint(
            model, checkpoint_dir, round_number
        )
        round_checkpoints[str(round_number)] = checkpoint_path.as_posix()

    test_metrics = None
    if args.evaluate_test:
        test_metrics = serialize_evaluation_metrics(
            MetricRecord(
                evaluate(
                    model,
                    load_centralized_dataset("test", args.dataset),
                    device,
                    args.dataset,
                )
            ),
            class_names,
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    confusion_matrix_plots = {
        "validation_by_round": {},
        "validation_final_recomputed": None,
        "test_final": None,
    }
    for round_number, metrics in validation_by_round.items():
        plot_path = plot_dir / f"validation-round-{int(round_number):03d}.png"
        save_confusion_matrix_plot(
            metrics["confusion_matrix"],
            class_names,
            plot_path,
            f"Centralized validation confusion matrix - round {round_number}",
        )
        confusion_matrix_plots["validation_by_round"][round_number] = (
            plot_path.as_posix()
        )
    if test_metrics is not None:
        test_plot_path = plot_dir / "test-final.png"
        save_confusion_matrix_plot(
            test_metrics["confusion_matrix"],
            class_names,
            test_plot_path,
            "Centralized test confusion matrix - final model",
        )
        confusion_matrix_plots["test_final"] = test_plot_path.as_posix()

    record = {
        "run_id": run_id,
        "generation_id": generation_id,
        "protocol_id": protocol_id,
        "execution_id": execution_id,
        "dataset": args.dataset,
        "experiment": experiment,
        "status": "completed",
        "training_mode": "centralized",
        "model_type": "centralized_full_train",
        "aggregation": None,
        "seed": args.seed,
        "loss_aggregation": "per_image",
        "configuration": {
            "training_data": "full_train_split",
            "training_examples": len(train_dataset),
            "epochs": total_epochs,
            "federated_round_equivalents": args.num_rounds,
            "local_epochs_per_round_equivalent": args.local_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
        },
        "metrics": {
            "class_names": class_names,
            "confusion_matrix_axes": {
                "rows": "true_class",
                "columns": "predicted_class",
            },
            "validation_by_round": validation_by_round,
            "validation_final_recomputed": None,
            "test_final": test_metrics,
            "confusion_matrix_plots": confusion_matrix_plots,
        },
        "training_loss_by_epoch": training_loss_by_epoch,
        "round_checkpoints": round_checkpoints,
        "model_artifact": model_path.as_posix(),
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved centralized full-training model to {model_path}")
    print(f"Saved centralized run results to {result_path}")


if __name__ == "__main__":
    main()
