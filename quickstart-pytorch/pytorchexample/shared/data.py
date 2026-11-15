"""Shared MedMNIST dataset loading and partitioning helpers."""

from collections import Counter

import medmnist
import torch
from medmnist import INFO
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize, ToTensor


SUPPORTED_DATASETS = {"organamnist", "organcmnist", "organsmnist"}

transforms = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])


def _validate_dataset(dataset: str) -> None:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported Organ dataset: {dataset}")


def get_class_names(dataset: str) -> list[str]:
    """Return class names in the numeric label order used by MedMNIST."""
    _validate_dataset(dataset)
    labels = INFO[dataset]["label"]
    return [labels[str(index)] for index in range(len(labels))]


def verify_partition_indices(
    partitions: list[list[int]], dataset_size: int
) -> None:
    """Raise if partitions do not assign every dataset index exactly once."""
    if dataset_size < 0:
        raise ValueError("Dataset size cannot be negative.")
    if not partitions:
        raise ValueError("At least one client partition is required.")

    assigned = [index for partition in partitions for index in partition]
    counts = Counter(assigned)
    invalid = sorted(index for index in counts if index < 0 or index >= dataset_size)
    duplicates = sorted(index for index, count in counts.items() if count > 1)
    missing = sorted(set(range(dataset_size)) - counts.keys())
    if invalid or duplicates or missing:
        raise ValueError(
            "Invalid client partition assignment: "
            f"out-of-range indices={invalid[:10]}, "
            f"overlapping indices={duplicates[:10]}, "
            f"unassigned indices={missing[:10]}."
        )


def get_partition_indices(
    dataset_size: int, num_partitions: int, seed: int
) -> list[list[int]]:
    """Create the seeded interleaved partition of dataset indices."""
    if dataset_size < 0:
        raise ValueError("Dataset size cannot be negative.")
    if num_partitions <= 0:
        raise ValueError("Number of partitions must be greater than zero.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    return [
        indices[partition_id::num_partitions]
        for partition_id in range(num_partitions)
    ]


def build_partition_manifest(
    dataset, dataset_name: str, num_clients: int, seed: int
) -> dict:
    """Verify a dataset partition and return per-client image and class counts."""
    dataset_size = len(dataset)
    partitions = get_partition_indices(dataset_size, num_clients, seed)
    verify_partition_indices(partitions, dataset_size)
    labels = getattr(dataset, "labels", None)
    if labels is None or len(labels) != dataset_size:
        raise ValueError(
            f"Dataset {dataset_name} must expose one label per training image."
        )

    class_names = get_class_names(dataset_name)
    clients = []
    for client_id, indices in enumerate(partitions):
        counts = Counter()
        for index in indices:
            label = labels[index]
            if hasattr(label, "item"):
                label = label.item()
            label_id = int(label)
            if label_id < 0 or label_id >= len(class_names):
                raise ValueError(
                    f"Dataset {dataset_name} has out-of-range class label "
                    f"{label_id} at training index {index}."
                )
            counts[label_id] += 1
        clients.append(
            {
                "partition_id": client_id,
                "num_images": len(indices),
                "class_counts": {
                    class_names[class_id]: counts.get(class_id, 0)
                    for class_id in range(len(class_names))
                },
            }
        )

    return {
        "dataset": dataset_name,
        "seed": seed,
        "num_clients": num_clients,
        "total_training_examples": dataset_size,
        "verification": {
            "partitions_are_disjoint": True,
            "all_training_examples_assigned": True,
            "each_training_example_assigned_once": True,
        },
        "clients": clients,
    }


def get_dataset(split: str, dataset: str, size: int = 28):
    """Load an official MedMNIST split without relying on global dataset state."""
    _validate_dataset(dataset)
    info = INFO[dataset]
    dataset_class = getattr(medmnist, info["python_class"])

    return dataset_class(
        split=split,
        transform=transforms,
        download=True,
        size=size,
    )


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    seed: int,
    dataset: str,
):
    """Load one uniform client partition and the shared validation split."""
    train_dataset = get_dataset("train", dataset)
    val_dataset = get_dataset("val", dataset)

    if partition_id < 0 or partition_id >= num_partitions:
        raise ValueError(
            f"Partition ID {partition_id} is outside the valid range "
            f"0..{num_partitions - 1}."
        )
    partitions = get_partition_indices(len(train_dataset), num_partitions, seed)
    client_train = Subset(train_dataset, partitions[partition_id])

    shuffle_generator = torch.Generator().manual_seed(seed + partition_id)
    trainloader = DataLoader(
        client_train,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )
    validation_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return trainloader, validation_loader


def load_centralized_dataset(
    split: str, dataset: str, batch_size: int = 128
):
    """Load a centralized validation or test split."""
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported centralized dataset split: {split}")

    return DataLoader(
        get_dataset(split, dataset),
        batch_size=batch_size,
        shuffle=False,
    )
