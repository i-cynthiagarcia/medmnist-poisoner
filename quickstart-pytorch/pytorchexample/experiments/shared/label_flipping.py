"""Targeted label-flipping helpers shared by client and server apps."""

import math
from collections.abc import Callable

import torch
from torch.utils.data import Dataset, Subset


class LabelFlippedDataset(Dataset):
    """A dataset view that changes selected labels without changing images."""

    def __init__(
        self,
        dataset: Dataset,
        flipped_positions: list[int],
        target_class_id: int,
    ) -> None:
        self.dataset = dataset
        self.flipped_positions = frozenset(flipped_positions)
        self.target_class_id = target_class_id
        self.num_flipped = len(self.flipped_positions)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        label = torch.as_tensor(label)
        if index in self.flipped_positions:
            label = label.clone()
            label[...] = self.target_class_id
        return image, label


def validate_label_flip_config(
    num_clients: int,
    num_poisoned_clients: int,
    flip_fraction: float,
    source_class_id: int,
    target_class_id: int,
    num_classes: int,
) -> None:
    """Validate poisoning settings shared by client and server apps."""
    integer_values = {
        "num-clients": num_clients,
        "poison-num-clients": num_poisoned_clients,
        "poison-source-class-id": source_class_id,
        "poison-target-class-id": target_class_id,
        "num-classes": num_classes,
    }
    for name, value in integer_values.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"Configuration value {name!r} must be an integer.")
    if num_clients <= 0:
        raise ValueError("num-clients must be greater than zero.")
    if not 0 <= num_poisoned_clients <= num_clients:
        raise ValueError(
            "poison-num-clients must be between zero and num-clients."
        )
    if isinstance(flip_fraction, bool) or not isinstance(
        flip_fraction, (int, float)
    ):
        raise TypeError("Configuration value 'poison-fraction' must be numeric.")
    if not math.isfinite(flip_fraction) or not 0.0 <= flip_fraction <= 1.0:
        raise ValueError("poison-fraction must be between zero and one.")

    if source_class_id == -1 and target_class_id == -1:
        if num_poisoned_clients != 0 or flip_fraction != 0:
            raise ValueError(
                "Set source and target class IDs to enable label flipping."
            )
        return
    if not (
        0 <= source_class_id < num_classes
        and 0 <= target_class_id < num_classes
    ):
        raise ValueError(
            "Poison source and target class IDs must be valid dataset class IDs."
        )
    if source_class_id == target_class_id:
        raise ValueError("Poison source and target class IDs must differ.")


def select_poisoned_clients(
    num_clients: int, num_poisoned_clients: int, seed: int
) -> list[int]:
    """Select a reproducible, seed-randomized set of client IDs."""
    if num_clients <= 0 or not 0 <= num_poisoned_clients <= num_clients:
        raise ValueError(
            "Poisoned client count must be between zero and num_clients."
        )
    generator = torch.Generator().manual_seed(seed)
    return sorted(
        torch.randperm(num_clients, generator=generator)[
            :num_poisoned_clients
        ].tolist()
    )


def select_flipped_positions(
    labels: torch.Tensor,
    source_class_id: int,
    flip_fraction: float,
    seed: int,
) -> list[int]:
    """Select a seeded rounded fraction of source-class labels to flip."""
    source_positions = (labels.reshape(-1) == source_class_id).nonzero(
        as_tuple=True
    )[0]
    num_flips = round(len(source_positions) * flip_fraction)
    if num_flips == 0:
        return []
    generator = torch.Generator().manual_seed(seed)
    selected = source_positions[
        torch.randperm(len(source_positions), generator=generator)[:num_flips]
    ]
    return selected.tolist()


def make_label_flip_transform(
    poisoned_client_ids: list[int],
    source_class_id: int,
    target_class_id: int,
    flip_fraction: float,
    seed: int,
) -> Callable[[Dataset, int], Dataset]:
    """Create a client transform that flips selected local source labels."""
    selected_clients = frozenset(poisoned_client_ids)

    def transform(dataset: Dataset, partition_id: int) -> Dataset:
        if partition_id not in selected_clients:
            return dataset
        if not isinstance(dataset, Subset):
            raise TypeError(
                "Label flipping requires a client Subset over a labeled dataset."
            )
        labels = getattr(dataset.dataset, "labels", None)
        if labels is None:
            raise ValueError("The training dataset must expose its labels.")
        local_labels = torch.as_tensor(labels).reshape(-1)[dataset.indices]
        flipped_positions = select_flipped_positions(
            local_labels,
            source_class_id,
            flip_fraction,
            seed + partition_id,
        )
        return LabelFlippedDataset(dataset, flipped_positions, target_class_id)

    return transform
