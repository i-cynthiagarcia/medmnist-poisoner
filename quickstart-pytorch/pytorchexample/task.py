"""Flower federated learning baseline using OrganAMNIST."""

import random
from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import medmnist

from medmnist import INFO
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """CNN for OrganAMNIST: grayscale images and 11 classes."""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


DATASET_NAME = "organamnist"


class EvaluationMetrics(TypedDict):
    """Image-weighted evaluation metrics and per-class details."""

    loss: float
    accuracy: float
    per_class_precision: list[float]
    per_class_recall: list[float]
    per_class_f1: list[float]
    per_class_support: list[int]
    confusion_matrix: list[int]


def set_dataset(dataset: str) -> None:
    """Select a supported Organ MedMNIST dataset."""
    global DATASET_NAME
    if dataset not in {"organamnist", "organcmnist", "organsmnist"}:
        raise ValueError(f"Unsupported Organ dataset: {dataset}")
    DATASET_NAME = dataset


def get_class_names(dataset: str | None = None) -> list[str]:
    """Return class names in the numeric label order used by MedMNIST."""
    dataset_name = dataset or DATASET_NAME
    labels = INFO[dataset_name]["label"]
    return [labels[str(index)] for index in range(len(labels))]


def set_seed(seed: int) -> None:
    """Initialize the random sources used by baseline training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


transforms = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])


def get_dataset(split):
    """Load the official MedMNIST dataset split."""

    info = INFO[DATASET_NAME]
    dataset_class = getattr(medmnist, info["python_class"])

    return dataset_class(
        split=split,
        transform=transforms,
        download=True,
        size=28,
    )


def load_data(partition_id: int, num_partitions: int, batch_size: int, seed: int = 42, dataset: str = "organamnist"):
    """Uniformly partition an Organ MedMNIST training dataset."""

    set_dataset(dataset)
    train_dataset = get_dataset("train")
    val_dataset = get_dataset("val")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(train_dataset), generator=generator
    ).tolist()

    client_indices = indices[partition_id::num_partitions]

    client_train = Subset(train_dataset, client_indices)

    shuffle_generator = torch.Generator().manual_seed(seed + partition_id)
    trainloader = DataLoader(
        client_train,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )

    # Shared validation data for initial baseline testing
    testloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return trainloader, testloader


def load_centralized_dataset(split: str):
    """Load a centralized validation or test split."""
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported centralized dataset split: {split}")

    dataset = get_dataset(split)

    return DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
    )


def train(net, trainloader, epochs, lr, device):
    """Train the CNN locally on a client."""

    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=0.9
    )

    net.train()
    loss_sum = 0.0
    total_images = 0

    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            image_count = labels.size(0)
            loss_sum += loss.item() * image_count
            total_images += image_count

    if total_images == 0:
        raise ValueError("Cannot train on an empty data loader or zero epochs.")

    return loss_sum / total_images


def evaluate(net, testloader, device) -> EvaluationMetrics:
    """Evaluate loss, accuracy, per-class metrics, and confusion matrix."""

    net.to(device)
    net.eval()

    loss_sum = 0.0
    correct = 0
    total_images = 0
    confusion_matrix = None
    class_names = get_class_names()

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)

            outputs = net(images)
            num_classes = outputs.shape[1]
            if num_classes != len(class_names):
                raise ValueError(
                    f"Model has {num_classes} outputs, but dataset "
                    f"{DATASET_NAME} has {len(class_names)} classes."
                )
            if confusion_matrix is None:
                confusion_matrix = torch.zeros(
                    (num_classes, num_classes), dtype=torch.int64
                )

            predictions = outputs.argmax(dim=1)
            image_count = labels.size(0)
            loss_sum += nn.functional.cross_entropy(
                outputs, labels, reduction="sum"
            ).item()
            correct += (predictions == labels).sum().item()
            total_images += image_count
            batch_confusion = torch.bincount(
                labels.cpu() * num_classes + predictions.cpu(),
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)
            confusion_matrix += batch_confusion

    if total_images == 0 or confusion_matrix is None:
        raise ValueError("Cannot evaluate an empty data loader.")

    true_positives = confusion_matrix.diag().to(torch.float64)
    class_support = confusion_matrix.sum(dim=1)
    predicted_count = confusion_matrix.sum(dim=0).to(torch.float64)
    precision = true_positives / predicted_count.clamp_min(1)
    recall = true_positives / class_support.to(torch.float64).clamp_min(1)
    f1_denominator = precision + recall
    f1 = torch.where(
        f1_denominator > 0,
        2 * precision * recall / f1_denominator.clamp_min(1e-12),
        0,
    )

    return {
        "loss": loss_sum / total_images,
        "accuracy": correct / total_images,
        "per_class_precision": [float(value) for value in precision.tolist()],
        "per_class_recall": [float(value) for value in recall.tolist()],
        "per_class_f1": [float(value) for value in f1.tolist()],
        "per_class_support": [int(value) for value in class_support.tolist()],
        "confusion_matrix": [
            int(value) for value in confusion_matrix.flatten().tolist()
        ],
    }


def test(net, testloader, device):
    """Return image-weighted loss and accuracy for existing callers."""
    metrics = evaluate(net, testloader, device)
    return metrics["loss"], metrics["accuracy"]
