"""Shared model initialization, training, and evaluation loops."""

import random
from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn

from pytorchexample.shared.data import get_class_names


class EvaluationMetrics(TypedDict):
    """Image-weighted evaluation metrics and per-class details."""

    loss: float
    accuracy: float
    per_class_precision: list[float]
    per_class_recall: list[float]
    per_class_f1: list[float]
    per_class_support: list[int]
    confusion_matrix: list[int]


def set_seed(seed: int) -> None:
    """Initialize the random sources used by baseline training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _train_epoch(net, trainloader, optimizer, device) -> tuple[float, int]:
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    net.train()
    loss_sum = 0.0
    total_images = 0

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

    return loss_sum, total_images


def train_epoch(net, trainloader, optimizer, device) -> float:
    """Train one epoch while retaining the caller's optimizer state."""
    loss_sum, total_images = _train_epoch(net, trainloader, optimizer, device)
    if total_images == 0:
        raise ValueError("Cannot train on an empty data loader.")
    return loss_sum / total_images


def train(net, trainloader, epochs, lr, device):
    """Train the CNN locally on a client."""
    net.to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=0.9
    )
    loss_sum = 0.0
    total_images = 0

    for _ in range(epochs):
        epoch_loss_sum, epoch_images = _train_epoch(
            net, trainloader, optimizer, device
        )
        loss_sum += epoch_loss_sum
        total_images += epoch_images

    if total_images == 0:
        raise ValueError("Cannot train on an empty data loader or zero epochs.")

    return loss_sum / total_images


def evaluate(net, testloader, device, dataset: str) -> EvaluationMetrics:
    """Evaluate loss, accuracy, per-class metrics, and confusion matrix."""
    net.to(device)
    net.eval()

    loss_sum = 0.0
    correct = 0
    total_images = 0
    confusion_matrix = None
    class_names = get_class_names(dataset)

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)

            outputs = net(images)
            num_classes = outputs.shape[1]
            if num_classes != len(class_names):
                raise ValueError(
                    f"Model has {num_classes} outputs, but dataset "
                    f"{dataset} has {len(class_names)} classes."
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


def test(net, testloader, device, dataset: str):
    """Return image-weighted loss and accuracy for existing callers."""
    metrics = evaluate(net, testloader, device, dataset)
    return metrics["loss"], metrics["accuracy"]
