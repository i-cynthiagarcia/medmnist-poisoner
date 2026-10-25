"""Flower federated learning baseline using OrganAMNIST."""

import random

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


def set_dataset(dataset: str) -> None:
    """Select a supported Organ MedMNIST dataset."""
    global DATASET_NAME
    if dataset not in {"organamnist", "organcmnist", "organsmnist"}:
        raise ValueError(f"Unsupported Organ dataset: {dataset}")
    DATASET_NAME = dataset


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


def load_centralized_dataset():
    """Load the currently selected official MedMNIST test dataset."""

    test_dataset = get_dataset("test")

    return DataLoader(
        test_dataset,
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
    running_loss = 0.0

    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    return running_loss / (epochs * len(trainloader))


def test(net, testloader, device):
    """Evaluate the CNN on OrganAMNIST."""

    net.to(device)
    net.eval()

    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)

            outputs = net(images)
            loss += criterion(outputs, labels).item()

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = loss / len(testloader)

    return avg_loss, accuracy
