
import torch
import medmnist
from medmnist import INFO
from collections import Counter

# Load OrganAMNIST training data
info = INFO["organamnist"]
dataset_class = getattr(medmnist, info["python_class"])
dataset = dataset_class(split="train", download=True, size=28)

# Reproduce the same partitioning used in task.py
num_clients = 20
generator = torch.Generator().manual_seed(42)
indices = torch.randperm(len(dataset), generator=generator).tolist()

print(f"Total training images: {len(dataset)}")
print(f"Number of clients: {num_clients}")
print()

for client_id in range(num_clients):
    client_indices = indices[client_id::num_clients]
    labels = [
        int(dataset.labels[i].item())
        for i in client_indices
    ]

    counts = Counter(labels)
    distribution = [counts.get(c, 0) for c in range(11)]

    print(
        f"Client {client_id:02d}: "
        f"{len(client_indices)} images | "
        f"Class counts: {distribution}"
    )
