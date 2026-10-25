
import argparse
from collections import Counter

import medmnist
import torch
from medmnist import INFO


SUPPORTED_DATASETS = {"organamnist", "organcmnist", "organsmnist"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check the uniform client partition used by the MedMNIST baseline."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(SUPPORTED_DATASETS),
        default="organamnist",
        help="MedMNIST Organ dataset to inspect.",
    )
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=28, choices=(28, 64, 128, 224))
    args = parser.parse_args()

    if args.num_clients <= 0:
        parser.error("--num-clients must be greater than zero")

    return args


def main():
    args = parse_args()

    info = INFO[args.dataset]
    dataset_class = getattr(medmnist, info["python_class"])
    dataset = dataset_class(
        split="train",
        download=True,
        size=args.size,
    )

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()

    print(f"Dataset: {args.dataset}")
    print(f"Image size: {args.size}x{args.size}")
    print(f"Total training images: {len(dataset)}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Partition seed: {args.seed}")
    print()

    num_classes = len(info["label"])
    for client_id in range(args.num_clients):
        client_indices = indices[client_id::args.num_clients]
        labels = [int(dataset.labels[i].item()) for i in client_indices]

        counts = Counter(labels)
        distribution = [counts.get(class_id, 0) for class_id in range(num_classes)]

        print(
            f"Client {client_id:02d}: "
            f"{len(client_indices)} images | "
            f"Class counts: {distribution}"
        )


if __name__ == "__main__":
    main()
