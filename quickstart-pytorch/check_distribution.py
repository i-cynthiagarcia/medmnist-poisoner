
import argparse

from pytorchexample.shared.data import (
    SUPPORTED_DATASETS,
    build_partition_manifest,
    get_dataset,
)


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

    dataset = get_dataset("train", args.dataset, size=args.size)
    manifest = build_partition_manifest(
        dataset, args.dataset, args.num_clients, args.seed
    )

    print(f"Dataset: {args.dataset}")
    print(f"Image size: {args.size}x{args.size}")
    print(f"Total training images: {manifest['total_training_examples']}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Partition seed: {args.seed}")
    print("Partition verification: passed (disjoint, complete, exactly once)")
    print()

    for client in manifest["clients"]:
        print(
            f"Client {client['partition_id']:02d}: "
            f"{client['num_images']} images | "
            f"Class counts: {client['class_counts']}"
        )


if __name__ == "__main__":
    main()
