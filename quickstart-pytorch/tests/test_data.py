import unittest

import torch

from pytorchexample.shared.data import (
    build_partition_manifest,
    get_class_names,
    verify_partition_indices,
)


class LabeledDataset:
    def __init__(self, labels):
        self.labels = torch.tensor(labels).reshape(-1, 1)

    def __len__(self):
        return len(self.labels)


class PartitionTests(unittest.TestCase):
    def test_partition_verification_rejects_overlaps_and_unassigned_indices(self):
        with self.assertRaisesRegex(ValueError, "overlapping indices"):
            verify_partition_indices([[0, 1], [1, 2]], dataset_size=3)
        with self.assertRaisesRegex(ValueError, "unassigned indices"):
            verify_partition_indices([[0], [2]], dataset_size=3)

    def test_manifest_verifies_complete_assignment_and_counts_each_client(self):
        dataset = LabeledDataset([0, 1, 1, 2, 2, 2, 10])

        manifest = build_partition_manifest(
            dataset, "organamnist", num_clients=3, seed=42
        )

        self.assertEqual(manifest["total_training_examples"], 7)
        self.assertEqual(
            manifest["verification"],
            {
                "partitions_are_disjoint": True,
                "all_training_examples_assigned": True,
                "each_training_example_assigned_once": True,
            },
        )
        self.assertEqual(
            sum(client["num_images"] for client in manifest["clients"]), 7
        )
        class_names = get_class_names("organamnist")
        aggregate_counts = {
            class_name: sum(
                client["class_counts"][class_name]
                for client in manifest["clients"]
            )
            for class_name in class_names
        }
        self.assertEqual(aggregate_counts[class_names[0]], 1)
        self.assertEqual(aggregate_counts[class_names[1]], 2)
        self.assertEqual(aggregate_counts[class_names[2]], 3)
        self.assertEqual(aggregate_counts[class_names[10]], 1)


if __name__ == "__main__":
    unittest.main()
