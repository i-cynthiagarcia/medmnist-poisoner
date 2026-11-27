import unittest

import torch
from torch.utils.data import Subset, TensorDataset

from pytorchexample.shared.data import (
    build_partition_manifest,
    get_class_names,
    verify_partition_indices,
)
from pytorchexample.experiments.poisoning.label_flipping import (
    LabelFlippedDataset,
    make_label_flip_transform,
    select_flipped_positions,
    select_poisoned_clients,
    validate_label_flip_config,
)


class LabeledDataset:
    def __init__(self, labels):
        self.labels = torch.tensor(labels).reshape(-1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor([index]), self.labels[index]


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


class LabelFlippingTests(unittest.TestCase):
    def test_selects_seeded_fraction_only_from_source_class(self):
        labels = torch.tensor([1, 0, 1, 1, 2, 1])

        selected = select_flipped_positions(labels, 1, 0.5, seed=7)

        self.assertEqual(len(selected), 2)
        self.assertTrue(all(labels[index].item() == 1 for index in selected))
        self.assertEqual(
            selected, select_flipped_positions(labels, 1, 0.5, seed=7)
        )

    def test_label_view_changes_only_selected_labels(self):
        labels = torch.tensor([1, 0, 1])
        dataset = TensorDataset(torch.zeros((3, 1)), labels)

        poisoned = LabelFlippedDataset(dataset, [0, 2], target_class_id=2)
        observed = [poisoned[index][1].item() for index in range(len(poisoned))]

        self.assertEqual(observed, [2, 0, 2])
        self.assertEqual(labels.tolist(), [1, 0, 1])
        self.assertEqual(poisoned.num_flipped, 2)

    def test_transform_only_poisons_selected_clients_source_labels(self):
        dataset = LabeledDataset([1, 0, 1])
        client_subset = Subset(dataset, [0, 1, 2])
        transform = make_label_flip_transform([1], 1, 2, 1.0, seed=9)

        clean_client = transform(client_subset, 0)
        poisoned_client = transform(client_subset, 1)

        self.assertIs(clean_client, client_subset)
        self.assertEqual(
            [poisoned_client[index][1].item() for index in range(3)],
            [2, 0, 2],
        )
        self.assertEqual(dataset.labels.reshape(-1).tolist(), [1, 0, 1])

    def test_full_flip_replay_is_deterministic_for_organ_c_and_s(self):
        labels = [4, 0, 5, 4, 6, 4]
        expected = [5, 0, 5, 5, 6, 5]

        for dataset_name in ("organcmnist", "organsmnist"):
            self.assertEqual(
                get_class_names(dataset_name)[4:6],
                ["kidney-left", "kidney-right"],
            )
            for seed in range(42, 47):
                with self.subTest(dataset=dataset_name, seed=seed):
                    dataset = LabeledDataset(labels)
                    subset = Subset(dataset, list(range(len(labels))))
                    poisoned_client_id = select_poisoned_clients(
                        20, 1, seed
                    )[0]
                    transform = make_label_flip_transform(
                        [poisoned_client_id], 4, 5, 1.0, seed
                    )

                    poisoned = transform(subset, poisoned_client_id)
                    replayed = [
                        poisoned[index][1].item()
                        for index in range(len(poisoned))
                    ]
                    self.assertEqual(replayed, expected)
                    self.assertEqual(
                        dataset.labels.reshape(-1).tolist(), labels
                    )
                    self.assertIs(
                        transform(subset, (poisoned_client_id + 1) % 20),
                        subset,
                    )

                    replayed_again = transform(subset, poisoned_client_id)
                    self.assertEqual(
                        [
                            replayed_again[index][1].item()
                            for index in range(len(replayed_again))
                        ],
                        replayed,
                    )

    def test_poisoned_client_selection_is_seeded_and_has_requested_size(self):
        selected = select_poisoned_clients(10, 3, seed=42)

        self.assertEqual(len(selected), 3)
        self.assertEqual(selected, select_poisoned_clients(10, 3, seed=42))
        self.assertEqual(selected, sorted(selected))

    def test_configuration_rejects_invalid_class_pair_and_fraction(self):
        with self.assertRaisesRegex(ValueError, "must differ"):
            validate_label_flip_config(5, 1, 0.5, 2, 2, 11)
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            validate_label_flip_config(5, 1, 1.1, 2, 3, 11)


if __name__ == "__main__":
    unittest.main()
