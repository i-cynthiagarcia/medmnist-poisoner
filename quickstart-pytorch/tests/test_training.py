import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pytorchexample.shared.model import Net
from pytorchexample.shared.training import evaluate, test, train, train_epoch


class ImageLogitModel(nn.Module):
    """Produce deterministic two-class logits from each input image value."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, images):
        values = images.view(-1)
        logits = torch.stack((torch.zeros_like(values), values), dim=1)
        return logits + self.anchor * 0.0


class PredictedClassModel(nn.Module):
    """Predict the class encoded in each input image."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, images):
        predicted_classes = images.view(-1).long()
        logits = torch.full(
            (len(predicted_classes), 11), -5.0, device=images.device
        )
        logits.scatter_(1, predicted_classes.unsqueeze(1), 5.0)
        return logits + self.anchor * 0.0


class ConstantClassModel(nn.Module):
    """Always predict one class for a majority-class reference."""

    def __init__(self, predicted_class, num_classes=11):
        super().__init__()
        self.register_buffer(
            "logits",
            torch.nn.functional.one_hot(
                torch.tensor(predicted_class), num_classes
            ).float(),
        )

    def forward(self, images):
        return self.logits.expand(len(images), -1)


class TrainingTests(unittest.TestCase):
    def test_train_loss_weights_partial_batch_by_image_count(self):
        images = torch.tensor([[0.0], [0.0], [10.0]])
        labels = torch.tensor([0, 0, 0])
        loader = DataLoader(
            TensorDataset(images, labels), batch_size=2, shuffle=False
        )
        model = ImageLogitModel()
        expected_loss = F.cross_entropy(
            model(images), labels, reduction="mean"
        ).item()

        actual_loss = train(model, loader, epochs=1, lr=0.1, device="cpu")

        self.assertAlmostEqual(actual_loss, expected_loss, places=6)

    def test_train_epoch_uses_the_supplied_optimizer(self):
        images = torch.tensor([[0.0], [1.0], [2.0]])
        labels = torch.tensor([0, 0, 1])
        loader = DataLoader(
            TensorDataset(images, labels), batch_size=2, shuffle=False
        )
        model = ImageLogitModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        epoch_loss = train_epoch(model, loader, optimizer, device="cpu")

        self.assertGreater(epoch_loss, 0)
        with self.assertRaisesRegex(ValueError, "empty data loader"):
            train_epoch(
                model,
                DataLoader(
                    TensorDataset(
                        torch.empty((0, 1)), torch.empty((0,), dtype=torch.long)
                    ),
                    batch_size=2,
                ),
                optimizer,
                device="cpu",
            )

    def test_evaluation_reports_image_weighted_loss_and_class_metrics(self):
        images = torch.tensor([[0.0], [1.0], [1.0]])
        labels = torch.tensor([0, 0, 1])
        loader = DataLoader(
            TensorDataset(images, labels), batch_size=2, shuffle=False
        )
        model = PredictedClassModel()
        expected_loss = F.cross_entropy(
            model(images), labels, reduction="mean"
        ).item()

        metrics = evaluate(model, loader, device="cpu", dataset="organamnist")
        legacy_loss, legacy_accuracy = test(
            model, loader, device="cpu", dataset="organamnist"
        )

        self.assertAlmostEqual(metrics["loss"], expected_loss, places=6)
        self.assertAlmostEqual(legacy_loss, expected_loss, places=6)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3, places=6)
        self.assertAlmostEqual(legacy_accuracy, 2 / 3, places=6)
        self.assertEqual(metrics["per_class_support"][:3], [2, 1, 0])
        self.assertAlmostEqual(metrics["per_class_precision"][0], 1.0)
        self.assertAlmostEqual(metrics["per_class_recall"][0], 0.5)
        self.assertAlmostEqual(metrics["per_class_f1"][0], 2 / 3)
        self.assertEqual(len(metrics["confusion_matrix"]), 11 * 11)
        self.assertEqual(metrics["confusion_matrix"][:3], [1, 1, 0])
        self.assertEqual(metrics["confusion_matrix"][11 : 11 + 3], [0, 1, 0])

    def test_evaluation_accuracy_matches_majority_class_reference(self):
        images = torch.zeros((8, 1))
        labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2])
        loader = DataLoader(
            TensorDataset(images, labels), batch_size=3, shuffle=False
        )
        majority_reference = (
            torch.bincount(labels).max().item() / len(labels)
        )

        metrics = evaluate(
            ConstantClassModel(predicted_class=0),
            loader,
            device="cpu",
            dataset="organamnist",
        )

        self.assertEqual(majority_reference, 5 / 8)
        self.assertAlmostEqual(metrics["accuracy"], majority_reference)

    def test_shared_training_can_overfit_a_small_image_subset(self):
        images = torch.zeros((8, 1, 28, 28))
        images[:4, 0, 3:11, 3:11] = 1.0
        images[4:, 0, 17:25, 17:25] = 1.0
        labels = torch.tensor([0] * 4 + [1] * 4)
        dataset = TensorDataset(images, labels)
        evaluation_loader = DataLoader(dataset, batch_size=8, shuffle=False)
        trainloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
        )
        majority_reference = 4 / len(labels)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = Net()
            initial_metrics = evaluate(
                model, evaluation_loader, device="cpu", dataset="organamnist"
            )
            train(
                model,
                trainloader,
                epochs=10,
                lr=0.1,
                device="cpu",
            )
            final_metrics = evaluate(
                model, evaluation_loader, device="cpu", dataset="organamnist"
            )

        self.assertGreaterEqual(final_metrics["accuracy"], 0.99)
        self.assertGreater(final_metrics["accuracy"], majority_reference)
        self.assertLess(final_metrics["loss"], initial_metrics["loss"])


if __name__ == "__main__":
    unittest.main()
