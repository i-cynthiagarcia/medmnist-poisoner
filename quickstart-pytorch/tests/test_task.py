import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pytorchexample.task import evaluate, set_dataset, test, train


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


class TaskMetricsTests(unittest.TestCase):
    def setUp(self):
        set_dataset("organamnist")

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

        metrics = evaluate(model, loader, device="cpu")
        legacy_loss, legacy_accuracy = test(model, loader, device="cpu")

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


if __name__ == "__main__":
    unittest.main()
