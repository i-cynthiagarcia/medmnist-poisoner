import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pytorchexample.experiments.baseline import centralized


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(1, 11)

    def forward(self, images):
        features = images.mean(dim=(2, 3))
        return self.classifier(features)


class CentralizedBaselineTests(unittest.TestCase):
    def test_runner_saves_separate_centralized_artifacts(self):
        images = torch.ones((2, 1, 28, 28))
        labels = torch.tensor([0, 1])
        dataset = TensorDataset(images, labels)
        validation_loader = DataLoader(dataset, batch_size=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir) / "results"
            models_dir = Path(temp_dir) / "models"
            argv = [
                "centralized.py",
                "--dataset",
                "organamnist",
                "--seed",
                "42",
                "--num-rounds",
                "1",
                "--local-epochs",
                "1",
                "--learning-rate",
                "0.01",
                "--batch-size",
                "2",
                "--generation-id",
                "generation-v3",
                "--protocol-id",
                "protocol_20261115T103000Z",
                "--results-dir",
                str(results_dir),
                "--models-dir",
                str(models_dir),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(centralized, "Net", TinyModel),
                patch.object(centralized, "get_dataset", return_value=dataset),
                patch.object(
                    centralized,
                    "load_centralized_dataset",
                    return_value=validation_loader,
                ),
                patch.object(
                    centralized.torch.cuda,
                    "is_available",
                    return_value=False,
                ),
            ):
                centralized.main()

            experiment = "centralized_r1_e1_lr0p010"
            stem = "seed-42_run-centralized"
            result_path = (
                results_dir
                / "generation-v3"
                / "protocol_20261115T103000Z"
                / "organamnist"
                / experiment
                / f"{stem}.json"
            )
            model_path = (
                models_dir
                / "generation-v3"
                / "protocol_20261115T103000Z"
                / "organamnist"
                / experiment
                / f"{stem}.pt"
            )
            record = json.loads(result_path.read_text())

            self.assertTrue(model_path.is_file())
            self.assertEqual(record["training_mode"], "centralized")
            self.assertEqual(record["model_type"], "centralized_full_train")
            self.assertIsNone(record["aggregation"])
            self.assertEqual(record["configuration"]["training_examples"], 2)
            self.assertEqual(record["configuration"]["epochs"], 1)
            self.assertEqual(set(record["metrics"]["validation_by_round"]), {"0", "1"})


if __name__ == "__main__":
    unittest.main()
