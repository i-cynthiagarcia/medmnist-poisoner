import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch
from flwr.app import ArrayRecord, MetricRecord

from pytorchexample.artifacts import resolve_protocol_id
from pytorchexample.server_app import (
    save_round_checkpoint,
    serialize_evaluation_metrics,
)
from recompute_validation_metrics import metrics_match


class ServerOutputTests(unittest.TestCase):
    def test_protocol_id_is_validated_or_generated(self):
        configured_id = "protocol_20261115T090141.123456Z"
        self.assertEqual(resolve_protocol_id(configured_id), configured_id)
        self.assertRegex(resolve_protocol_id("auto"), r"^protocol_[0-9TZ.]+$")
        with self.assertRaises(ValueError):
            resolve_protocol_id("../protocol-escape")

    def test_metric_verification_detects_changed_values(self):
        expected = {"loss": 0.25, "matrix": [[1, 0], [0, 1]]}

        self.assertTrue(
            metrics_match(
                expected,
                {"loss": 0.25000000001, "matrix": [[1, 0], [0, 1]]},
            )
        )
        self.assertFalse(
            metrics_match(expected, {"loss": 0.25, "matrix": [[0, 1], [1, 0]]})
        )

    def test_metrics_are_serialized_with_class_names_and_matrix_axes(self):
        metrics = MetricRecord(
            {
                "loss": 0.25,
                "accuracy": 0.75,
                "per_class_precision": [0.5, 1.0],
                "per_class_recall": [1.0, 0.5],
                "per_class_f1": [2 / 3, 2 / 3],
                "per_class_support": [2, 2],
                "confusion_matrix": [1, 1, 0, 1],
            }
        )

        serialized = serialize_evaluation_metrics(metrics, ["left", "right"])

        self.assertEqual(serialized["per_class"]["left"]["support"], 2)
        self.assertAlmostEqual(serialized["per_class"]["right"]["precision"], 1.0)
        self.assertEqual(serialized["confusion_matrix"], [[1, 1], [0, 1]])

    def test_round_checkpoint_is_saved_under_seed_run_and_round(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = Mock()
            context.run_id = "abc123"
            context.run_config = {
                "dataset": "organamnist",
                "seed": 42,
                "num-server-rounds": 3,
                "local-epochs": 1,
                "learning-rate": 0.1,
                "models-dir": temp_dir,
            }
            arrays = ArrayRecord({"weight": torch.tensor([1.0])})

            checkpoint_path = save_round_checkpoint(
                context, arrays, 2, "protocol_test"
            )

            self.assertEqual(
                checkpoint_path,
                Path(temp_dir)
                / "protocol_test"
                / "organamnist"
                / "baseline_r3_e1_lr0p100"
                / "checkpoints"
                / "seed-42_run-abc123"
                / "round-002.pt",
            )
            saved_state = torch.load(checkpoint_path, weights_only=True)
            self.assertTrue(torch.equal(saved_state["weight"], torch.tensor([1.0])))

    def test_repeated_round_checkpoint_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = Mock()
            context.run_id = "abc123"
            context.run_config = {
                "dataset": "organamnist",
                "seed": 42,
                "num-server-rounds": 3,
                "local-epochs": 1,
                "learning-rate": 0.1,
                "models-dir": temp_dir,
            }
            arrays = ArrayRecord({"weight": torch.tensor([1.0])})
            checkpoint_path = save_round_checkpoint(
                context, arrays, 2, "protocol_test"
            )

            with self.assertRaises(FileExistsError):
                save_round_checkpoint(context, arrays, 2, "protocol_test")

            saved_state = torch.load(checkpoint_path, weights_only=True)
            self.assertTrue(torch.equal(saved_state["weight"], torch.tensor([1.0])))


if __name__ == "__main__":
    unittest.main()
