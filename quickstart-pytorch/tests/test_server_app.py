import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch
from flwr.app import ArrayRecord, MetricRecord

from pytorchexample.shared.artifacts import (
    resolve_execution_id,
    resolve_protocol_id,
    validate_generation_id,
)
from pytorchexample.experiments.shared.server import (
    _contains_only_finite_values,
    save_partition_manifest,
    save_round_checkpoint,
    targeted_source_to_target_metrics,
)
from pytorchexample.shared.metrics import serialize_evaluation_metrics
from pytorchexample.shared.model import Net
from recompute_validation_metrics import metrics_match


class ServerOutputTests(unittest.TestCase):
    def test_run_values_reject_non_finite_metrics_and_parameters(self):
        finite_values = {
            "loss": 0.25,
            "confusion_matrix": [[1, 0], [0, 1]],
            "weights": torch.tensor([1.0, -2.0]),
        }
        self.assertTrue(_contains_only_finite_values(finite_values))
        self.assertFalse(_contains_only_finite_values({"loss": float("nan")}))
        self.assertFalse(
            _contains_only_finite_values(
                {"weights": torch.tensor([1.0, float("inf")])}
            )
        )

    def test_targeted_misclassification_rate_uses_source_row_support(self):
        metrics = {"confusion_matrix": [[4, 3, 1], [1, 6, 0], [0, 2, 5]]}

        targeted = targeted_source_to_target_metrics(metrics, 0, 1)

        self.assertEqual(targeted, {"rate": 3 / 8, "support": 8})

    def test_partition_manifest_is_saved_for_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = Mock()
            context.run_id = "abc123"
            context.run_config = {
                "dataset": "organcmnist",
                "seed": 42,
                "num-server-rounds": 3,
                "local-epochs": 1,
                "learning-rate": 0.1,
                "results-dir": temp_dir,
            }
            manifest = {
                "dataset": "organcmnist",
                "seed": 42,
                "num_clients": 2,
                "total_training_examples": 5,
                "verification": {
                    "partitions_are_disjoint": True,
                    "all_training_examples_assigned": True,
                    "each_training_example_assigned_once": True,
                },
                "clients": [
                    {
                        "partition_id": 0,
                        "num_images": 3,
                        "class_counts": {"class-a": 2, "class-b": 1},
                    },
                    {
                        "partition_id": 1,
                        "num_images": 2,
                        "class_counts": {"class-a": 1, "class-b": 1},
                    },
                ],
            }

            path = save_partition_manifest(
                context, manifest, "generation-v3", "protocol-v1", "run-1"
            )

            self.assertEqual(
                path,
                Path(temp_dir)
                / "generation-v3"
                / "protocol-v1"
                / "run-1"
                / "organcmnist"
                / "baseline_r3_e1_lr0p100"
                / "partition-seed-42_run-abc123.json",
            )
            self.assertEqual(
                json.loads(path.read_text())["clients"], manifest["clients"]
            )
            with self.assertRaises(FileExistsError):
                save_partition_manifest(
                    context,
                    manifest,
                    "generation-v3",
                    "protocol-v1",
                    "run-1",
                )

    def test_protocol_id_is_validated_or_defaulted(self):
        configured_id = "protocol-v3"
        self.assertEqual(resolve_protocol_id(configured_id), configured_id)
        self.assertEqual(resolve_protocol_id("auto"), "protocol-v1")
        with self.assertRaises(ValueError):
            resolve_protocol_id("../protocol-escape")
        with self.assertRaises(ValueError):
            resolve_protocol_id("protocol_custom")

    def test_execution_id_auto_advances_across_result_and_model_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results_root = Path(temp_dir) / "results"
            models_root = Path(temp_dir) / "models"
            (results_root / "generation-v3" / "protocol-v1" / "run-1").mkdir(
                parents=True
            )
            (models_root / "generation-v3" / "protocol-v1" / "run-3").mkdir(
                parents=True
            )

            self.assertEqual(
                resolve_execution_id(
                    "auto",
                    results_root,
                    models_root,
                    "generation-v3",
                    "protocol-v1",
                ),
                "run-4",
            )
            self.assertEqual(
                resolve_execution_id(
                    "run-7",
                    results_root,
                    models_root,
                    "generation-v3",
                    "protocol-v1",
                ),
                "run-7",
            )
            with self.assertRaises(ValueError):
                resolve_execution_id(
                    "../run-escape",
                    results_root,
                    models_root,
                    "generation-v3",
                    "protocol-v1",
                )

    def test_generation_id_must_follow_versioned_directory_format(self):
        self.assertEqual(validate_generation_id("generation-v3"), "generation-v3")
        with self.assertRaises(ValueError):
            validate_generation_id("../generation-v3")

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
                context,
                arrays,
                2,
                "generation-v3",
                "protocol-v1",
                "run-1",
            )

            self.assertEqual(
                checkpoint_path,
                Path(temp_dir)
                / "generation-v3"
                / "protocol-v1"
                / "run-1"
                / "organamnist"
                / "baseline_r3_e1_lr0p100"
                / "checkpoints"
                / "seed-42_run-abc123"
                / "round-002.pt",
            )
            saved_state = torch.load(checkpoint_path, weights_only=True)
            self.assertTrue(torch.equal(saved_state["weight"], torch.tensor([1.0])))

    def test_round_checkpoint_loads_into_model_with_matching_outputs(self):
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
            original_model = Net().eval()
            arrays = ArrayRecord(original_model.state_dict())

            checkpoint_path = save_round_checkpoint(
                context,
                arrays,
                2,
                "generation-v3",
                "protocol-v1",
                "run-1",
            )

            restored_model = Net().eval()
            restored_model.load_state_dict(
                torch.load(checkpoint_path, weights_only=True), strict=True
            )
            sample = torch.zeros((2, 1, 28, 28))
            with torch.no_grad():
                original_output = original_model(sample)
                restored_output = restored_model(sample)

            torch.testing.assert_close(restored_output, original_output)

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
                context,
                arrays,
                2,
                "generation-v3",
                "protocol-v1",
                "run-1",
            )

            with self.assertRaises(FileExistsError):
                save_round_checkpoint(
                    context,
                    arrays,
                    2,
                    "generation-v3",
                    "protocol-v1",
                    "run-1",
                )

            saved_state = torch.load(checkpoint_path, weights_only=True)
            self.assertTrue(torch.equal(saved_state["weight"], torch.tensor([1.0])))


if __name__ == "__main__":
    unittest.main()
