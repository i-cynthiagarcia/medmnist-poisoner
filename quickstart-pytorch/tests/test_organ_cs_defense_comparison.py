import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_organ_cs_defense_comparison as comparison
import run_organa_defense_sweep as sweep
from pytorchexample.experiments.poisoning.label_flipping import select_poisoned_clients
from pytorchexample.shared.data import get_class_names


def clean_record(accuracy=0.8, recall=0.5):
    return {
        "metrics": {"validation_by_round": {"5": {
            "accuracy": accuracy, "per_class": {"kidney-left": {"recall": recall}}
        }}}
    }


class OrganCSComparisonTests(unittest.TestCase):
    def test_runs_80_fresh_conditions_and_reports_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outcomes = [SimpleNamespace(returncode=7 if i == 0 else 0) for i in range(80)]
            checksums = {dataset: "checksum" for dataset in comparison.DATASETS}
            with (
                mock.patch.object(comparison, "RESULTS", root / "results"),
                mock.patch.object(comparison, "MODELS", root / "models"),
                mock.patch.object(comparison, "comparison_environment", return_value={"git_revision": "revision"}),
                mock.patch.object(comparison, "prepare_datasets", return_value=checksums),
                mock.patch.object(comparison, "verify_environment"),
                mock.patch.object(comparison.subprocess, "run", side_effect=outcomes) as run,
                mock.patch.object(comparison, "validate_condition", side_effect=[ValueError("missing checkpoint")] + [clean_record()] * 78),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(comparison.main(), 1)
            self.assertEqual(run.call_count, 80)
            conditions = set()
            for call in run.call_args_list:
                command = call.args[0]
                config = command[command.index("--run-config") + 1]
                for setting in (
                    "evaluate-test=false", "require-complete-round=true", "num-clients=20",
                    "num-server-rounds=5", "local-epochs=3", "learning-rate=0.01",
                    "batch-size=32", "poison-source-class-id=4", "poison-target-class-id=5",
                    'protocol-id="protocol-v8"',
                ):
                    self.assertIn(setting, config)
                conditions.add(tuple(part for part in config.split() if part.startswith(
                    ("dataset=", "seed=", "aggregation=", "poison-num-clients=", "poison-fraction=")
                )))
            expected = {
                (f'aggregation="{aggregation}"', f'dataset="{dataset}"', f"seed={seed}",
                 f"poison-num-clients={count}", f"poison-fraction={fraction}")
                for dataset in ("organcmnist", "organsmnist")
                for seed in range(42, 52)
                for aggregation in ("fedavg", "median")
                for count, fraction in ((0, "0.0"), (4, "1.0"))
            }
            self.assertEqual(conditions, expected)
            self.assertIn("Flower exited 7", stderr.getvalue())
            self.assertIn("missing checkpoint", stderr.getvalue())
            first = json.loads((root / "results/run-1/comparison-outcome.json").read_text())
            second = json.loads((root / "results/run-2/comparison-outcome.json").read_text())
            self.assertEqual(first["attempted"], 40)
            self.assertEqual(len(first["failures"]), 2)
            self.assertFalse(first["complete"])
            self.assertFalse(first["clean_summary"]["fedavg"]["complete"])
            self.assertTrue(second["complete"])
            manifest = json.loads((root / "results/run-2/comparison-manifest.json").read_text())
            self.assertEqual(manifest["dataset"], "organsmnist")
            self.assertFalse(manifest["evaluate_test"])
            self.assertEqual(manifest["clean_floors"], {"accuracy": 0.55, "kidney_left_recall": 0.25})

    def test_provenance_drift_stops_and_persists_partial_outcome(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (
                mock.patch.object(comparison, "RESULTS", root / "results"),
                mock.patch.object(comparison, "MODELS", root / "models"),
                mock.patch.object(comparison, "comparison_environment", return_value={}),
                mock.patch.object(comparison, "prepare_datasets", return_value={d: "checksum" for d in comparison.DATASETS}),
                mock.patch.object(comparison, "verify_environment", side_effect=[None, None, ValueError("changed")]),
                mock.patch.object(comparison.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run,
                mock.patch.object(comparison, "validate_condition", return_value=clean_record()),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(comparison.main(), 1)
            self.assertEqual(run.call_count, 1)
            outcome = json.loads((root / "results/run-1/comparison-outcome.json").read_text())
            self.assertEqual(outcome["attempted"], 1)
            self.assertFalse(outcome["complete"])
            self.assertIn("remaining conditions not attempted", stderr.getvalue())

    def test_floors_use_all_ten_unrounded_values(self):
        records = {
            (seed, aggregation, 0): clean_record()
            for seed in comparison.SEEDS for aggregation in comparison.AGGREGATIONS
        }
        summary = comparison.clean_summary("organcmnist", records)
        self.assertTrue(summary["fedavg"]["accuracy_floor_met"])
        self.assertTrue(summary["median"]["kidney_left_recall_floor_met"])
        records[42, "fedavg", 0] = clean_record(0.7999, 0.4999)
        summary = comparison.clean_summary("organcmnist", records)
        self.assertFalse(summary["fedavg"]["accuracy_floor_met"])
        self.assertFalse(summary["fedavg"]["kidney_left_recall_floor_met"])
        del records[43, "median", 0]
        self.assertEqual(comparison.clean_summary("organcmnist", records)["median"], {"complete": False})

    def test_condition_verifies_partitions_attack_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selected = select_poisoned_clients(20, 4, 42)
            clients = [
                {"partition_id": i, "class_counts": {"kidney-left": i + 1}}
                for i in range(20)
            ]
            (root / "partition.json").write_text(json.dumps({
                "dataset": "organcmnist", "seed": 42, "num_clients": 20,
                "verification": {
                    "partitions_are_disjoint": True,
                    "all_training_examples_assigned": True,
                    "each_training_example_assigned_once": True,
                },
                "clients": clients,
            }))
            (root / "model.pt").touch()
            names = get_class_names("organcmnist")
            flipped = {str(i): i + 1 for i in selected}
            record = {
                "generation_id": "generation-v1", "protocol_id": "protocol-v8",
                "execution_id": "run-1", "status": "completed", "dataset": "organcmnist",
                "seed": 42, "aggregation_id": "fedavg", "aggregation": "FedAvg",
                "artifact_layout": "aggregation_subdirectories",
                "partition_manifest": "partition.json", "model_artifact": "model.pt",
                "round_checkpoints": {str(i): "model.pt" for i in range(1, 6)},
                "metrics": {
                    "test_final": None, "class_names": names,
                    "validation_by_round": {str(i): {
                        "accuracy": 0.8, "loss": 0.5,
                        "per_class": {name: {"recall": 0.5} for name in names},
                    } for i in range(6)},
                    "targeted_source_to_target_misclassification": {
                        "validation_by_round": {str(i): {"rate": 0.2, "support": 4} for i in range(6)},
                        "test_final": None,
                    },
                },
                "training_results": {
                    "valid_run": True, "metrics_finite": True, "model_parameters_finite": True,
                    "received_results_by_round": {str(i): 20 for i in range(1, 6)},
                    "client_failures_by_round": {str(i): 0 for i in range(1, 6)},
                },
                "configuration": {
                    "clients": 20, "federated_rounds": 5, "local_epochs": 3,
                    "learning_rate": 0.01, "batch_size": 32, "require_complete_round": True,
                    "label_flipping": {
                        "num_poisoned_clients": 4,
                        "fraction_of_source_labels_flipped_per_poisoned_client": 1.0,
                        "source_class_id": 4, "target_class_id": 5,
                        "poisoned_client_ids": selected,
                        "flipped_source_examples_by_client": flipped,
                        "total_flipped_source_examples": sum(flipped.values()),
                    },
                },
            }
            directory = root / "results/run-1/fedavg/organcmnist/label_flip_r5_e3_lr0p010_clients4_f1p000_s4_t5"
            directory.mkdir(parents=True)
            path = directory / "seed-42_run-1.json"
            path.write_text(json.dumps(record))
            with (
                mock.patch.object(comparison, "ROOT", root),
                mock.patch.object(comparison, "RESULTS", root / "results"),
                mock.patch.object(sweep, "ROOT", root),
            ):
                comparison.validate_condition("organcmnist", "run-1", 42, "fedavg", 4, {})
                with self.assertRaisesRegex(ValueError, "Partition summaries differ"):
                    comparison.validate_condition("organcmnist", "run-1", 42, "fedavg", 4, {42: []})
                record["configuration"]["label_flipping"]["total_flipped_source_examples"] += 1
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "full-flip counts"):
                    comparison.validate_condition("organcmnist", "run-1", 42, "fedavg", 4, {})
                record["round_checkpoints"]["5"] = "missing.pt"
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "Missing checkpoint"):
                    comparison.validate_condition("organcmnist", "run-1", 42, "fedavg", 4, {})


if __name__ == "__main__":
    unittest.main()
