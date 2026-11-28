import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_organ_cs_defense_pilot as pilot
import run_organa_defense_sweep as sweep
from pytorchexample.shared.data import get_class_names


class OrganCSPilotTests(unittest.TestCase):
    def test_runs_all_24_conditions_and_reports_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outcomes = [SimpleNamespace(returncode=7 if i == 0 else 0) for i in range(24)]
            checksums = {dataset: "checksum" for dataset in pilot.DATASETS}
            with (
                mock.patch.object(pilot, "RESULTS", root / "results"),
                mock.patch.object(pilot, "MODELS", root / "models"),
                mock.patch.object(pilot, "capture_environment", return_value={"git_revision": "revision"}),
                mock.patch.object(pilot, "prepare_datasets", return_value=checksums),
                mock.patch.object(pilot, "verify_environment"),
                mock.patch.object(pilot.subprocess, "run", side_effect=outcomes) as run,
                mock.patch.object(pilot, "validate_condition", side_effect=[ValueError("missing checkpoint")] + [None] * 22),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(pilot.main(), 1)
            self.assertEqual(run.call_count, 24)
            conditions = set()
            for call in run.call_args_list:
                command = call.args[0]
                config = command[command.index("--run-config") + 1]
                for setting in (
                    "evaluate-test=false", "require-complete-round=true", "num-clients=20",
                    "num-server-rounds=5", "local-epochs=1", "learning-rate=0.01",
                    "batch-size=32", "poison-source-class-id=4", "poison-target-class-id=5",
                ):
                    self.assertIn(setting, config)
                conditions.add(tuple(part for part in config.split() if part.startswith(
                    ("dataset=", "seed=", "aggregation=", "poison-num-clients=", "poison-fraction=")
                )))
            expected = {
                (f'aggregation="{aggregation}"', f'dataset="{dataset}"', f"seed={seed}",
                 f"poison-num-clients={count}", f"poison-fraction={fraction}")
                for dataset in ("organcmnist", "organsmnist")
                for seed in (42, 49, 50)
                for aggregation in ("fedavg", "median")
                for count, fraction in ((0, "0.0"), (4, "1.0"))
            }
            self.assertEqual(conditions, expected)
            self.assertIn("Flower exited 7", stderr.getvalue())
            self.assertIn("missing checkpoint", stderr.getvalue())
            first = json.loads((root / "results/run-1/pilot-outcome.json").read_text())
            second = json.loads((root / "results/run-2/pilot-outcome.json").read_text())
            self.assertEqual(first["attempted"], 12)
            self.assertEqual(len(first["failures"]), 2)
            self.assertTrue(second["complete"])
            manifest = json.loads((root / "results/run-2/pilot-manifest.json").read_text())
            self.assertEqual(manifest["dataset"], "organsmnist")
            self.assertFalse(manifest["evaluate_test"])

    def test_prepares_and_verifies_both_archives(self):
        from medmnist import INFO

        with (
            mock.patch("pytorchexample.shared.data.get_dataset") as load,
            mock.patch.object(pilot, "archive_checksum", side_effect=lambda dataset: INFO[dataset]["MD5"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            checksums = pilot.prepare_datasets()
        self.assertEqual(set(checksums), set(pilot.DATASETS))
        self.assertEqual(load.call_args_list, [
            mock.call("train", "organcmnist"), mock.call("train", "organsmnist")
        ])
        with (
            mock.patch("pytorchexample.shared.data.get_dataset"),
            mock.patch.object(pilot, "archive_checksum", return_value="wrong"),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, "archive checksum"),
        ):
            pilot.prepare_datasets()

    def test_provenance_drift_stops_and_records_partial_outcome(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (
                mock.patch.object(pilot, "RESULTS", root / "results"),
                mock.patch.object(pilot, "MODELS", root / "models"),
                mock.patch.object(pilot, "capture_environment", return_value={}),
                mock.patch.object(pilot, "prepare_datasets", return_value={d: "checksum" for d in pilot.DATASETS}),
                mock.patch.object(pilot, "verify_environment", side_effect=[None, ValueError("changed")]),
                mock.patch.object(pilot.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run,
                mock.patch.object(pilot, "validate_condition"),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(pilot.main(), 1)
            self.assertEqual(run.call_count, 1)
            outcome = json.loads((root / "results/run-1/pilot-outcome.json").read_text())
            self.assertFalse(outcome["complete"])
            self.assertEqual(outcome["attempted"], 1)
            self.assertIn("remaining conditions not attempted", stderr.getvalue())

    def test_condition_checks_dataset_partitions_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "model.pt").touch()
            (root / "partition.json").write_text(json.dumps({
                "verification": {
                    "partitions_are_disjoint": True,
                    "all_training_examples_assigned": True,
                    "each_training_example_assigned_once": True,
                },
                "clients": [{"partition_id": 0}],
            }))
            record = {
                "generation_id": "generation-v1", "protocol_id": "protocol-v5",
                "execution_id": "run-1", "status": "completed", "dataset": "organcmnist",
                "seed": 42, "aggregation_id": "fedavg", "aggregation": "FedAvg",
                "partition_manifest": "partition.json", "model_artifact": "model.pt",
                "round_checkpoints": {str(i): "model.pt" for i in range(1, 6)},
                "metrics": {
                    "test_final": None,
                    "class_names": get_class_names("organcmnist"),
                    "validation_by_round": {"5": {
                        "accuracy": 0.5, "loss": 1.0,
                        "per_class": {
                            name: {"recall": 0.5} for name in get_class_names("organcmnist")
                        },
                    }},
                    "targeted_source_to_target_misclassification": {
                        "validation_by_round": {"5": {"rate": 0.25, "support": 4}},
                        "test_final": None,
                    },
                },
                "training_results": {
                    "valid_run": True, "metrics_finite": True, "model_parameters_finite": True,
                    "received_results_by_round": {str(i): 20 for i in range(1, 6)},
                    "client_failures_by_round": {str(i): 0 for i in range(1, 6)},
                },
                "configuration": {
                    "clients": 20, "federated_rounds": 5, "local_epochs": 1,
                    "learning_rate": 0.01, "batch_size": 32, "require_complete_round": True,
                    "label_flipping": {
                        "num_poisoned_clients": 0,
                        "fraction_of_source_labels_flipped_per_poisoned_client": 0.0,
                        "source_class_id": 4, "target_class_id": 5,
                    },
                },
            }
            results = root / "results"
            directory = results / "run-1/fedavg/organcmnist/label_flip_r5_e1_lr0p010_clients0_f0p000_s4_t5"
            directory.mkdir(parents=True)
            path = directory / "seed-42_run-1.json"
            path.write_text(json.dumps(record))
            with mock.patch.object(pilot, "RESULTS", results), mock.patch.object(sweep, "ROOT", root):
                pilot.validate_condition("organcmnist", "run-1", 42, "fedavg", 0, {}, {})
                with self.assertRaisesRegex(ValueError, "Partition summaries differ"):
                    pilot.validate_condition("organcmnist", "run-1", 42, "fedavg", 0, {42: []}, {})
                record["dataset"] = "organsmnist"
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "Invalid condition"):
                    pilot.validate_condition("organcmnist", "run-1", 42, "fedavg", 0, {}, {})
                record["dataset"] = "organcmnist"
                del record["metrics"]["validation_by_round"]["5"]["per_class"]["kidney-left"]
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "invalid final validation metrics"):
                    pilot.validate_condition("organcmnist", "run-1", 42, "fedavg", 0, {}, {})
                record["round_checkpoints"]["5"] = "missing.pt"
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "Missing checkpoint"):
                    pilot.validate_condition("organcmnist", "run-1", 42, "fedavg", 0, {}, {})
                record["round_checkpoints"]["5"] = "model.pt"
                record["metrics"]["test_final"] = {"accuracy": 1.0}
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "Invalid condition"):
                    pilot.validate_condition("organcmnist", "run-1", 42, "fedavg", 0, {}, {})


if __name__ == "__main__":
    unittest.main()
