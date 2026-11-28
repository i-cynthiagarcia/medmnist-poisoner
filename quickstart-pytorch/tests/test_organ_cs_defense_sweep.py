import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_organ_cs_defense_sweep as sweep
import run_organ_defense_sweeps as launcher
import run_organa_defense_sweep as organa
from pytorchexample.experiments.poisoning.label_flipping import select_poisoned_clients
from pytorchexample.shared.data import get_class_names


class OrganCSDefenseSweepTests(unittest.TestCase):
    def test_reference_rejects_incomplete_or_incompatible_comparisons(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "run-1"
            reference.mkdir()
            environment = {key: f"value-{key}" for key in sweep.ENVIRONMENT_KEYS}
            environment["dependencies"] = ["torch==1"]
            manifest = {
                **environment, "dataset": "organcmnist", "dataset_archive_md5": "checksum",
                "seeds": [42], "aggregations": list(sweep.AGGREGATIONS), "clients": 20,
                "rounds": 5, "local_epochs": 3, "learning_rate": 0.01, "batch_size": 32,
                "attack": {"poisoned_clients": 4, "flip_fraction": 1.0, "source": 4, "target": 5},
                "clean_floors": sweep.comparison.FLOORS["organcmnist"],
                "evaluate_test": False, "automatic_retries": False,
            }
            summary = {agg: {"complete": True, "accuracy_floor_met": True,
                             "kidney_left_recall_floor_met": True} for agg in sweep.AGGREGATIONS}
            outcome = {"dataset": "organcmnist", "attempted": 40, "expected": 40,
                       "complete": True, "failures": [], "in_progress": None, "clean_summary": summary}
            manifest_path = reference / "comparison-manifest.json"
            outcome_path = reference / "comparison-outcome.json"
            manifest_path.write_text(json.dumps(manifest))
            outcome_path.write_text(json.dumps(outcome))
            records = {(42, agg, count): {} for agg in sweep.AGGREGATIONS for count in (0, 4)}
            with (
                mock.patch.object(sweep.comparison, "RESULTS", root),
                mock.patch.object(sweep, "SEEDS", (42,)),
                mock.patch.object(sweep, "load_records", return_value=records),
                mock.patch.object(sweep.comparison, "validate_condition"),
                mock.patch.object(sweep.comparison, "clean_summary", return_value=summary),
                mock.patch.object(sweep, "reference_snapshot", return_value={}),
            ):
                sweep.validate_reference("organcmnist", environment, "checksum")
                changed = {**environment, "source_sha256": "different"}
                with self.assertRaisesRegex(ValueError, "source_sha256"):
                    sweep.validate_reference("organcmnist", changed, "checksum")
                outcome["failures"] = ["failed cell"]
                outcome_path.write_text(json.dumps(outcome))
                with self.assertRaisesRegex(ValueError, "incomplete or incompatible"):
                    sweep.validate_reference("organcmnist", environment, "checksum")

    def test_reference_snapshot_detects_artifact_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference/run-1"
            reference.mkdir(parents=True)
            for name in ("comparison-manifest.json", "comparison-outcome.json", "seed-42_run-1.json",
                         "partition.json", "model.pt", "checkpoint.pt"):
                (reference / name).write_text(name)
            records = {(42, "fedavg", 0): {
                "partition_manifest": "reference/run-1/partition.json",
                "model_artifact": "reference/run-1/model.pt",
                "round_checkpoints": {"1": "reference/run-1/checkpoint.pt"},
            }}
            with (
                mock.patch.object(sweep, "ROOT", root),
                mock.patch.object(sweep.comparison, "RESULTS", root / "reference"),
                mock.patch.object(sweep, "load_records", return_value=records),
            ):
                snapshot = sweep.reference_snapshot(reference, records)
                state = {"execution": "run-1", "snapshot": snapshot}
                sweep.verify_reference("organcmnist", state)
                (reference / "checkpoint.pt").write_text("changed")
                with self.assertRaisesRegex(ValueError, "artifacts changed"):
                    sweep.verify_reference("organcmnist", state)

    def test_runs_600_new_conditions_and_preserves_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            references = {
                dataset: {"execution": f"run-{i}", "manifest_sha256": "hash",
                          "snapshot": {}, "partitions": {}}
                for i, dataset in enumerate(sweep.DATASETS, 1)
            }
            outcomes = [SimpleNamespace(returncode=7 if i == 0 else 0) for i in range(600)]
            with (
                mock.patch.object(sweep, "ROOT", root),
                mock.patch.object(sweep, "RESULTS", root / "results"),
                mock.patch.object(sweep, "MODELS", root / "models"),
                mock.patch.object(sweep.comparison, "RESULTS", root / "reference"),
                mock.patch.object(sweep, "preflight", return_value=(
                    {"git_revision": "revision"}, {d: "checksum" for d in sweep.DATASETS}, references
                )),
                mock.patch.object(sweep, "verify_environment"),
                mock.patch.object(sweep, "verify_reference"),
                mock.patch.object(sweep, "validate_condition", side_effect=[ValueError("missing checkpoint")] + [None] * 598),
                mock.patch.object(sweep.subprocess, "run", side_effect=outcomes) as run,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(sweep.main([]), 1)
            self.assertEqual(run.call_count, 600)
            configs = [call.args[0][call.args[0].index("--run-config") + 1]
                       for call in run.call_args_list]
            conditions = {
                tuple(part for part in config.split() if part.startswith(
                    ("dataset=", "seed=", "aggregation=", "poison-num-clients=", "poison-fraction=")
                )) for config in configs
            }
            expected = {
                (f'aggregation="{agg}"', f'dataset="{dataset}"', f"seed={seed}",
                 f"poison-num-clients={count}", f"poison-fraction={fraction}")
                for dataset in sweep.DATASETS for count, fraction in sweep.matrix_conditions()
                for seed in sweep.SEEDS for agg in sweep.AGGREGATIONS
            }
            self.assertEqual(conditions, expected)
            for config in configs:
                for setting in ("num-clients=20", "num-server-rounds=5", "local-epochs=3",
                                "learning-rate=0.01", "batch-size=32", "evaluate-test=false",
                                "require-complete-round=true", "poison-source-class-id=4",
                                "poison-target-class-id=5", 'protocol-id="protocol-v9"'):
                    self.assertIn(setting, config)
            first = json.loads((root / "results/run-1/sweep-outcome.json").read_text())
            second = json.loads((root / "results/run-2/sweep-outcome.json").read_text())
            self.assertEqual(first["attempted"], 300)
            self.assertEqual(len(first["failures"]), 2)
            self.assertFalse(first["complete"])
            self.assertTrue(second["complete"])
            self.assertIn("Flower exited 7", stderr.getvalue())
            self.assertIn("missing checkpoint", stderr.getvalue())
            manifest = json.loads((root / "results/run-2/sweep-manifest.json").read_text())
            self.assertEqual(manifest["reference_execution"], "reference/run-2")
            self.assertEqual(len(manifest["attack_settings"]), 15)
            self.assertFalse(manifest["evaluate_test"])

    def test_environment_drift_stops_and_persists_partial_outcome(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = {"execution": "run-1", "manifest_sha256": "hash",
                         "snapshot": {}, "partitions": {}}
            with (
                mock.patch.object(sweep, "ROOT", root),
                mock.patch.object(sweep, "RESULTS", root / "results"),
                mock.patch.object(sweep, "MODELS", root / "models"),
                mock.patch.object(sweep.comparison, "RESULTS", root / "reference"),
                mock.patch.object(sweep, "verify_environment", side_effect=[None, ValueError("changed")]),
                mock.patch.object(sweep, "verify_reference"),
                mock.patch.object(sweep, "validate_condition"),
                mock.patch.object(sweep.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                failures, stopped = sweep.run_dataset("organcmnist", {}, {"organcmnist": "hash"}, reference)
            self.assertTrue(stopped)
            self.assertEqual(run.call_count, 1)
            self.assertIn("changed", failures[0])
            outcome = json.loads((root / "results/run-1/sweep-outcome.json").read_text())
            self.assertFalse(outcome["complete"])
            self.assertEqual(outcome["attempted"], 1)
            self.assertIsNone(outcome["in_progress"])

    def test_reference_mutation_prevents_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = {"execution": "run-1", "manifest_sha256": "hash", "snapshot": {}, "partitions": {}}
            with (
                mock.patch.object(sweep, "ROOT", root),
                mock.patch.object(sweep, "RESULTS", root / "results"),
                mock.patch.object(sweep, "MODELS", root / "models"),
                mock.patch.object(sweep.comparison, "RESULTS", root / "reference"),
                mock.patch.object(sweep, "matrix_conditions", return_value=[(1, "0.25")]),
                mock.patch.object(sweep, "SEEDS", (42,)),
                mock.patch.object(sweep, "verify_environment"),
                mock.patch.object(sweep, "verify_reference", side_effect=ValueError("reference changed")),
                mock.patch.object(sweep, "validate_condition"),
                mock.patch.object(sweep.subprocess, "run", return_value=SimpleNamespace(returncode=0)),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                failures, stopped = sweep.run_dataset("organcmnist", {}, {"organcmnist": "hash"}, reference)
            outcome = json.loads((root / "results/run-1/sweep-outcome.json").read_text())
            self.assertTrue(stopped)
            self.assertFalse(outcome["complete"])
            self.assertEqual(outcome["attempted"], 2)
            self.assertIn("reference changed", failures[0])

    def test_partial_flip_counts_and_reference_partitions_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selected = select_poisoned_clients(20, 2, 42)
            clients = [{"partition_id": i, "class_counts": {"kidney-left": i + 1}} for i in range(20)]
            partition = {
                "dataset": "organcmnist", "seed": 42, "num_clients": 20,
                "verification": {"partitions_are_disjoint": True, "all_training_examples_assigned": True,
                                 "each_training_example_assigned_once": True},
                "clients": clients,
            }
            (root / "partition.json").write_text(json.dumps(partition))
            (root / "model.pt").touch()
            names = get_class_names("organcmnist")
            flips = {str(i): round((i + 1) * 0.25) for i in selected}
            record = {
                "generation_id": "generation-v1", "protocol_id": "protocol-v9", "execution_id": "run-1",
                "status": "completed", "dataset": "organcmnist", "seed": 42,
                "aggregation_id": "fedavg", "aggregation": "FedAvg",
                "artifact_layout": "aggregation_subdirectories", "partition_manifest": "partition.json",
                "model_artifact": "model.pt", "round_checkpoints": {str(i): "model.pt" for i in range(1, 6)},
                "training_results": {"valid_run": True, "metrics_finite": True, "model_parameters_finite": True,
                                     "received_results_by_round": {str(i): 20 for i in range(1, 6)},
                                     "client_failures_by_round": {str(i): 0 for i in range(1, 6)}},
                "metrics": {
                    "test_final": None, "class_names": names,
                    "validation_by_round": {str(i): {"accuracy": 0.8, "loss": 0.5,
                                             "per_class": {name: {"recall": 0.5} for name in names}}
                                            for i in range(6)},
                    "targeted_source_to_target_misclassification": {
                        "validation_by_round": {str(i): {"rate": 0.2, "support": 4} for i in range(6)},
                        "test_final": None},
                },
                "configuration": {
                    "clients": 20, "federated_rounds": 5, "local_epochs": 3, "learning_rate": 0.01,
                    "batch_size": 32, "require_complete_round": True,
                    "label_flipping": {"num_poisoned_clients": 2,
                                       "fraction_of_source_labels_flipped_per_poisoned_client": 0.25,
                                       "source_class_id": 4, "target_class_id": 5,
                                       "poisoned_client_ids": selected,
                                       "flipped_source_examples_by_client": flips,
                                       "total_flipped_source_examples": sum(flips.values())},
                },
            }
            directory = root / "results/run-1/fedavg/organcmnist/label_flip_r5_e3_lr0p010_clients2_f0p250_s4_t5"
            directory.mkdir(parents=True)
            path = directory / "seed-42_run-1.json"
            path.write_text(json.dumps(record))
            with (mock.patch.object(sweep, "ROOT", root), mock.patch.object(sweep, "RESULTS", root / "results"),
                  mock.patch.object(organa, "ROOT", root)):
                sweep.validate_condition("organcmnist", "run-1", 42, "fedavg", 2, "0.25", {42: clients})
                with self.assertRaisesRegex(ValueError, "Partition summary differs"):
                    sweep.validate_condition("organcmnist", "run-1", 42, "fedavg", 2, "0.25", {42: []})
                record["configuration"]["label_flipping"]["total_flipped_source_examples"] += 1
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "flip counts"):
                    sweep.validate_condition("organcmnist", "run-1", 42, "fedavg", 2, "0.25", {42: clients})
                record["round_checkpoints"]["5"] = "missing.pt"
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "Missing checkpoint"):
                    sweep.validate_condition("organcmnist", "run-1", 42, "fedavg", 2, "0.25", {42: clients})


class OrganSweepLauncherTests(unittest.TestCase):
    def test_runs_both_sweeps_in_order(self):
        with (
            mock.patch.object(launcher.subprocess, "check_output", return_value=launcher.RUNNER.read_bytes()),
            mock.patch.object(launcher.organ_a, "validate_reference", return_value=({}, {})),
            mock.patch.object(launcher.organ_a, "validate_environment"),
            mock.patch.object(launcher.organ_cs, "preflight"),
            mock.patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(launcher.main([]), 0)
        self.assertEqual(
            [Path(call.args[0][-1]).name for call in run.call_args_list],
            ["run_organa_defense_sweep.py", "run_organ_cs_defense_sweep.py"],
        )

    def test_preflights_both_before_running_and_stops_after_failure(self):
        with (
            mock.patch.object(launcher.subprocess, "check_output", return_value=launcher.RUNNER.read_bytes()),
            mock.patch.object(launcher.organ_a, "validate_reference", return_value=({}, {})),
            mock.patch.object(launcher.organ_a, "validate_environment"),
            mock.patch.object(launcher.organ_cs, "preflight") as cs_preflight,
            mock.patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as run,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            self.assertEqual(launcher.main([]), 1)
        cs_preflight.assert_called_once()
        self.assertEqual(run.call_count, 1)
        self.assertIn("remaining sweeps not attempted", stderr.getvalue())

    def test_preflight_failure_starts_no_sweep(self):
        with (
            mock.patch.object(launcher.subprocess, "check_output", return_value=launcher.RUNNER.read_bytes()),
            mock.patch.object(launcher.organ_a, "validate_reference", return_value=({}, {})),
            mock.patch.object(launcher.organ_a, "validate_environment"),
            mock.patch.object(launcher.organ_cs, "preflight", side_effect=ValueError("bad reference")),
            mock.patch.object(launcher.subprocess, "run") as run,
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            self.assertEqual(launcher.main([]), 1)
        run.assert_not_called()
        self.assertIn("bad reference", stderr.getvalue())

    def test_unknown_options_never_start_a_sweep(self):
        for module in (sweep, launcher):
            with (
                self.subTest(module=module.__name__),
                mock.patch.object(module.subprocess, "run") as run,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                module.main(["--dataset", "organcmnist"])
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
