import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


script = Path(__file__).resolve().parents[1] / "run_organa_defense_sweep.py"
spec = importlib.util.spec_from_file_location("organa_defense_sweep", script)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


class OrganADefenseSweepTests(unittest.TestCase):
    def test_matrix_has_300_new_conditions_without_reusing_attack(self):
        conditions = sweep.matrix_conditions()
        self.assertEqual(len(conditions), 15)
        expected = {(c, f) for c in (1, 2, 4, 8) for f in ("0.25", "0.5", "0.75", "1.0")}
        expected.remove((4, "1.0"))
        self.assertEqual(set(conditions), expected)
        self.assertEqual(len(conditions) * len(sweep.SEEDS) * len(sweep.AGGREGATIONS), 300)

    def test_matrix_continues_and_reports_failed_settings(self):
        executions = [f"run-{i}" for i in range(1, 16)]
        outcomes = [SimpleNamespace(returncode=7 if i == 1 else 0) for i in range(15)]
        with (
            mock.patch.object(sweep, "next_execution_id", side_effect=executions),
            mock.patch.object(sweep.subprocess, "run", side_effect=outcomes) as run,
            mock.patch.object(sweep.subprocess, "check_output", return_value="revision\n"),
            mock.patch.object(sweep, "validate_environment"),
            mock.patch.object(sweep, "record_sweep_provenance") as provenance,
            mock.patch.object(sweep, "validate_new_results") as validate,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            status = sweep.run_matrix({}, {42: []}, "revision")
        self.assertEqual(status, 1)
        self.assertEqual(run.call_count, 15)
        self.assertEqual(provenance.call_count, 15)
        self.assertEqual(validate.call_count, 15)
        self.assertIn("run-2: clients=1, fraction=0.5", stderr.getvalue())
        env = run.call_args_list[0].kwargs["env"]
        self.assertEqual(env["SEEDS"], "42 43 44 45 46 47 48 49 50 51")
        self.assertEqual(env["AGGREGATIONS"], "fedavg median")
        self.assertEqual(env["RUN_CLEAN_CONTROLS"], "false")
        self.assertEqual(env["CONTINUE_ON_FAILURE"], "true")
        self.assertEqual(env["LEARNING_RATE"], "0.01")
        self.assertEqual(env["PROTOCOL_ID"], "protocol-v4")

    def test_matrix_reports_result_validation_failure_and_continues(self):
        with (
            mock.patch.object(sweep, "matrix_conditions", return_value=[(1, "0.25"), (2, "0.25")]),
            mock.patch.object(sweep, "next_execution_id", side_effect=["run-1", "run-2"]),
            mock.patch.object(sweep.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run,
            mock.patch.object(sweep.subprocess, "check_output", return_value="revision\n"),
            mock.patch.object(sweep, "validate_environment"),
            mock.patch.object(sweep, "record_sweep_provenance"),
            mock.patch.object(sweep, "validate_new_results", side_effect=[ValueError("partition differs"), None]),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            self.assertEqual(sweep.run_matrix({}, {}, "revision"), 1)
        self.assertEqual(run.call_count, 2)
        self.assertIn("partition differs", stderr.getvalue())
        self.assertIn("run-1: clients=1, fraction=0.25", stderr.getvalue())

    def test_preflight_failure_prevents_all_runs(self):
        with (
            mock.patch.object(sweep, "validate_reference", side_effect=ValueError("bad reference")),
            mock.patch.object(sweep, "run_matrix") as run,
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            self.assertEqual(sweep.main(), 1)
        run.assert_not_called()
        self.assertIn("bad reference", stderr.getvalue())

    def test_environment_change_stops_remaining_settings(self):
        with (
            mock.patch.object(sweep, "matrix_conditions", return_value=[(1, "0.25"), (2, "0.25")]),
            mock.patch.object(sweep, "next_execution_id", return_value="run-1"),
            mock.patch.object(sweep.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run,
            mock.patch.object(sweep.subprocess, "check_output", return_value="revision\n"),
            mock.patch.object(sweep, "validate_environment", side_effect=[None, ValueError("source changed")]),
            mock.patch.object(sweep, "record_sweep_provenance"),
            mock.patch.object(sweep, "validate_new_results"),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            self.assertEqual(sweep.run_matrix({}, {}, "revision"), 1)
        self.assertEqual(run.call_count, 1)
        self.assertIn("source changed", stderr.getvalue())

    def test_records_reference_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference"
            execution = root / "outputs" / "run-1"
            reference.mkdir()
            execution.mkdir(parents=True)
            (reference / "pilot-manifest.json").write_text('{"reference": true}')
            reference_manifest = {
                key: None for key in (
                    "source_sha256", "dataset_archive_md5", "python", "platform",
                    "torch_cuda_available", "cuda_version", "cuda_devices",
                    "visible_cuda_devices", "omp_num_threads", "mkl_num_threads",
                )
            }
            reference_manifest["dependencies"] = ["torch==1"]
            execution_manifest = {
                **reference_manifest, "original": True, "git_revision": "revision",
                "git_status": [], "seeds": list(sweep.SEEDS),
                "aggregations": list(sweep.AGGREGATIONS), "includes_clean_controls": False,
                "learning_rate": 0.01, "poison_num_clients": 2, "poison_fraction": 0.5,
                "continue_on_failure": True, "evaluate_test": False,
            }
            (execution / "pilot-manifest.json").write_text(json.dumps(execution_manifest))
            with (
                mock.patch.object(sweep, "ROOT", root),
                mock.patch.object(sweep, "REFERENCE", reference),
                mock.patch.object(sweep, "RESULTS", root / "outputs"),
            ):
                sweep.record_sweep_provenance("run-1", 2, "0.5", reference_manifest, "revision")
            saved = json.loads((execution / "pilot-manifest.json").read_text())
            self.assertTrue(saved["original"])
            self.assertEqual(saved["sweep"]["reference_execution"], "reference")
            self.assertEqual(saved["sweep"]["poison_num_clients"], 2)
            self.assertEqual(saved["sweep"]["poison_fraction"], 0.5)
            self.assertEqual(len(saved["sweep"]["reference_manifest_sha256"]), 64)
            execution_manifest["source_sha256"] = {"modified.py": "changed"}
            (execution / "pilot-manifest.json").write_text(json.dumps(execution_manifest))
            with (
                mock.patch.object(sweep, "ROOT", root),
                mock.patch.object(sweep, "REFERENCE", reference),
                mock.patch.object(sweep, "RESULTS", root / "outputs"),
                self.assertRaisesRegex(ValueError, "manifest does not match"),
            ):
                sweep.record_sweep_provenance("run-1", 2, "0.5", reference_manifest, "revision")

    def test_reference_rejects_mismatched_partition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference"
            reference.mkdir()
            manifest = {
                "seeds": [42],
                "learning_rate": 0.01,
                "aggregations": ["fedavg", "median"],
                "includes_clean_controls": True,
                "poison_num_clients": 4,
                "poison_fraction": 1.0,
                "evaluate_test": False,
            }
            (reference / "pilot-manifest.json").write_text(json.dumps(manifest))
            (root / "model.pt").touch()
            for aggregation in sweep.AGGREGATIONS:
                for count in (0, 4):
                    partition = f"{aggregation}-{count}-partition.json"
                    (root / partition).write_text(json.dumps({
                        "verification": {
                            "partitions_are_disjoint": True,
                            "all_training_examples_assigned": True,
                            "each_training_example_assigned_once": True,
                        },
                        "clients": [{"indices": [0]}],
                    }))
                    record = {
                        "status": "completed", "dataset": "organamnist", "seed": 42,
                        "aggregation_id": aggregation,
                        "aggregation": "FedAvg" if aggregation == "fedavg" else "coordinate-wise median",
                        "partition_manifest": partition,
                        "model_artifact": "model.pt",
                        "round_checkpoints": {str(i): "model.pt" for i in range(1, 6)},
                        "metrics": {"test_final": None},
                        "training_results": {
                            "valid_run": True, "metrics_finite": True,
                            "model_parameters_finite": True,
                            "received_results_by_round": {str(i): 20 for i in range(1, 6)},
                            "client_failures_by_round": {str(i): 0 for i in range(1, 6)},
                        },
                        "configuration": {
                            "clients": 20, "federated_rounds": 5, "local_epochs": 1,
                            "learning_rate": 0.01, "batch_size": 32, "require_complete_round": True,
                            "label_flipping": {
                                "num_poisoned_clients": count,
                                "fraction_of_source_labels_flipped_per_poisoned_client": float(count == 4),
                                "source_class_id": 4, "target_class_id": 5,
                                "poisoned_client_ids": [1, 2, 3, 4] if count else [],
                            },
                        },
                    }
                    (reference / f"seed-{aggregation}-{count}.json").write_text(json.dumps(record))
            with (
                mock.patch.object(sweep, "ROOT", root),
                mock.patch.object(sweep, "REFERENCE", reference),
                mock.patch.object(sweep, "SEEDS", range(42, 43)),
            ):
                _, partitions = sweep.validate_reference()
                self.assertIn(42, partitions)
                path = root / "median-4-partition.json"
                partition = json.loads(path.read_text())
                partition["clients"] = [{"indices": [1]}]
                path.write_text(json.dumps(partition))
                with self.assertRaisesRegex(ValueError, "partitions differ"):
                    sweep.validate_reference()


if __name__ == "__main__":
    unittest.main()
