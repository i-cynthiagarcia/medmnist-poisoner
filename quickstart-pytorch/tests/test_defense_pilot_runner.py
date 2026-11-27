import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DefensePilotRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        script = Path(__file__).resolve().parents[1] / "run_median_defense_pilot.sh"
        shutil.copy(script, self.root / script.name)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        python = bin_dir / "python"
        python.write_text("#!/bin/sh\nexit 0\n")
        flwr = bin_dir / "flwr"
        flwr.write_text(
            '#!/bin/sh\n'
            'printf "%s\\n" "$*" >> "$FLWR_CALLS"\n'
            'if [ -n "${FLWR_FAIL_SEED:-}" ]; then\n'
            '  case "$*" in *"seed=$FLWR_FAIL_SEED "*) exit 7 ;; esac\n'
            'fi\n'
        )
        python.chmod(0o755)
        flwr.chmod(0o755)
        self.calls = self.root / "calls.txt"
        self.env = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "FLWR_CALLS": str(self.calls),
            "SEEDS": "42 43",
        }
        for key in (
            "RESULTS_DIR",
            "MODELS_DIR",
            "GENERATION_ID",
            "PROTOCOL_ID",
            "EXECUTION_ID",
            "AGGREGATIONS",
            "RUN_CLEAN_CONTROLS",
            "POISON_NUM_CLIENTS",
            "POISON_FRACTION",
            "LEARNING_RATE",
            "CONTINUE_ON_FAILURE",
            "FLWR_FAIL_SEED",
        ):
            self.env.pop(key, None)

    def run_script(self):
        return subprocess.run(
            ["bash", "run_median_defense_pilot.sh"],
            cwd=self.root,
            env=self.env,
            text=True,
            capture_output=True,
        )

    def test_runs_four_matched_validation_only_conditions_per_seed(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls.read_text().splitlines()
        self.assertEqual(len(calls), 8)
        for seed in (42, 43):
            seed_calls = [call for call in calls if f"seed={seed} " in call]
            self.assertEqual(len(seed_calls), 4)
            for aggregation in ("fedavg", "median"):
                for count, fraction in ((0, "0.0"), (1, "1.0")):
                    matches = [
                        call for call in seed_calls
                        if f'aggregation="{aggregation}"' in call
                        and f"poison-num-clients={count} " in call
                        and f"poison-fraction={fraction} " in call
                    ]
                    self.assertEqual(len(matches), 1)
                    self.assertIn("evaluate-test=false", matches[0])
                    self.assertIn("require-complete-round=true", matches[0])
                    self.assertIn("aggregation-subdirectories=true", matches[0])
                    self.assertIn("poison-source-class-id=4", matches[0])
                    self.assertIn("poison-target-class-id=5", matches[0])
                    self.assertIn("learning-rate=0.1 ", matches[0])

    def test_lower_rate_is_matched_across_all_conditions(self):
        self.env.update(
            SEEDS="49",
            PROTOCOL_ID="protocol-v3",
            LEARNING_RATE="0.01",
            POISON_NUM_CLIENTS="4",
        )
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls.read_text().splitlines()
        self.assertEqual(len(calls), 4)
        for aggregation in ("fedavg", "median"):
            for count, fraction in ((0, "0.0"), (4, "1.0")):
                matching = [
                    call for call in calls
                    if f'aggregation="{aggregation}"' in call
                    and f"poison-num-clients={count} " in call
                    and f"poison-fraction={fraction} " in call
                ]
                self.assertEqual(len(matching), 1)
                self.assertIn("learning-rate=0.01 ", matching[0])
                self.assertIn('protocol-id="protocol-v3"', matching[0])

    def test_rejects_mixed_rate_protocols_and_invalid_rate(self):
        for rate, protocol in (
            ("0.01", "protocol-v1"),
            ("0.01", "protocol-v2"),
            ("0.1", "protocol-v3"),
            ("0", "protocol-v3"),
        ):
            with self.subTest(rate=rate, protocol=protocol):
                self.env.update(LEARNING_RATE=rate, PROTOCOL_ID=protocol)
                self.assertNotEqual(self.run_script().returncode, 0)
                self.assertFalse(self.calls.exists())

    def test_lower_rate_result_validation_checks_all_rounds_and_rate(self):
        script = (self.root / "run_median_defense_pilot.sh").read_text()
        validator = script.split('<<\'PY\'\n')[2].split('\nPY\n', 1)[0]
        result_dir = (
            self.root / "median" / "organamnist"
            / "median_label_flip_r5_e1_lr0p010_clients4_f1p000_s4_t5"
        )
        result_dir.mkdir(parents=True)
        path = result_dir / "seed-49_run-1.json"
        record = {
            "status": "completed",
            "seed": 49,
            "aggregation": "coordinate-wise median",
            "aggregation_id": "median",
            "artifact_layout": "aggregation_subdirectories",
            "metrics": {"test_final": None},
            "training_results": {
                "valid_run": True,
                "metrics_finite": True,
                "model_parameters_finite": True,
                "received_results_by_round": {str(i): 20 for i in range(1, 6)},
                "client_failures_by_round": {str(i): 0 for i in range(1, 6)},
            },
            "configuration": {
                "clients": 20,
                "federated_rounds": 5,
                "local_epochs": 1,
                "learning_rate": 0.01,
                "batch_size": 32,
                "require_complete_round": True,
                "label_flipping": {
                    "num_poisoned_clients": 4,
                    "fraction_of_source_labels_flipped_per_poisoned_client": 1.0,
                    "source_class_id": 4,
                    "target_class_id": 5,
                },
            },
        }

        def validate():
            path.write_text(json.dumps(record))
            return subprocess.run(
                [sys.executable, "-", str(self.root), "49", "median", "4", "1p000", "1.0", "0.01"],
                input=validator, text=True, capture_output=True,
            )

        self.assertEqual(validate().returncode, 0)
        del record["training_results"]["received_results_by_round"]["5"]
        self.assertNotEqual(validate().returncode, 0)
        record["training_results"]["received_results_by_round"]["5"] = 20
        record["configuration"]["learning_rate"] = 0.1
        self.assertNotEqual(validate().returncode, 0)

    def test_runs_only_selected_poisoned_aggregation_when_controls_are_reused(self):
        self.env.update(
            SEEDS="49",
            AGGREGATIONS="median",
            RUN_CLEAN_CONTROLS="false",
            POISON_NUM_CLIENTS="4",
            POISON_FRACTION="1.0",
        )
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls.read_text().splitlines()
        self.assertEqual(len(calls), 1)
        self.assertIn('aggregation="median"', calls[0])
        self.assertIn("seed=49 ", calls[0])
        self.assertIn("poison-num-clients=4 ", calls[0])
        self.assertIn("poison-fraction=1.0 ", calls[0])
        self.assertIn("require-complete-round=true", calls[0])
        self.assertIn("evaluate-test=false", calls[0])

    def test_continue_on_failure_attempts_remaining_seeds_and_reports_failure(self):
        self.env.update(
            SEEDS="42 43",
            AGGREGATIONS="median",
            RUN_CLEAN_CONTROLS="false",
            POISON_NUM_CLIENTS="4",
            CONTINUE_ON_FAILURE="true",
            FLWR_FAIL_SEED="42",
        )
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        calls = self.calls.read_text().splitlines()
        self.assertEqual(len(calls), 2)
        self.assertIn("seed=42 ", calls[0])
        self.assertIn("seed=43 ", calls[1])
        self.assertIn("seed=42 aggregation=median clients=4", result.stderr)
        self.assertIn("Failed conditions:", result.stderr)

    def test_rejects_repeated_seeds_and_existing_execution(self):
        self.env["SEEDS"] = "42 42"
        self.assertNotEqual(self.run_script().returncode, 0)
        self.assertFalse(self.calls.exists())
        self.env.update(SEEDS="42", EXECUTION_ID="run-1")
        self.assertEqual(self.run_script().returncode, 0)
        self.assertNotEqual(self.run_script().returncode, 0)
        self.assertEqual(len(self.calls.read_text().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
