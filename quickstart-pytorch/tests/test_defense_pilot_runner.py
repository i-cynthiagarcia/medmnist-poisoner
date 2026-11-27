import os
import shutil
import subprocess
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
