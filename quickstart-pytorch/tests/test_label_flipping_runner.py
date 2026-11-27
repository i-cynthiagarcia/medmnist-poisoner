import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class LabelFlippingRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        script = Path(__file__).resolve().parents[1] / "run_label_flipping_experiments.sh"
        shutil.copy(script, self.root / script.name)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        python = bin_dir / "python"
        python.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "-c" ]; then exit 0; fi\n'
            f'exec {shlex.quote(sys.executable)} "$@"\n'
        )
        flwr = bin_dir / "flwr"
        flwr.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$FLWR_CALLS"\n')
        python.chmod(0o755)
        flwr.chmod(0o755)
        self.calls = self.root / "calls.txt"
        self.env = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "FLWR_CALLS": str(self.calls),
            "SEEDS": "42",
            "POISONED_CLIENT_COUNTS": "1",
            "POISON_FRACTIONS": "0.5",
        }
        for key in (
            "RESULTS_DIR",
            "MODELS_DIR",
            "GENERATION_ID",
            "PROTOCOL_ID",
            "EXECUTION_ID",
        ):
            self.env.pop(key, None)

    def run_script(self):
        return subprocess.run(
            ["bash", "run_label_flipping_experiments.sh", "organamnist", "4", "5"],
            cwd=self.root,
            env=self.env,
            text=True,
            capture_output=True,
        )

    def test_defaults_use_flipping_roots_and_next_execution(self):
        (self.root / "models/poisoning/flipping/generation-v3/protocol-v1/run-3").mkdir(
            parents=True
        )

        result = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls.read_text().splitlines()
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertIn('results-dir="results/poisoning/flipping"', call)
            self.assertIn('models-dir="models/poisoning/flipping"', call)
            self.assertIn('execution-id="run-4"', call)
        self.assertTrue(
            (self.root / "results/poisoning/flipping/generation-v3/protocol-v1/run-4").is_dir()
        )

    def test_custom_roots_are_used_directly_and_existing_execution_is_refused(self):
        self.env.update(
            RESULTS_DIR="custom-results",
            MODELS_DIR="custom-models",
            EXECUTION_ID="run-2",
        )

        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        for call in self.calls.read_text().splitlines():
            self.assertIn('results-dir="custom-results"', call)
            self.assertIn('models-dir="custom-models"', call)
            self.assertIn('execution-id="run-2"', call)

        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("output path already exists", result.stderr)
        self.assertEqual(len(self.calls.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
