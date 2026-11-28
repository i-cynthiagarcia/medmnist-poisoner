"""Preflight and run all three expanded Organ defense sweeps sequentially."""

import argparse
import subprocess
import sys
from pathlib import Path

import run_organ_cs_defense_sweep as organ_cs
import run_organa_defense_sweep as organ_a

ROOT = Path(__file__).resolve().parent
RUNNER = Path(__file__).resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run all three expanded Organ defense sweeps: 900 new conditions."
    )
    parser.parse_args(argv)
    try:
        committed = subprocess.check_output(
            ["git", "show", "HEAD:quickstart-pytorch/run_organ_defense_sweeps.py"], cwd=ROOT
        )
        if committed != RUNNER.read_bytes():
            raise ValueError("Commit the all-dataset launcher before starting")
        reference, _ = organ_a.validate_reference()
        organ_a.validate_environment(reference)
        organ_cs.preflight()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"All-dataset preflight failed: {error}", file=sys.stderr)
        return 1

    for script in ("run_organa_defense_sweep.py", "run_organ_cs_defense_sweep.py"):
        print(f"\nStarting {script}", flush=True)
        try:
            result = subprocess.run([sys.executable, "-u", str(ROOT / script)], cwd=ROOT)
        except OSError as error:
            print(f"Could not start {script}: {error}; remaining sweeps not attempted", file=sys.stderr)
            return 1
        if result.returncode:
            print(f"{script} exited {result.returncode}; remaining sweeps not attempted", file=sys.stderr)
            return 1
    print("\nAll three Organ defense sweeps completed and passed verification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
