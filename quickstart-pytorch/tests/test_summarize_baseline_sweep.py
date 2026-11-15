import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from summarize_baseline_sweep import (
    CONFIGURATIONS,
    DATASETS,
    DEFAULT_SEEDS,
    build_parser,
    resolve_execution_id,
    render_summary,
    summarize_records,
    write_report,
)


class BaselineSweepSummaryTests(unittest.TestCase):
    def make_records(self, project_root, include_centralized=True):
        records = []
        partition_index = 0
        for dataset in DATASETS:
            for config_index, (rounds, epochs, learning_rate) in enumerate(
                CONFIGURATIONS
            ):
                for seed_index, seed in enumerate(DEFAULT_SEEDS):
                    experiment = f"baseline_r{rounds}_e{epochs}"
                    accuracy = 0.4 + config_index * 0.01 + seed_index * 0.01
                    loss = 2.0 - seed_index * 0.1
                    validations = {
                        str(round_number): {
                            "accuracy": accuracy if round_number == rounds else 0.1,
                            "loss": loss if round_number == rounds else 2.3,
                        }
                        for round_number in range(rounds + 1)
                    }
                    manifest_path = (
                        project_root / "partitions" / f"partition-{partition_index}.json"
                    )
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    manifest_path.write_text(
                        json.dumps(
                            {
                                "generation_id": "generation-v3",
                                "protocol_id": "protocol-v1",
                                "execution_id": "run-4",
                                "dataset": dataset,
                                "seed": seed,
                                "num_clients": 20,
                                "verification": {
                                    "partitions_are_disjoint": True,
                                    "all_training_examples_assigned": True,
                                    "each_training_example_assigned_once": True,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    partition_ref = manifest_path.relative_to(
                        project_root
                    ).as_posix()
                    records.append(
                        {
                            "generation_id": "generation-v3",
                            "protocol_id": "protocol-v1",
                            "execution_id": "run-4",
                            "dataset": dataset,
                            "experiment": experiment,
                            "seed": seed,
                            "status": "completed",
                            "training_mode": "federated",
                            "configuration": {
                                "clients": 20,
                                "federated_rounds": rounds,
                                "local_epochs": epochs,
                                "learning_rate": learning_rate,
                            },
                            "training_results": {
                                "valid_run": True,
                                "received_results_by_round": {
                                    str(number): 20
                                    for number in range(1, rounds + 1)
                                },
                                "client_failures_by_round": {
                                    str(number): 0
                                    for number in range(1, rounds + 1)
                                },
                            },
                            "partition_manifest": partition_ref,
                            "loss_aggregation": "per_image",
                            "metrics": {
                                "validation_by_round": deepcopy(validations),
                                "test_final": None,
                            },
                        }
                    )
                    partition_index += 1

                    if include_centralized:
                        records.append(
                            {
                                "generation_id": "generation-v3",
                                "protocol_id": "protocol-v1",
                                "execution_id": "run-4",
                                "dataset": dataset,
                                "experiment": f"centralized_r{rounds}_e{epochs}",
                                "seed": seed,
                                "status": "completed",
                                "training_mode": "centralized",
                                "configuration": {
                                    "training_data": "full_train_split",
                                    "epochs": rounds * epochs,
                                    "federated_round_equivalents": rounds,
                                    "local_epochs_per_round_equivalent": epochs,
                                    "learning_rate": learning_rate,
                                },
                                "loss_aggregation": "per_image",
                                "metrics": {
                                    "validation_by_round": deepcopy(validations),
                                    "test_final": None,
                                },
                            }
                        )
        return records

    def test_summarize_full_grid_and_population_statistics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            summary = summarize_records(
                self.make_records(project_root),
                project_root,
                "generation-v3",
                "protocol-v1",
                "run-4",
            )

        self.assertEqual(len(summary.federated), 21)
        self.assertEqual(len(summary.centralized), 21)
        self.assertEqual(summary.partition_count, 105)
        first = summary.federated[0]
        self.assertEqual(first.dataset, "organamnist")
        self.assertEqual(first.rounds, 3)
        self.assertEqual(first.local_epochs, 1)
        self.assertAlmostEqual(first.mean_accuracy, 0.42)
        self.assertAlmostEqual(first.accuracy_stddev, 0.0141421356)
        self.assertAlmostEqual(first.mean_loss, 1.8)
        report = render_summary(summary)
        self.assertIn("105 valid federated runs", report)
        self.assertIn("Centralized full-train comparison results", report)

    def test_rejects_federated_client_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            records = self.make_records(project_root, include_centralized=False)
            records[0]["training_results"]["client_failures_by_round"]["1"] = 1

            with self.assertRaisesRegex(ValueError, "Client failures found"):
                summarize_records(
                    records,
                    project_root,
                    "generation-v3",
                    "protocol-v1",
                    "run-4",
                )

    def test_omits_centralized_summary_when_metrics_are_non_finite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            records = self.make_records(project_root)
            affected = next(
                record
                for record in records
                if record["training_mode"] == "centralized"
                and record["dataset"] == "organamnist"
                and record["configuration"]["learning_rate"] == 0.1
                and record["seed"] == 42
            )
            affected["metrics"]["validation_by_round"]["3"]["loss"] = float(
                "nan"
            )
            summary = summarize_records(
                records,
                project_root,
                "generation-v3",
                "protocol-v1",
                "run-4",
            )

        self.assertIsNone(summary.centralized)
        self.assertEqual(len(summary.centralized_issues), 1)
        self.assertIn("Non-finite final validation metrics", summary.centralized_issues[0])

    def test_report_defaults_to_stdout_and_can_be_saved(self):
        self.assertIsNone(build_parser().parse_args([]).output)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "reports" / "baseline_summary.md"
            write_report("# Baseline summary\n", output_path)

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "# Baseline summary\n",
            )

    def test_auto_execution_id_uses_highest_numeric_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "results"
            protocol_dir = root / "generation-v3" / "protocol-v1"
            (protocol_dir / "run-4").mkdir(parents=True)
            (protocol_dir / "run-12").mkdir()

            self.assertEqual(
                resolve_execution_id(
                    root, "generation-v3", "protocol-v1", "auto"
                ),
                "run-12",
            )


if __name__ == "__main__":
    unittest.main()
