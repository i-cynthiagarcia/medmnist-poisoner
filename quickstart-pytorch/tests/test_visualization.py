import tempfile
import unittest
from pathlib import Path

from pytorchexample.shared.visualization import save_confusion_matrix_plot
from render_confusion_matrices import plan_run_plots


class ConfusionMatrixPlotTests(unittest.TestCase):
    def test_validation_v2_plan_uses_unique_run_plot_path(self):
        result_path = Path(
            "results/baseline/generation-v2/organamnist/config/seed-42.json"
        )
        record = {
            "seed": 42,
            "run_id": "run-123",
            "metrics": {
                "class_names": ["a", "b"],
                "validation_final_recomputed": {
                    "confusion_matrix": [[2, 0], [1, 1]]
                },
            },
        }

        plot_refs, jobs = plan_run_plots(result_path, record)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(
            plot_refs["validation_final_recomputed"],
            (
                result_path.parent
                / "plots"
                / "seed-42_run-run-123"
                / "validation-final-recomputed.png"
            ).as_posix(),
        )
        self.assertEqual(plot_refs["test_final"], None)

    def test_round_plan_includes_each_round_and_optional_test_matrix(self):
        result_path = Path(
            "results/baseline/generation-v3/"
            "protocol_20261115T090141Z/"
            "organamnist/config/run.json"
        )
        record = {
            "seed": 42,
            "run_id": "run-123",
            "metrics": {
                "class_names": ["a", "b"],
                "validation_by_round": {
                    "0": {"confusion_matrix": [[2, 0], [1, 1]]},
                    "1": {"confusion_matrix": [[1, 1], [0, 2]]},
                },
                "test_final": {"confusion_matrix": [[2, 0], [0, 2]]},
            },
        }

        plot_refs, jobs = plan_run_plots(result_path, record)
        plot_dir = (
            result_path.parent / "plots" / "seed-42_run-run-123"
        )

        self.assertEqual(len(jobs), 3)
        self.assertEqual(
            plot_refs["validation_by_round"],
            {
                "0": (plot_dir / "validation-round-000.png").as_posix(),
                "1": (plot_dir / "validation-round-001.png").as_posix(),
            },
        )
        self.assertEqual(plot_refs["validation_final_recomputed"], None)
        self.assertEqual(
            plot_refs["test_final"],
            (plot_dir / "test-final.png").as_posix(),
        )

    def test_plot_is_written_and_existing_file_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_path = Path(temp_dir) / "matrix.png"
            save_confusion_matrix_plot(
                [[2, 0], [1, 1]], ["a", "b"], plot_path, "Confusion matrix"
            )
            original_bytes = plot_path.read_bytes()

            with self.assertRaises(FileExistsError):
                save_confusion_matrix_plot(
                    [[0, 2], [1, 1]],
                    ["a", "b"],
                    plot_path,
                    "Different matrix",
                )

            self.assertEqual(plot_path.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
