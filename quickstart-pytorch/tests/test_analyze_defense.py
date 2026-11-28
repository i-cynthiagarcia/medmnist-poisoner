import json
import shutil
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from analysis.analyze_defense import (
    COMPARISONS,
    ROOT,
    build_tables,
    load_comparison,
    summarize,
)


class DefenseAnalysisTests(unittest.TestCase):
    def test_completed_comparisons_reproduce_reported_interactions(self):
        paired = []
        classes = []
        for spec in COMPARISONS:
            sources, seed_rows, class_rows = build_tables(spec, load_comparison(ROOT, spec))
            self.assertEqual(len(sources), 40)
            paired.extend(seed_rows)
            classes.extend(class_rows)
        summaries = summarize(paired, "metric")
        for dataset, expected in (("organamnist", -17.82), ("organcmnist", -9.92), ("organsmnist", -7.93)):
            row = next(r for r in summaries if r["dataset"] == dataset and r["metric"] == "targeted_error")
            self.assertAlmostEqual(row["interaction_mean"] * 100, expected, delta=0.01)
            self.assertEqual(row["n"], 10)
            self.assertGreater(row["interaction_sd"], 0)
        seed44 = next(r for r in paired if r["dataset"] == "organsmnist"
                      and r["seed"] == 44 and r["metric"] == "targeted_error")
        self.assertAlmostEqual(seed44["interaction"] * 100, 1.43, delta=0.01)
        zero_recalls = [r for r in classes if r["dataset"] == "organsmnist"
                        and r["class"] == "femur-right" and r["fedavg_clean"] == 0]
        self.assertEqual(len(zero_recalls), 5)

    def test_variability_is_calculated_on_paired_differences(self):
        spec = COMPARISONS[0]
        records = deepcopy(load_comparison(ROOT, spec))
        for seed in range(42, 52):
            baseline = (seed - 42) / 100
            for aggregation, change in (("fedavg", 0.2), ("median", 0.1)):
                records[seed, aggregation, 0]["values"]["targeted_error"] = baseline
                records[seed, aggregation, 4]["values"]["targeted_error"] = baseline + change
        _, paired, _ = build_tables(spec, records)
        # Summarize expects all explicitly selected datasets.
        for other in COMPARISONS[1:]:
            paired.extend(build_tables(other, load_comparison(ROOT, other))[1])
        summary = next(r for r in summarize(paired, "metric")
                       if r["dataset"] == "organamnist" and r["metric"] == "targeted_error")
        self.assertAlmostEqual(summary["fedavg_change_mean"], 0.2)
        self.assertAlmostEqual(summary["interaction_mean"], -0.1)
        self.assertAlmostEqual(summary["interaction_sd"], 0.0)
        self.assertGreater(summary["fedavg_clean_sd"], 0)

    def test_missing_duplicate_and_wrong_protocol_are_rejected(self):
        spec = COMPARISONS[0]
        source = ROOT / "results/defense/generation-v1/protocol-v3/run-2"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "results/defense/generation-v1/protocol-v3/run-2"
            shutil.copytree(source, dest)
            path = next(dest.rglob("seed-42_run-*.json"))
            contents = path.read_text()
            path.unlink()
            with self.assertRaisesRegex(ValueError, "missing conditions"):
                load_comparison(root, spec)
            path.write_text(contents)
            duplicate = path.with_name("seed-42_run-999.json")
            duplicate.write_text(contents)
            with self.assertRaisesRegex(ValueError, "Invalid or duplicate"):
                load_comparison(root, spec)
            duplicate.unlink()
            record = json.loads(contents)
            record["configuration"]["learning_rate"] = 0.1
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "Invalid or duplicate"):
                load_comparison(root, spec)
            path.write_text(contents)
            attack_path = next(dest.glob("median/organamnist/*clients4*/seed-42_run-*.json"))
            attack_record = json.loads(attack_path.read_text())
            attack_record["configuration"]["label_flipping"]["poisoned_client_ids"] = [0, 1, 2, 3]
            attack_path.write_text(json.dumps(attack_record))
            with self.assertRaisesRegex(ValueError, "Unmatched attackers"):
                load_comparison(root, spec)


if __name__ == "__main__":
    unittest.main()
