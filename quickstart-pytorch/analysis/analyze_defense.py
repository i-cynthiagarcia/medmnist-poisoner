"""Reproduce the frozen validation-only defense comparisons."""

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis" / "outputs"
SEEDS = tuple(range(42, 52))
AGGREGATIONS = ("fedavg", "median")
CLASSES = (
    "bladder", "femur-left", "femur-right", "heart", "kidney-left",
    "kidney-right", "liver", "lung-left", "lung-right", "pancreas", "spleen",
)
COMPARISONS = (
    ("organamnist", "protocol-v3", "run-2", 1),
    ("organcmnist", "protocol-v8", "run-1", 3),
    ("organsmnist", "protocol-v8", "run-2", 3),
)


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def finite(value, name, rate=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid {name}: {value!r}")
    if rate and not 0 <= value <= 1:
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def load_comparison(root, spec):
    dataset, protocol, execution, epochs = spec
    directory = root / "results" / "defense" / "generation-v1" / protocol / execution
    records = {}
    partitions = {}
    selections = {}
    supports = None
    for path in sorted(directory.rglob("seed-*_run-*.json")):
        record = read_json(path)
        config = record["configuration"]
        attack = config["label_flipping"]
        seed, aggregation, count = record["seed"], record["aggregation_id"], attack["num_poisoned_clients"]
        key = (seed, aggregation, count)
        expected_experiment = (
            f"{'median_' if aggregation == 'median' else ''}label_flip_r5_e{epochs}_"
            f"lr0p010_clients{count}_f{'1p000' if count == 4 else '0p000'}_s4_t5"
        )
        expected_dir = directory / aggregation / dataset / expected_experiment
        training = record["training_results"]
        if (
            key not in {(s, a, c) for s in SEEDS for a in AGGREGATIONS for c in (0, 4)}
            or key in records
            or path.parent != expected_dir
            or path.name != f"seed-{seed}_run-{record['run_id']}.json"
            or record["generation_id"] != "generation-v1"
            or record["protocol_id"] != protocol
            or record["execution_id"] != execution
            or record["dataset"] != dataset
            or record["experiment"] != expected_experiment
            or record["status"] != "completed"
            or record["loss_aggregation"] != "per_image"
            or not all(training[k] is True for k in ("valid_run", "metrics_finite", "model_parameters_finite"))
            or training["received_results_by_round"] != {str(i): 20 for i in range(1, 6)}
            or training["client_failures_by_round"] != {str(i): 0 for i in range(1, 6)}
            or (config["clients"], config["federated_rounds"], config["local_epochs"],
                config["learning_rate"], config["batch_size"], config["require_complete_round"])
            != (20, 5, epochs, 0.01, 32, True)
            or (attack["source_class_id"], attack["target_class_id"],
                attack["fraction_of_source_labels_flipped_per_poisoned_client"])
            != (4, 5, float(count == 4))
        ):
            raise ValueError(f"Invalid or duplicate condition in {path}")
        metrics = record["metrics"]
        if metrics["test_final"] is not None or metrics["targeted_source_to_target_misclassification"]["test_final"] is not None:
            raise ValueError(f"Unexpected test evaluation in {path}")
        if tuple(metrics["class_names"]) != CLASSES:
            raise ValueError(f"Unexpected classes in {path}")
        validation = metrics["validation_by_round"]["5"]
        per_class = validation["per_class"]
        if set(per_class) != set(CLASSES):
            raise ValueError(f"Incomplete class metrics in {path}")
        recalls = {name: finite(per_class[name]["recall"], f"{path}: {name} recall", rate=True) for name in CLASSES}
        current_supports = {name: per_class[name]["support"] for name in CLASSES}
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in current_supports.values()):
            raise ValueError(f"Invalid class support in {path}")
        if supports is not None and current_supports != supports:
            raise ValueError(f"Validation supports differ in {path}")
        supports = current_supports
        target = metrics["targeted_source_to_target_misclassification"]["validation_by_round"]["5"]
        if target["support"] != supports["kidney-left"]:
            raise ValueError(f"Targeted error support mismatch in {path}")
        values = {
            "accuracy": finite(validation["accuracy"], f"{path}: accuracy", rate=True),
            "targeted_error": finite(target["rate"], f"{path}: targeted error", rate=True),
            "kidney_left_recall": recalls["kidney-left"],
            "macro_recall": statistics.fmean(recalls.values()),
            "loss": finite(validation["loss"], f"{path}: loss"),
        }
        partition = read_json(root / record["partition_manifest"])
        if (
            (partition["dataset"], partition["seed"], partition["num_clients"]) != (dataset, seed, 20)
            or not all(partition["verification"][k] is True for k in
                       ("partitions_are_disjoint", "all_training_examples_assigned", "each_training_example_assigned_once"))
            or (seed in partitions and partitions[seed] != partition["clients"])
        ):
            raise ValueError(f"Invalid or unmatched partition in {path}")
        partitions[seed] = partition["clients"]
        if count == 4:
            selected = attack["poisoned_client_ids"]
            if len(set(selected)) != 4 or (seed in selections and selections[seed] != selected):
                raise ValueError(f"Unmatched attackers in {path}")
            selections[seed] = selected
        records[key] = {
            "values": values, "recalls": recalls, "supports": supports,
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    expected = {(seed, agg, count) for seed in SEEDS for agg in AGGREGATIONS for count in (0, 4)}
    if set(records) != expected:
        raise ValueError(f"{directory}: missing conditions: {sorted(expected - set(records))}")
    return records


def build_tables(spec, records):
    dataset, protocol, execution, epochs = spec
    base = {"dataset": dataset, "protocol": protocol, "execution": execution, "local_epochs": epochs}
    sources, paired, classes = [], [], []
    for seed in SEEDS:
        for aggregation in AGGREGATIONS:
            for count in (0, 4):
                record = records[seed, aggregation, count]
                sources.append({**base, "seed": seed, "aggregation": aggregation,
                                "condition": "attacked" if count else "clean",
                                "record": record["path"], "sha256": record["sha256"]})
        for metric in ("accuracy", "targeted_error", "kidney_left_recall", "macro_recall", "loss"):
            fc, fa, mc, ma = (records[seed, agg, count]["values"][metric]
                              for agg, count in (("fedavg", 0), ("fedavg", 4), ("median", 0), ("median", 4)))
            paired.append({**base, "seed": seed, "metric": metric,
                           "fedavg_clean": fc, "fedavg_attacked": fa, "fedavg_change": fa - fc,
                           "median_clean": mc, "median_attacked": ma, "median_change": ma - mc,
                           "interaction": (ma - mc) - (fa - fc)})
        for name in CLASSES:
            fc, fa, mc, ma = (records[seed, agg, count]["recalls"][name]
                              for agg, count in (("fedavg", 0), ("fedavg", 4), ("median", 0), ("median", 4)))
            classes.append({**base, "seed": seed, "class": name,
                            "support": records[seed, "fedavg", 0]["supports"][name],
                            "fedavg_clean": fc, "fedavg_attacked": fa, "fedavg_change": fa - fc,
                            "median_clean": mc, "median_attacked": ma, "median_change": ma - mc,
                            "interaction": (ma - mc) - (fa - fc)})
    return sources, paired, classes


def summarize(rows, group_key):
    summaries = []
    for spec in COMPARISONS:
        dataset, protocol, execution, epochs = spec
        subset = [row for row in rows if row["dataset"] == dataset]
        for group in dict.fromkeys(row[group_key] for row in subset):
            group_rows = [row for row in subset if row[group_key] == group]
            row = {"dataset": dataset, "protocol": protocol, "execution": execution,
                   "local_epochs": epochs, group_key: group, "n": len(group_rows)}
            if group_key == "class":
                row["support"] = group_rows[0]["support"]
            for field in ("fedavg_clean", "fedavg_attacked", "fedavg_change", "median_clean",
                          "median_attacked", "median_change", "interaction"):
                values = [item[field] for item in group_rows]
                row[f"{field}_mean"] = statistics.fmean(values)
                row[f"{field}_sd"] = statistics.stdev(values)
            summaries.append(row)
    return summaries


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plots(paired, output):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, constrained_layout=True)
    for ax, (dataset, protocol, execution, _) in zip(axes, COMPARISONS):
        rows = [r for r in paired if r["dataset"] == dataset and r["metric"] == "targeted_error"]
        values = [100 * r["interaction"] for r in rows]
        ax.axhline(0, color="black", linewidth=0.8)
        ax.scatter(SEEDS, values, color="#356a9a")
        ax.axhline(statistics.fmean(values), color="#b04638", linestyle="--", label="Mean")
        ax.set_title(f"{dataset}\n{protocol}/{execution}")
        ax.set_xlabel("Seed")
        ax.set_xticks(SEEDS[::2])
    axes[0].set_ylabel("Targeted-error interaction (pp)")
    axes[-1].legend()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"targeted_error_interaction.{suffix}", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, constrained_layout=True)
    for ax, (dataset, protocol, execution, _) in zip(axes, COMPARISONS):
        rows = [r for r in paired if r["dataset"] == dataset and r["metric"] == "targeted_error"]
        for agg, color in (("fedavg", "#356a9a"), ("median", "#b04638")):
            for row in rows:
                ax.plot((0, 1), [100 * row[f"{agg}_clean"], 100 * row[f"{agg}_attacked"]],
                        color=color, alpha=0.25, linewidth=1)
            ax.plot((0, 1), [100 * statistics.fmean(r[f"{agg}_{condition}"] for r in rows)
                             for condition in ("clean", "attacked")],
                    color=color, marker="o", linewidth=2, label=agg)
        ax.set_title(f"{dataset}\n{protocol}/{execution}")
        ax.set_xticks((0, 1), ("Clean", "Attacked"))
    axes[0].set_ylabel("Targeted error (%)")
    axes[-1].legend()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"targeted_error_conditions.{suffix}", dpi=200)
    plt.close(fig)


def main():
    all_sources, all_paired, all_classes = [], [], []
    for spec in COMPARISONS:
        tables = build_tables(spec, load_comparison(ROOT, spec))
        all_sources.extend(tables[0])
        all_paired.extend(tables[1])
        all_classes.extend(tables[2])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("sources.csv", all_sources),
        ("paired_by_seed.csv", all_paired),
        ("metric_summary.csv", summarize(all_paired, "metric")),
        ("class_recall_by_seed.csv", all_classes),
        ("class_recall_summary.csv", summarize(all_classes, "class")),
    ):
        write_csv(OUTPUT / name, rows)
    plots(all_paired, OUTPUT)
    print(f"Wrote analysis to {OUTPUT}")


if __name__ == "__main__":
    main()
