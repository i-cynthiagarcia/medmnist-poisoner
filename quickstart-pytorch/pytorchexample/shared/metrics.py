"""Shared serialization helpers for training and evaluation metrics."""

from flwr.app import MetricRecord


def json_value(value):
    """Convert Flower metric values to JSON-compatible primitives."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def serialize_evaluation_metrics(
    metric_record: MetricRecord, class_names: list[str]
) -> dict:
    """Convert flat Flower metrics into a readable per-class JSON structure."""
    num_classes = len(class_names)
    per_class_values = {
        key: json_value(metric_record[key])
        for key in (
            "per_class_precision",
            "per_class_recall",
            "per_class_f1",
            "per_class_support",
        )
    }
    if any(len(values) != num_classes for values in per_class_values.values()):
        raise ValueError("Per-class metrics do not match the dataset class count.")

    confusion_values = json_value(metric_record["confusion_matrix"])
    if len(confusion_values) != num_classes * num_classes:
        raise ValueError("Confusion matrix does not match the dataset class count.")

    return {
        "loss": json_value(metric_record["loss"]),
        "accuracy": json_value(metric_record["accuracy"]),
        "per_class": {
            class_name: {
                "precision": per_class_values["per_class_precision"][index],
                "recall": per_class_values["per_class_recall"][index],
                "f1": per_class_values["per_class_f1"][index],
                "support": per_class_values["per_class_support"][index],
            }
            for index, class_name in enumerate(class_names)
        },
        "confusion_matrix": [
            confusion_values[row * num_classes : (row + 1) * num_classes]
            for row in range(num_classes)
        ],
    }
