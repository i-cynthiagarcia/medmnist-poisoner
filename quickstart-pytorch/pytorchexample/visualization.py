"""Plot helpers for persisted evaluation outputs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def save_confusion_matrix_plot(
    confusion_matrix: list[list[int]],
    class_names: list[str],
    output_path: Path,
    title: str,
) -> Path:
    """Save an annotated confusion matrix without replacing existing plots."""
    num_classes = len(class_names)
    if num_classes == 0 or len(confusion_matrix) != num_classes:
        raise ValueError("Confusion matrix dimensions do not match class names.")
    if any(len(row) != num_classes for row in confusion_matrix):
        raise ValueError("Confusion matrix must be square.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace existing plot: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_count = max(max(row) for row in confusion_matrix)
    figure, axis = plt.subplots(
        figsize=(max(8, num_classes * 0.8), max(7, num_classes * 0.7))
    )
    image = axis.imshow(confusion_matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, label="Image count")
    axis.set(
        xticks=range(num_classes),
        xticklabels=class_names,
        yticks=range(num_classes),
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = max_count / 2
    for row_index, row in enumerate(confusion_matrix):
        for column_index, count in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(count),
                ha="center",
                va="center",
                color="white" if count > threshold else "black",
                fontsize=8,
            )

    figure.tight_layout()
    try:
        figure.savefig(output_path, dpi=180, format="png")
    finally:
        plt.close(figure)
    return output_path
