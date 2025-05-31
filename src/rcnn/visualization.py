"""visualization.py
Utility functions for inspecting training progress and model performance.

Example
-------
>>> from rcnn.visualization import plot_history, print_classification_report

"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import classification_report, confusion_matrix

__all__ = [
    "plot_history",
    "history_to_polars",
    "print_classification_report",
    "plot_confusion_matrix",
]

# helper


def _to_numpy(obj):
    """Convert *obj* to NumPy array if it is possible."""

    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "to_numpy"):
        return obj.to_numpy()
    return np.asarray(obj)


# training


def plot_history(
    history: Mapping[str, Sequence[float]],
    metrics: Iterable[str] | None = None,
    separate: bool = False,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot training curves stored in *history*.

    Args:
        history (Mapping[str, Sequence[float]]): Dict containing metric names as keys
            and a list of values (one per epoch) as values. Keys: `loss`, `val_loss`,
            `acc`, `val_acc`.
        metrics (Iterable[str] | None, optional): Subset of *history* keys to plot.
            If *None*, all keys are used. Defaults to None.
        separate (bool, optional): Wheteher to separate the metrics in different
            figures. Defaults to False.
        save_path (str | Path | None, optional): Whether to save the figure to this path.
            Defaults to None.
        show (bool, optional): Whether to display the plot immediately via `plt.show()`.
            Defaults to True.
    """
    metrics = list(metrics or history.keys())

    if not separate:
        plt.figure(figsize=(6, 4))
        for key in metrics:
            if key in history:
                plt.plot(
                    history[key],
                    label=(
                        f"train_{key}"
                        if "val_" not in key
                        else f"test_{key.split('_')[1]}"
                    ),
                )
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
    else:
        for key in metrics:
            if "val_" in key or key not in history:
                continue
            plt.figure(figsize=(6, 4))
            plt.plot(history[key], label="train")
            plt.plot(history["val_" + key], label="test")
            plt.title(key)
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()

            if save_path is not None and save_path.is_file():
                path, ext = save_path.split(".")
                plt.savefig(f"{path}_{key}.{ext}")

    if show:
        plt.show()
    else:
        plt.close()


def history_to_polars(history: Mapping[str, Sequence[float]]) -> pl.DataFrame:
    """Convert a history dict to a ``polars.DataFrame`` for inspection."""

    max_len = max(len(v) for v in history.values())
    padded = {k: list(v) + [None] * (max_len - len(v)) for k, v in history.items()}
    return (
        pl.DataFrame(padded)
        .with_columns(pl.Series("epoch", range(1, max_len + 1)))
        .select(["epoch", *history.keys()])
    )


# classification


def print_classification_report(
    y_true, y_pred, target_names: Sequence[str] | None = None, digits: int = 4
) -> None:
    """Compute and print skelarn's classification report."""

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    report = classification_report(
        y_true, y_pred, target_names=target_names, digits=digits
    )
    print(report)


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: Sequence[str] | None = None,
    normalize: bool | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot confusion matrix."""

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(
        y_true, y_pred, labels=labels, normalize="true" if normalize else None
    )

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = range(len(labels) if labels else cm.shape[0])
    plt.xticks(tick_marks, labels or tick_marks)
    plt.yticks(tick_marks, labels or tick_marks)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        fmt = "{:.2f}" if normalize else "{:.0f}"
        plt.text(
            j,
            i,
            fmt.format(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()
