import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import RESULTS_DIR

LABEL_NAMES = ["Sell", "Hold", "Buy"]


def _plot_confusion_matrix(y_true, y_pred, title, ax):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")


def plot_comparison(results):
    """results: list of (name, y_true, y_pred, accuracy)"""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, y_true, y_pred, acc) in zip(axes, results):
        _plot_confusion_matrix(y_true, y_pred, f"{name}\nAccuracy={acc:.4f}", ax)

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved to {out_path}")


def print_summary(results):
    print("\n" + "=" * 50)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 50)
    for name, _, _, acc in results:
        print(f"  {name:<20} Accuracy: {acc:.4f}")
    best = max(results, key=lambda x: x[3])
    print(f"\nBest model: {best[0]}  ({best[3]:.4f})")
    print("=" * 50)
