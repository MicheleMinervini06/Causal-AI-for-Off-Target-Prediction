from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_pr_roc(y_true: np.ndarray, y_score: np.ndarray, output_dir: str | Path, prefix: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_pr_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_roc_curve.png", dpi=160)
    plt.close()


def plot_importance_heatmap(
    importance: np.ndarray,
    output_path: str | Path,
    feature_names: list[str] | None = None,
) -> None:
    matrix = np.asarray(importance, dtype=float)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    sns.heatmap(matrix, cmap="mako", xticklabels=feature_names)
    plt.title("Feature Importance Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
