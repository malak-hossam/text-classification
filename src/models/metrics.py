from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def compute_metrics_bundle(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    accuracy = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred).tolist()
    return {"accuracy": accuracy, "classification_report": report, "confusion_matrix": matrix}


def hf_compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float(accuracy_score(labels, preds))}


def save_json(data: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_confusion_matrix_plot(
    matrix: Sequence[Sequence[int]],
    output_path: str | Path,
    class_names: Sequence[str] | None = None,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    array = np.array(matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(array, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if class_names:
        ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)), class_names)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            ax.text(j, i, str(array[i, j]), ha="center", va="center")
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

