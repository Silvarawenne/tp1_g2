"""Métricas apropriadas à tarefa (enunciado §4.4): classificação binária desbalanceada.

Acurácia isolada NÃO é aceita como métrica de desempenho (um classificador que
sempre prevê "negativo" atinge ~acurácia alta em bases desbalanceadas — erro comum
#3 do enunciado). Reportamos AUC-ROC, AUC-PR, sensibilidade, especificidade, F1,
acurácia balanceada e matriz de confusão.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    """y_score: probabilidade (ou score) da classe positiva, usado nas métricas baseadas em ranking."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "auc_roc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "auc_pr": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Resume média +- desvio-padrão entre dobras (enunciado §4.4: 'resultados com
    média e desvio-padrão entre dobras')."""
    keys = [k for k in fold_metrics[0] if k not in {"tn", "fp", "fn", "tp"}]
    summary = {}
    for key in keys:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values))
    return summary
