"""Protocolo experimental: particionamento por paciente + validação cruzada + ajuste
de hiperparâmetros em validação separada (nunca no conjunto de teste).

Requisito crítico (enunciado §4.4, erro comum #1): "Particionamento por paciente e
exame, jamais por corte/imagem." Usamos StratifiedGroupKFold, em que o `group`
é o `patient_id` — garante que nenhum paciente apareça simultaneamente em treino
e teste, mesmo quando o exame gera múltiplos cortes/derivações.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from src.config import N_SPLITS, SEED
from src.evaluation.metrics import aggregate_fold_metrics, compute_metrics


def run_cross_validation(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    param_grid: dict | None = None,
    n_splits: int = N_SPLITS,
    inner_splits: int = 3,
) -> dict:
    """Executa StratifiedGroupKFold externo; se `param_grid` for fornecido, faz
    busca de hiperparâmetros em validação cruzada aninhada (GridSearchCV interno,
    também agrupado por paciente) dentro de cada dobra externa de treino.

    Retorna métricas por dobra, resumo (média +- desvio) e o modelo final ajustado
    em todos os dados (para inspeção de importância de características).
    """
    outer_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    fold_metrics = []
    fold_predictions = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        if param_grid:
            inner_cv = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=SEED)
            search = GridSearchCV(model, param_grid, scoring="roc_auc", cv=inner_cv, n_jobs=-1, refit=True)
            search.fit(X_train, y_train, groups=groups_train)
            fitted = search.best_estimator_
        else:
            fitted = model
            fitted.fit(X_train, y_train)

        y_pred = fitted.predict(X_test)
        if hasattr(fitted, "predict_proba"):
            y_score = fitted.predict_proba(X_test)[:, 1]
        else:
            y_score = fitted.decision_function(X_test)

        m = compute_metrics(y_test, y_pred, y_score)
        m["fold"] = fold_idx
        fold_metrics.append(m)
        fold_predictions.append({"y_true": y_test, "y_pred": y_pred, "y_score": y_score, "test_idx": test_idx})

    summary = aggregate_fold_metrics(fold_metrics)

    # Ajuste final em todos os dados para inspeção de importância/serialização (não usado para reportar métricas).
    final_model = model
    final_model.fit(X, y)

    return {
        "fold_metrics": fold_metrics,
        "summary": summary,
        "predictions": fold_predictions,
        "final_model": final_model,
    }
