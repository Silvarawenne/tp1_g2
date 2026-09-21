#!/usr/bin/env python3
"""Roda a grade experimental completa: baseline trivial + modelos clássicos sobre a
matriz de características, com StratifiedGroupKFold por paciente, e salva:
  - outputs/tables/results_summary.csv        (tabela comparativa média +- desvio)
  - outputs/figures/roc_pr_<melhor_modelo>.png (curvas ROC/PR do melhor modelo)
  - outputs/figures/confusion_matrix_<melhor_modelo>.png
  - outputs/models/<modelo>.joblib             (modelos finais serializados)

Uso:
    python scripts/run_experiment.py --features outputs/tables/features.parquet

ATENÇÃO: os números produzidos a partir de outputs/tables/features.parquet gerado
com dados SINTÉTICOS (ver src/data/synthetic.py) não têm valor científico — servem
apenas para validar que o pipeline roda ponta a ponta. Rode este mesmo script sobre
as características extraídas da amostra REAL do RSNA STR PE Dataset antes de
reportar qualquer número no artigo (ver README.md).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from src.config import FIGURES_DIR, GROUP_COLUMN, MODELS_DIR, TABLES_DIR, TARGET_COLUMN
from src.evaluation.protocol import run_cross_validation
from src.models.classical import PARAM_GRIDS, build_model_zoo, build_trivial_baseline
from src.utils.seed import set_global_seed
from src.config import SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = {TARGET_COLUMN, GROUP_COLUMN, "study_instance_uid"}


def load_dataset(features_path: Path):
    df = pd.read_parquet(features_path)
    y = df[TARGET_COLUMN].to_numpy()
    groups = df[GROUP_COLUMN].to_numpy()
    X = df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])
    X = X.select_dtypes(include=["number"]).fillna(0.0)
    return X, y, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=str, default=str(TABLES_DIR / "features.parquet"))
    parser.add_argument("--tune", action="store_true", help="Ativa busca de hiperparâmetros (GridSearchCV aninhado)")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    set_global_seed(SEED)

    X, y, groups = load_dataset(Path(args.features))
    logger.info("Dataset: X=%s, positivos=%d/%d, pacientes únicos=%d", X.shape, int(y.sum()), len(y), len(set(groups)))

    results = {}

    logger.info("Rodando baseline trivial...")
    trivial = build_trivial_baseline()
    results["baseline_trivial"] = run_cross_validation(trivial, X, y, groups, param_grid=None, n_splits=args.n_splits)

    zoo = build_model_zoo()
    for name, pipeline in zoo.items():
        logger.info("Rodando modelo: %s", name)
        grid = PARAM_GRIDS.get(name) if args.tune else None
        results[name] = run_cross_validation(pipeline, X, y, groups, param_grid=grid, n_splits=args.n_splits)

    summary_rows = []
    for name, res in results.items():
        row = {"model": name, **res["summary"]}
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("auc_roc_mean", ascending=False)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = TABLES_DIR / "results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Tabela comparativa salva em %s", summary_path)
    print(summary_df.to_string(index=False))

    best_model_name = summary_df.iloc[0]["model"]
    best_result = results[best_model_name]
    _plot_diagnostics(best_model_name, best_result)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, res in results.items():
        joblib.dump(res["final_model"], MODELS_DIR / f"{name}.joblib")
    logger.info("Modelos finais salvos em %s", MODELS_DIR)


def _plot_diagnostics(model_name: str, result: dict) -> None:
    """Gera curva ROC/PR e matriz de confusão (concatenando as predições de todas as
    dobras out-of-fold) para o melhor modelo — enunciado §4.4: 'ao menos uma figura
    (curva ROC/PR)' e 'matriz de confusão'."""
    import numpy as np

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    preds = result["predictions"]
    y_true = np.concatenate([p["y_true"] for p in preds])
    y_pred = np.concatenate([p["y_pred"] for p in preds])
    y_score = np.concatenate([p["y_score"] for p in preds])

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, name=model_name)
    ax.set_title(f"Curva ROC (out-of-fold) — {model_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"roc_{model_name}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax, name=model_name)
    ax.set_title(f"Curva Precisão-Recall (out-of-fold) — {model_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"pr_{model_name}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, display_labels=["negativo", "positivo"])
    ax.set_title(f"Matriz de confusão (out-of-fold) — {model_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"confusion_matrix_{model_name}.png", dpi=150)
    plt.close(fig)

    logger.info("Figuras de diagnóstico salvas em %s (modelo=%s)", FIGURES_DIR, model_name)


if __name__ == "__main__":
    main()
