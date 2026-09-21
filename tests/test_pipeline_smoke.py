"""Teste de fumaça (smoke test): garante que leitura DICOM -> pré-processamento ->
extração de características -> protocolo de validação -> modelagem rodam sem erro
de ponta a ponta, usando um punhado de exames sintéticos gerados em tempo de teste.

Isto NÃO valida desempenho preditivo (os dados são sintéticos e sem significado
clínico) — apenas a integridade estrutural do pipeline, que é o requisito de
reprodutibilidade do enunciado (§4.5): "a partir de uma máquina limpa [...] deve
ser possível chegar aos mesmos números reportados".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dicom_io import load_exam_slices
from src.data.synthetic import generate_synthetic_dataset
from src.evaluation.protocol import run_cross_validation
from src.features.aggregate import extract_slice_features, pool_exam_features
from src.models.classical import build_trivial_baseline
from src.preprocessing.pipeline import preprocess_slice


def test_end_to_end_pipeline(tmp_path: Path):
    records = generate_synthetic_dataset(
        tmp_path / "synthetic", n_patients=10, slices_per_exam=4, size=64, seed=0
    )
    assert len(records) == 10

    rows = []
    for record in records:
        exam_dir = tmp_path / "synthetic" / record["patient_id"]
        exam_dir = next(exam_dir.iterdir())  # único StudyInstanceUID por paciente aqui
        slices = load_exam_slices(exam_dir)
        assert len(slices) == 4

        slice_feats = [extract_slice_features(preprocess_slice(s), families=["texture", "shape", "gradient", "intensity"]) for s in slices]
        exam_feats = pool_exam_features(slice_feats)
        exam_feats["patient_id"] = record["patient_id"]
        exam_feats["pe_present_on_exam"] = record["pe_present_on_exam"]
        rows.append(exam_feats)

    df = pd.DataFrame(rows)
    assert df.shape[0] == 10
    assert df.drop(columns=["patient_id", "pe_present_on_exam"]).shape[1] > 10

    X = df.drop(columns=["patient_id", "pe_present_on_exam"]).select_dtypes(include=["number"]).fillna(0.0)
    y = df["pe_present_on_exam"].to_numpy()
    groups = df["patient_id"].to_numpy()

    result = run_cross_validation(build_trivial_baseline(), X, y, groups, param_grid=None, n_splits=2)
    assert "auc_roc_mean" in result["summary"]
    assert not np.isnan(result["summary"]["balanced_accuracy_mean"])
