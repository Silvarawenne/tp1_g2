"""Família de descritores de INTENSIDADE: histograma, momentos estatísticos e percentis.

Simples, porém essencial como referência: quantifica quanto do sinal discriminativo
já está presente na distribuição bruta de HU/intensidade dentro da janela aplicada,
antes de qualquer descritor estrutural mais sofisticado (ponto de comparação citado
na pergunta-chave do enunciado: "quais características carregam sinal?").
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


def extract_intensity_features(image: np.ndarray, n_bins: int = 16) -> dict:
    flat = image.ravel()
    hist, _ = np.histogram(flat, bins=n_bins, range=(0.0, 1.0), density=True)

    features = {f"intensity_hist_bin{i:02d}": float(v) for i, v in enumerate(hist)}
    features.update(
        {
            "intensity_mean": float(np.mean(flat)),
            "intensity_std": float(np.std(flat)),
            "intensity_skewness": float(skew(flat)),
            "intensity_kurtosis": float(kurtosis(flat)),
            "intensity_p10": float(np.percentile(flat, 10)),
            "intensity_p50": float(np.percentile(flat, 50)),
            "intensity_p90": float(np.percentile(flat, 90)),
            "intensity_iqr": float(np.percentile(flat, 75) - np.percentile(flat, 25)),
        }
    )
    return features
