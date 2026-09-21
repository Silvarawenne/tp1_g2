"""Família de descritores de GRADIENTE E BORDAS: HOG.

Fundamentação: Dalal & Triggs (2005) mostram que histogramas de gradientes
orientados, agregados em blocos normalizados localmente, são robustos a variações
de iluminação/contraste e capturam bem estruturas alongadas — potencialmente
relevante para o formato tubular/linear de defeitos de enchimento vasculares.
"""

from __future__ import annotations

import numpy as np
from skimage.feature import hog


def extract_hog(image: np.ndarray, pixels_per_cell: tuple[int, int] = (16, 16), cells_per_block: tuple[int, int] = (2, 2), orientations: int = 9) -> dict:
    features = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        feature_vector=True,
    )
    # HOG bruto tem alta dimensionalidade; resumimos em estatísticas por robustez
    # e para manter o vetor de características em escala comparável às demais famílias
    # (a redução de dimensionalidade formal — PCA/seleção — ocorre depois, no Pipeline).
    return {
        "hog_mean": float(np.mean(features)),
        "hog_std": float(np.std(features)),
        "hog_max": float(np.max(features)),
        "hog_p90": float(np.percentile(features, 90)),
        "hog_energy": float(np.sum(features**2)),
    }


def extract_gradient_features(image: np.ndarray) -> dict:
    return extract_hog(image)
