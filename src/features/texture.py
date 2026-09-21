"""Família de descritores de TEXTURA: GLCM/Haralick e LBP.

Fundamentação:
- Haralick, Shanmugam & Dinstein (1973) propõem os descritores de co-ocorrência
  (GLCM) — contraste, homogeneidade, energia, correlação — amplamente usados em
  CAD de imagens médicas por capturarem padrões de heterogeneidade de tecido.
- Ojala, Pietikäinen & Mäenpää (2002) propõem o LBP uniforme e invariante à
  rotação, robusto a variações monotônicas de iluminação/janela — relevante aqui
  porque o janelamento em HU pode variar entre protocolos de aquisição.
Ver references/fichamento.md para a leitura crítica completa.
"""

from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def _to_uint8(image: np.ndarray, n_levels: int = 32) -> np.ndarray:
    """Quantiza uma imagem float [0,1] para n_levels níveis inteiros (necessário para GLCM)."""
    quantized = np.clip(image, 0.0, 1.0) * (n_levels - 1)
    return quantized.astype(np.uint8)


def extract_glcm_haralick(image: np.ndarray, distances=(1, 2), angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4), n_levels: int = 32) -> dict:
    """Extrai propriedades de Haralick da matriz de co-ocorrência (GLCM), com média
    sobre distâncias/ângulos (para reduzir dimensionalidade e ganhar invariância
    aproximada à orientação, cf. prática comum na literatura de textura radiográfica).
    """
    img_q = _to_uint8(image, n_levels)
    glcm = graycomatrix(img_q, distances=list(distances), angles=list(angles), levels=n_levels, symmetric=True, normed=True)

    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    features = {}
    for prop in props:
        values = graycoprops(glcm, prop)
        features[f"glcm_{prop}_mean"] = float(np.mean(values))
        features[f"glcm_{prop}_std"] = float(np.std(values))
    return features


def extract_lbp(image: np.ndarray, radius: int = 2, n_points: int | None = None, method: str = "uniform") -> dict:
    """Extrai o histograma normalizado do LBP uniforme (invariante à rotação)."""
    if n_points is None:
        n_points = 8 * radius
    img_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    lbp = local_binary_pattern(img_uint8, P=n_points, R=radius, method=method)

    n_bins = n_points + 2  # padrões uniformes + 1 bin "não uniforme"
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    return {f"lbp_bin{i:02d}": float(v) for i, v in enumerate(hist)}


def extract_texture_features(image: np.ndarray) -> dict:
    features = {}
    features.update(extract_glcm_haralick(image))
    features.update(extract_lbp(image))
    return features
