"""Família de descritores de FORMA E CONTORNO: momentos de Hu e propriedades de região.

Fundamentação: momentos de Hu (Hu, 1962) são invariantes a translação, escala e
rotação, úteis para caracterizar a morfologia de estruturas segmentadas mesmo sem
alinhamento prévio da imagem. Aqui aplicamos sobre uma segmentação simples por
limiar de intensidade (proxy para "vaso"/estrutura de interesse), documentando a
limitação de não termos uma segmentação anatômica supervisionada (discutido na
Conclusão do artigo como trabalho futuro).
"""

from __future__ import annotations

import numpy as np
from skimage import measure


def _binary_mask(image: np.ndarray, percentile: float = 85.0) -> np.ndarray:
    threshold = np.percentile(image, percentile)
    return image >= threshold


def extract_hu_moments(image: np.ndarray) -> dict:
    mask = _binary_mask(image).astype(np.float64)
    moments = measure.moments_central(mask)
    hu = measure.moments_hu(moments)
    # log-transform para estabilizar a escala dinâmica dos momentos de Hu (prática padrão).
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return {f"hu_moment_{i}": float(v) for i, v in enumerate(hu_log)}


def extract_region_properties(image: np.ndarray) -> dict:
    """Propriedades de região (área, excentricidade, solidez etc.) da maior componente conexa."""
    mask = _binary_mask(image)
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)

    if not props:
        return {
            "region_area_ratio": 0.0,
            "region_eccentricity": 0.0,
            "region_solidity": 0.0,
            "region_extent": 0.0,
            "region_perimeter_ratio": 0.0,
        }

    largest = max(props, key=lambda p: p.area)
    total_pixels = mask.size
    return {
        "region_area_ratio": float(largest.area / total_pixels),
        "region_eccentricity": float(largest.eccentricity),
        "region_solidity": float(largest.solidity),
        "region_extent": float(largest.extent),
        "region_perimeter_ratio": float(largest.perimeter / np.sqrt(total_pixels)),
    }


def extract_shape_features(image: np.ndarray) -> dict:
    features = {}
    features.update(extract_hu_moments(image))
    features.update(extract_region_properties(image))
    return features
