"""Orquestração da extração de características e agregação corte -> exame.

Decisão metodológica central (enunciado §4.2, destacada como "uma das decisões
mais importantes do trabalho"): para a tarefa volumétrica (TC 3D), extraímos
características POR CORTE e agregamos por EXAME via pooling estatístico
(média, desvio-padrão e percentis 90/95 por característica). A justificativa:

  1) a EP costuma ocupar poucos cortes dentro do exame (sinal esparso no eixo Z),
     então usar apenas a média mascararia o sinal — por isso incluímos também o
     máximo/percentis altos, que capturam o corte mais "anômalo" do exame;
  2) evita treinar sobre uma unidade de amostra (corte) diferente do rótulo
     disponível no baseline (exame), o que violaria o protocolo por paciente;
  3) é computacionalmente tratável para um baseline amplo em descritores x modelos,
     ao contrário de features 3D diretas (custosas) ou 2.5D (aumentam muito a
     dimensionalidade sem uma justificativa clara neste estágio).

Esta escolha é reavaliada na Conclusão do artigo como limitação/trabalho futuro.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.features.gradient import extract_gradient_features
from src.features.intensity import extract_intensity_features
from src.features.radiomics_features import extract_radiomics_features
from src.features.shape import extract_shape_features
from src.features.texture import extract_texture_features

logger = logging.getLogger(__name__)

FAMILY_EXTRACTORS = {
    "texture": extract_texture_features,
    "shape": extract_shape_features,
    "gradient": extract_gradient_features,
    "intensity": extract_intensity_features,
    "radiomics": extract_radiomics_features,
}

POOLING_FUNCS = {
    "mean": np.mean,
    "std": np.std,
    "p90": lambda x: np.percentile(x, 90),
    "max": np.max,
}


def extract_slice_features(image: np.ndarray, families: list[str] | None = None) -> dict:
    """Extrai todas as famílias de descritores solicitadas para um único corte 2D pré-processado."""
    families = families or list(FAMILY_EXTRACTORS.keys())
    features = {}
    for family in families:
        extractor = FAMILY_EXTRACTORS[family]
        try:
            features.update(extractor(image))
        except Exception as exc:
            logger.warning("Falha ao extrair família '%s': %s", family, exc)
    return features


def pool_exam_features(slice_features: list[dict]) -> dict:
    """Agrega uma lista de dicionários de features por corte em um único vetor por exame."""
    if not slice_features:
        return {}
    df = pd.DataFrame(slice_features)
    pooled = {}
    for col in df.columns:
        values = df[col].to_numpy(dtype=float)
        for pool_name, pool_fn in POOLING_FUNCS.items():
            pooled[f"{col}__{pool_name}"] = float(pool_fn(values))
    return pooled
