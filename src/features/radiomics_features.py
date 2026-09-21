"""Família RADIÔMICA: wrapper para PyRadiomics (IBSI-compatível).

Fundamentação: van Griethuysen et al. (2017) — PyRadiomics é a implementação de
referência, compatível com o padrão IBSI, amplamente usada em radiômica clínica.
Extraímos aqui um subconjunto 2D (first order + GLCM/GLRLM/GLSZM nativos do
PyRadiomics) por corte, tratando a própria imagem pré-processada como ROI (máscara
= tudo diferente de zero), já que não há segmentação anatômica disponível no baseline.

PyRadiomics é uma dependência pesada e opcional: se não estiver instalada, a função
retorna um dicionário vazio e registra um aviso, para que o restante do pipeline
continue funcionando (as outras >=4 famílias de descritores já satisfazem o mínimo
de 3 exigido pelo enunciado §4.2).
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)

try:
    import SimpleITK as sitk
    from radiomics import featureextractor
    from radiomics import logger as _radiomics_logger

    _radiomics_logger.setLevel(logging.ERROR)  # a extração roda por corte; log INFO padrão é extremamente verboso
    _PYRADIOMICS_AVAILABLE = True
except ImportError:  # pragma: no cover - ambiente sem pyradiomics instalado
    _PYRADIOMICS_AVAILABLE = False


def extract_radiomics_features(image: np.ndarray) -> dict:
    if not _PYRADIOMICS_AVAILABLE:
        logger.warning("pyradiomics/SimpleITK não instalados — pulando família radiômica.")
        return {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_uint = (np.clip(image, 0.0, 1.0) * 4095).astype(np.int32)
        sitk_image = sitk.GetImageFromArray(img_uint[np.newaxis, :, :])
        mask = (img_uint > 0).astype(np.uint8)
        if mask.sum() == 0:
            return {}
        sitk_mask = sitk.GetImageFromArray(mask[np.newaxis, :, :])

        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName("firstorder")
        extractor.enableFeatureClassByName("glcm")
        extractor.enableFeatureClassByName("glszm")

        try:
            result = extractor.execute(sitk_image, sitk_mask)
        except Exception as exc:  # radiomics levanta várias exceções específicas
            logger.warning("Falha ao extrair radiômica: %s", exc)
            return {}

    features = {}
    for key, value in result.items():
        if key.startswith("diagnostics_"):
            continue
        try:
            features[f"radiomics_{key}"] = float(value)
        except (TypeError, ValueError):
            continue
    return features
