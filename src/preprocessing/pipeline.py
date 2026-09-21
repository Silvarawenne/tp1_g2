"""Pipeline de pré-processamento: janelamento, reamostragem isotrópica, recorte de ROI.

Cada etapa é justificada no artigo (Metodologia) e encapsulada aqui para que TODO o
pré-processamento ajustado a dados (normalização, PCA, SMOTE) seja parte do
sklearn.Pipeline ajustado *somente* na partição de treino de cada dobra (enunciado §4.4,
erro comum #2). As funções abaixo lidam com o pré-processamento de IMAGEM, que é
determinístico (não aprende parâmetros nos dados) e por isso pode ser aplicado antes
da divisão treino/teste sem causar vazamento.
"""

from __future__ import annotations

import numpy as np
from skimage.transform import resize

from src.config import DEFAULT_WINDOW, TARGET_SIZE, WINDOWS
from src.data.dicom_io import DicomSlice, apply_window


def resample_isotropic(image: np.ndarray, pixel_spacing: tuple[float, float], target_spacing: float = 1.0) -> np.ndarray:
    """Reamostra a imagem para espaçamento isotrópico (mm/pixel) fixo.

    Necessário porque PixelSpacing varia entre aquisições/scanners (enunciado §4.1);
    sem isso, descritores de textura/forma deixam de ser comparáveis entre exames.
    """
    row_spacing, col_spacing = pixel_spacing
    scale = (row_spacing / target_spacing, col_spacing / target_spacing)
    new_shape = (max(1, int(round(image.shape[0] * scale[0]))), max(1, int(round(image.shape[1] * scale[1]))))
    return resize(image, new_shape, preserve_range=True, anti_aliasing=True)


def crop_body_roi(image: np.ndarray, margin_ratio: float = 0.05) -> np.ndarray:
    """Recorte simples de região de interesse: remove bordas de fundo/mesa de exame.

    Baseline intencionalmente simples (recorte por margem fixa) — uma segmentação de
    pulmão/tórax mais sofisticada é discutida como trabalho futuro na Conclusão do artigo.
    """
    h, w = image.shape
    mh, mw = int(h * margin_ratio), int(w * margin_ratio)
    return image[mh : h - mh, mw : w - mw]


def preprocess_slice(dicom_slice: DicomSlice, window_name: str = DEFAULT_WINDOW, target_size: int = TARGET_SIZE) -> np.ndarray:
    """Aplica a sequência completa: janelamento -> reamostragem isotrópica -> recorte -> resize final.

    Retorna uma imagem 2D float em [0, 1] de tamanho (target_size, target_size).
    """
    window = WINDOWS[window_name]
    windowed = apply_window(dicom_slice.pixel_array_hu, window["center"], window["width"])
    resampled = resample_isotropic(windowed, dicom_slice.pixel_spacing)
    cropped = crop_body_roi(resampled)
    if cropped.shape[0] < 8 or cropped.shape[1] < 8:
        cropped = resampled
    final = resize(cropped, (target_size, target_size), preserve_range=True, anti_aliasing=True)
    return np.clip(final, 0.0, 1.0)
