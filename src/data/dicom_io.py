"""Leitura de DICOM e conversão para unidades físicas (Hounsfield Units).

Referência metodológica: a conversão pixel -> HU via RescaleSlope/RescaleIntercept
e o janelamento (WindowCenter/WindowWidth) são etapas obrigatórias antes de qualquer
extração de características em CT (enunciado §4.1; erro comum #6 do enunciado §10).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom


@dataclass
class DicomSlice:
    """Um corte de CT já convertido para HU, com os metadados relevantes."""

    pixel_array_hu: np.ndarray
    patient_id: str
    study_instance_uid: str
    sop_instance_uid: str
    instance_number: int
    pixel_spacing: tuple[float, float]
    slice_thickness: float
    rescale_slope: float
    rescale_intercept: float
    modality: str


def read_dicom_slice(path: str | Path) -> DicomSlice:
    """Lê um arquivo DICOM e converte a matriz de pixels para Hounsfield Units (HU).

    HU = pixel_value * RescaleSlope + RescaleIntercept
    """
    ds = pydicom.dcmread(str(path))

    raw = ds.pixel_array.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = raw * slope + intercept

    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    pixel_spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]))

    return DicomSlice(
        pixel_array_hu=hu,
        patient_id=str(getattr(ds, "PatientID", "unknown")),
        study_instance_uid=str(getattr(ds, "StudyInstanceUID", "unknown")),
        sop_instance_uid=str(getattr(ds, "SOPInstanceUID", Path(path).stem)),
        instance_number=int(getattr(ds, "InstanceNumber", 0)),
        pixel_spacing=pixel_spacing,
        slice_thickness=float(getattr(ds, "SliceThickness", 1.0)),
        rescale_slope=slope,
        rescale_intercept=intercept,
        modality=str(getattr(ds, "Modality", "CT")),
    )


def apply_window(hu_array: np.ndarray, center: float, width: float) -> np.ndarray:
    """Aplica janelamento (window center/width) e normaliza para [0, 1].

    Valores fora da janela são truncados (clip), como de praxe em leitura radiológica.
    """
    low = center - width / 2.0
    high = center + width / 2.0
    windowed = np.clip(hu_array, low, high)
    return (windowed - low) / (high - low)


def load_exam_slices(exam_dir: str | Path) -> list[DicomSlice]:
    """Carrega e ordena (por InstanceNumber) todos os cortes de um exame (uma pasta = um StudyInstanceUID)."""
    exam_dir = Path(exam_dir)
    slices = [read_dicom_slice(p) for p in sorted(exam_dir.glob("*.dcm"))]
    slices.sort(key=lambda s: s.instance_number)
    return slices
