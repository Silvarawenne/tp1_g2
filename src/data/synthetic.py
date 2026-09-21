"""Gerador de exames DICOM SINTÉTICOS para validar o pipeline ponta a ponta.

IMPORTANTE — leia antes de usar:
Este módulo NÃO substitui os dados reais do desafio RSNA 2020 (STR Pulmonary Embolism
Detection). Ele existe porque o ambiente usado para montar este repositório não tem
acesso ao Kaggle. Os "exames" gerados aqui são elipsoides sintéticos com ruído,
correlacionados artificialmente com um rótulo binário, e servem apenas para:

  1) provar que a leitura DICOM, o pré-processamento, a extração de características,
     o protocolo de validação e os modelos executam sem erro, de ponta a ponta;
  2) dar ao grupo um exemplo reproduzível de como rodar `scripts/run_experiment.py`.

Os números produzidos a partir destes dados NÃO têm significado clínico e NÃO devem
ser reportados no artigo como resultado científico. Assim que a amostra real do
RSNA STR PE Dataset estiver em `data/raw/` (ver README.md), o mesmo código de
extração/modelagem deve ser reexecutado sobre ela.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from src.config import ROOT_DIR, SEED

RNG = np.random.default_rng(SEED)


def _make_synthetic_hu_slice(size: int, positive: bool, rng: np.random.Generator) -> np.ndarray:
    """Cria uma matriz HU sintética: fundo de 'parênquima', um 'vaso' central e,
    se positive=True, uma inclusão de alta atenuação (proxy de trombo) dentro do vaso.
    """
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2, size / 2

    # Fundo tipo tecido mole/pulmão com ruído gaussiano.
    hu = rng.normal(loc=-500, scale=40, size=(size, size))

    # "Vaso" (estrutura circular de atenuação de partes moles / contraste).
    vessel_radius = size * 0.18
    vessel_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= vessel_radius**2
    hu[vessel_mask] = rng.normal(loc=250, scale=25, size=vessel_mask.sum())  # vaso contrastado

    if positive:
        # Inclusão excêntrica de baixa atenuação dentro do vaso (proxy de trombo/EP).
        offset_y = rng.uniform(-0.4, 0.4) * vessel_radius
        offset_x = rng.uniform(-0.4, 0.4) * vessel_radius
        clot_radius = vessel_radius * rng.uniform(0.3, 0.55)
        clot_mask = (yy - (cy + offset_y)) ** 2 + (xx - (cx + offset_x)) ** 2 <= clot_radius**2
        clot_mask &= vessel_mask
        hu[clot_mask] = rng.normal(loc=40, scale=15, size=clot_mask.sum())

    # Estrutura óssea periférica (proxy de costelas) para dar textura adicional.
    ring_outer = size * 0.48
    ring_inner = size * 0.44
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    bone_mask = (dist >= ring_inner) & (dist <= ring_outer)
    hu[bone_mask] = rng.normal(loc=700, scale=60, size=bone_mask.sum())

    return hu


def _write_dicom(path: Path, hu_array: np.ndarray, patient_id: str, study_uid: str, sop_uid: str, instance_number: int) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientID = patient_id
    ds.StudyInstanceUID = study_uid
    ds.SOPInstanceUID = sop_uid
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.Modality = "CT"
    ds.InstanceNumber = instance_number
    ds.PixelSpacing = [0.75, 0.75]
    ds.SliceThickness = 1.25
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = hu_array.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed, pois HU pode ser negativo
    pixel_array = np.clip(hu_array, -2000, 3000).astype(np.int16)
    ds.PixelData = pixel_array.tobytes()

    ds.save_as(str(path), enforce_file_format=True)


def generate_synthetic_dataset(
    out_dir: Path,
    n_patients: int = 24,
    slices_per_exam: int = 12,
    positive_ratio: float = 0.25,
    size: int = 96,
    seed: int = SEED,
) -> "list[dict]":
    """Gera um conjunto de exames sintéticos em `out_dir/<patient_id>/<study_uid>/*.dcm`.

    Retorna a lista de metadados (um registro por exame) com o rótulo binário
    `pe_present_on_exam`, pronta para virar um DataFrame de referência (manifest).
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    n_positive = int(round(n_patients * positive_ratio))
    labels = [1] * n_positive + [0] * (n_patients - n_positive)
    rng.shuffle(labels)

    for i in range(n_patients):
        patient_id = f"SYN{i:04d}"
        study_uid = generate_uid()
        label = labels[i]
        exam_dir = out_dir / patient_id / study_uid
        exam_dir.mkdir(parents=True, exist_ok=True)

        # Nem todo corte do exame positivo contém o achado — reflete a esparsidade real
        # do sinal na EP (poucos cortes por exame carregam o trombo).
        positive_slice_ratio = rng.uniform(0.15, 0.35) if label == 1 else 0.0

        for s in range(slices_per_exam):
            sop_uid = generate_uid()
            slice_is_positive = label == 1 and rng.uniform() < positive_slice_ratio
            hu = _make_synthetic_hu_slice(size=size, positive=slice_is_positive, rng=rng)
            path = exam_dir / f"{s:03d}.dcm"
            _write_dicom(path, hu, patient_id, study_uid, sop_uid, instance_number=s)

        try:
            exam_dir_str = str(exam_dir.resolve().relative_to(ROOT_DIR))
        except ValueError:
            exam_dir_str = str(exam_dir.resolve())

        manifest.append(
            {
                "patient_id": patient_id,
                "study_instance_uid": study_uid,
                "exam_dir": exam_dir_str,
                "pe_present_on_exam": label,
                "n_slices": slices_per_exam,
            }
        )

    return manifest


if __name__ == "__main__":
    import json

    from src.config import DATA_SAMPLE_DIR

    records = generate_synthetic_dataset(DATA_SAMPLE_DIR / "synthetic")
    manifest_path = DATA_SAMPLE_DIR / "synthetic_manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"Gerados {len(records)} exames sintéticos em {DATA_SAMPLE_DIR / 'synthetic'}")
    print(f"Manifesto: {manifest_path}")
