#!/usr/bin/env python3
"""Extrai características (todas as famílias) para cada exame do manifesto e salva
a matriz de características agregada por exame em outputs/tables/features.parquet.

Uso:
    python scripts/extract_features.py --manifest data/sample/synthetic_manifest.json \\
        --data-root data/sample/synthetic --out outputs/tables/features.parquet

Para rodar sobre os dados reais, gere um manifesto equivalente (patient_id,
study_instance_uid, exam_dir, pe_present_on_exam) a partir de train.csv do Kaggle
(ver README.md) e aponte --manifest/--data-root para ele.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from tqdm import tqdm

from src.config import DATA_SAMPLE_DIR, ROOT_DIR, TABLES_DIR
from src.data.dicom_io import load_exam_slices
from src.features.aggregate import extract_slice_features, pool_exam_features
from src.preprocessing.pipeline import preprocess_slice

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_exam_features(exam_dir: Path, families: list[str] | None = None) -> dict:
    slices = load_exam_slices(exam_dir)
    slice_features = []
    for dicom_slice in slices:
        image = preprocess_slice(dicom_slice)
        slice_features.append(extract_slice_features(image, families=families))
    return pool_exam_features(slice_features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=str, default=str(DATA_SAMPLE_DIR / "synthetic_manifest.json"))
    parser.add_argument("--data-root", type=str, default=str(DATA_SAMPLE_DIR / "synthetic"))
    parser.add_argument("--out", type=str, default=str(TABLES_DIR / "features.parquet"))
    parser.add_argument("--families", type=str, default=None, help="Lista separada por vírgula (ex.: texture,shape,gradient,intensity)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    records = json.loads(manifest_path.read_text())
    families = args.families.split(",") if args.families else None

    rows = []
    for record in tqdm(records, desc="Extraindo características por exame"):
        raw_exam_dir = Path(record["exam_dir"])
        exam_dir = raw_exam_dir if raw_exam_dir.is_absolute() else ROOT_DIR / raw_exam_dir
        features = extract_exam_features(exam_dir, families=families)
        features["patient_id"] = record["patient_id"]
        features["study_instance_uid"] = record["study_instance_uid"]
        features["pe_present_on_exam"] = record["pe_present_on_exam"]
        rows.append(features)

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Matriz de características salva em %s (shape=%s)", out_path, df.shape)


if __name__ == "__main__":
    main()
