#!/usr/bin/env python3
"""Gera um conjunto de exames DICOM sintéticos para validar o pipeline (ver src/data/synthetic.py).

Uso:
    python scripts/generate_synthetic_sample.py

NÃO é um substituto para os dados reais do RSNA STR PE Dataset — ver README.md
para instruções de download da amostra real via Kaggle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_SAMPLE_DIR
from src.data.synthetic import generate_synthetic_dataset

if __name__ == "__main__":
    out_dir = DATA_SAMPLE_DIR / "synthetic"
    records = generate_synthetic_dataset(out_dir)

    manifest_path = DATA_SAMPLE_DIR / "synthetic_manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    n_pos = sum(r["pe_present_on_exam"] for r in records)
    print(f"Gerados {len(records)} exames sintéticos ({n_pos} positivos, {len(records) - n_pos} negativos) em {out_dir}")
    print(f"Manifesto salvo em {manifest_path}")
