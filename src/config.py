"""Configuração central do projeto TP1 (Grupo 2 — RSNA 2020 Pulmonary Embolism Detection)."""

from __future__ import annotations

from pathlib import Path

SEED = 42

# Caminhos relativos à raiz do repositório (nunca usar caminhos absolutos, cf. §4.5 do enunciado).
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_SAMPLE_DIR = ROOT_DIR / "data" / "sample"
OUTPUTS_DIR = ROOT_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"

# Janelamento (window center/width, em HU) usado para CT de tórax / angio-TC pulmonar.
# Mediastinal/PE window: realça vasos e trombos contra o parênquima pulmonar.
WINDOWS = {
    "mediastinal_pe": {"center": 100, "width": 700},   # janela principal para embolia pulmonar
    "lung": {"center": -600, "width": 1500},            # janela pulmonar, útil para achados parenquimatosos
    "soft_tissue": {"center": 40, "width": 400},
}
DEFAULT_WINDOW = "mediastinal_pe"

# Tamanho alvo (pixels) após recorte/reamostragem isotrópica, antes da extração de descritores.
TARGET_SIZE = 224

# Rótulo alvo do baseline: classificação binária no nível do EXAME
# ("negative_exam_for_pe" invertido -> 1 = exame positivo para EP, 0 = negativo).
# Ver Metodologia do artigo para a justificativa da unidade de análise (exame vs. corte).
TARGET_COLUMN = "pe_present_on_exam"
GROUP_COLUMN = "patient_id"
EXAM_COLUMN = "study_instance_uid"
SLICE_COLUMN = "sop_instance_uid"

N_SPLITS = 5
