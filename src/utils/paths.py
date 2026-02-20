from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"
HF_RUNS_DIR = OUTPUTS_DIR / "hf_runs"
CONFIG_DIR = PROJECT_ROOT / "src" / "config"


def ensure_project_dirs() -> None:
    for path in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUTS_DIR,
        EMBEDDINGS_DIR,
        MODELS_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        HF_RUNS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

