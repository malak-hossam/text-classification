from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

EXPECTED_COLUMNS = {"content", "label"}
DEFAULT_DATASET_NAME = "arabic_sentiment_reviews.csv"


def _validate_columns(columns: Iterable[str]) -> None:
    missing = EXPECTED_COLUMNS - set(columns)
    if missing:
        raise ValueError(
            f"Dataset must include columns {sorted(EXPECTED_COLUMNS)}, missing: {sorted(missing)}"
        )


def load_raw_dataset(csv_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else RAW_DATA_DIR / DEFAULT_DATASET_NAME
    if not path.exists():
        raise FileNotFoundError(
            "Dataset file not found.\n"
            f"Expected path: {path}\n"
            "Download it from Kaggle and place it locally at data/raw/arabic_sentiment_reviews.csv"
        )

    df = pd.read_csv(path)
    _validate_columns(df.columns)
    df = df.dropna(subset=["content", "label"]).drop_duplicates(subset=["content"]).reset_index(drop=True)
    return df


def save_processed_dataset(df: pd.DataFrame, filename: str = "arabic_sentiment_reviews_processed.csv") -> Path:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / filename
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path

