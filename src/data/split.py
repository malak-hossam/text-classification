from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.30,
    random_state: int = 42,
    stratify_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify_vals = df[stratify_col] if stratify_col in df.columns else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_vals,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

