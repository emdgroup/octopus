"""Check col with few uniques items."""

import pandas as pd


def check_int_col_with_few_uniques(df: pd.DataFrame, threshold: int = 20) -> dict:
    """Check for integer columns with a small number of unique elements."""
    return {
        col: df[col].nunique()
        for col in df.columns
        if pd.api.types.is_integer_dtype(df[col]) and 1 < df[col].nunique() <= threshold
    }
