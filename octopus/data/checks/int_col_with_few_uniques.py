"""Check col with few uniques items."""

import pandas as pd


def check_int_col_with_few_uniques(
    df: pd.DataFrame, columns: list, threshold: int = 5
) -> dict:
    """Check for integer columns with a small number of unique elements."""
    return {
        col: df[col].nunique()
        for col in df[columns].columns
        if pd.api.types.is_integer_dtype(df[col]) and 2 < df[col].nunique() <= threshold
    }
