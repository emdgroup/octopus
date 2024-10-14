"""Check for object dtype."""

import pandas as pd


def check_column_dtypes(df: pd.DataFrame) -> dict:
    """Check for object dtype."""
    return {
        col: True
        for col in df.columns
        if df[col].dtype == "object" or df[col].dtype.name == "category"
    }
