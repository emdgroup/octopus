"""Check columns dypes."""

from typing import Dict, List

import pandas as pd


def check_columns_dtype(
    df: pd.DataFrame, columns: List[str], dtype_kinds: str
) -> Dict[str, bool]:
    """Check columns against specified dtype kinds."""
    results = {}
    for column in columns:
        if df[column].dtype.name == "object" and dtype_kinds == "object":
            results[column] = True
        elif df[column].dtype.kind not in dtype_kinds and dtype_kinds != "object":
            results[column] = False

    return results
