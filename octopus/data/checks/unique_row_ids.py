"""Check unique row ids."""

import pandas as pd


def check_unique_row_id_values(df: pd.DataFrame, row_id_column: str) -> dict:
    """Check if all values in the specified row_id column are unique."""
    unique_values = df[row_id_column].is_unique
    if not unique_values:
        return {row_id_column: unique_values}
