"""Check for duplicated rows."""

import pandas as pd


def check_duplicated_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Check all duplicated rows."""
    duplicated_mask = data.duplicated()
    duplicated_rows = data[duplicated_mask]
    if duplicated_rows.empty:
        return None
    return duplicated_rows
