"""Check single value."""

import pandas as pd


def check_single_value(data: pd.DataFrame) -> dict[str, bool]:
    """Check which columns in a DataFrame have a single unique value."""
    return {col: True for col in data.columns if data[col].nunique() == 1}
