"""Check missing values."""

import numpy as np
import pandas as pd

DEFAULT_NULL_VALUES = {"none", "null", "nan", "na", "", "\x00", "\x00\x00", "n/a"}


def check_missing_values(df: pd.DataFrame) -> dict[str, float]:
    """Check for missing values in a DataFrame."""
    # Create a case-insensitive version of DEFAULT_NULL_VALUES
    null_values_case_insensitive = {val.lower() for val in DEFAULT_NULL_VALUES}

    # Replace string null values with np.nan (case-insensitive)
    df = df.map(
        lambda x: (
            np.nan
            if isinstance(x, str) and x.strip().lower() in null_values_case_insensitive
            else x
        )
    )

    # Check for missing values
    missing_value_share_col = df.isnull().mean(axis=0)
    missing_value_share_row = df.isnull().mean(axis=1)

    # Create a dictionary for columns with missing values
    missing_value_dict_col = {
        col: share for col, share in missing_value_share_col.items() if share > 0
    }
    # Create a dictionary for rows with missing values
    missing_value_dict_row = {
        idx: share for idx, share in missing_value_share_row.items() if share > 0
    }

    return missing_value_dict_col, missing_value_dict_row
