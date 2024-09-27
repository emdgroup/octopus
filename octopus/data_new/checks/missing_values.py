import re

import numpy as np
import pandas as pd

DEFAULT_NULL_VALUES = {"none", "null", "nan", "na", "", "\x00", "\x00\x00"}


def check_missing_values(df: pd.DataFrame) -> dict[str, float]:
    """Check for missing values in a DataFrame.
    Args:
        df: The DataFrame to check for missing values.

    Returns:
        A dictionary with column names as keys and the proportion
        of missing values as values.
    """
    # Create a case-insensitive version of DEFAULT_NULL_VALUES
    null_values_case_insensitive = {val.lower() for val in DEFAULT_NULL_VALUES}

    # Replace string null values with np.nan (case-insensitive)
    df = df.applymap(
        lambda x: (
            np.nan
            if isinstance(x, str) and x.strip().lower() in null_values_case_insensitive
            else x
        )
    )

    # Check for missing values
    return df.isnull().mean().to_dict()
