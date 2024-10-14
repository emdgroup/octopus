"""Check for inifity values."""

import numpy as np
import pandas as pd

DEFAULT_INFINITY_VALUES = {"inf", "infinity", "∞"}


def check_infinity_values(
    df: pd.DataFrame, infinity_values: set[str] = None
) -> dict[str, float]:
    """Check for infinity values in a DataFrame."""
    if infinity_values is None:
        infinity_values = DEFAULT_INFINITY_VALUES

    # Create a case-insensitive version of infinity_values
    infinity_values_case_insensitive = {val.lower() for val in infinity_values}

    # Replace string infinity values with np.inf or -np.inf (case-insensitive)
    def replace_infinity(x):
        if isinstance(x, str):
            if x.strip().lower() in infinity_values_case_insensitive:
                return np.inf
            elif x.strip().lower() in {
                f"-{val}" for val in infinity_values_case_insensitive
            }:
                return -np.inf
        return x

    df = df.map(replace_infinity)

    # Ensure all data is numeric before checking for infinity
    df = df.apply(pd.to_numeric, errors="coerce")

    # Check for positive and negative infinity values
    infinity_mask = df.map(lambda x: np.isinf(x))

    # Calculate the proportion of infinity values in each column
    infinity_value_share = infinity_mask.mean()

    # Create a dictionary for columns with infinity values
    infinity_value_dict = {
        col: share for col, share in infinity_value_share.items() if share > 0
    }

    return infinity_value_dict
