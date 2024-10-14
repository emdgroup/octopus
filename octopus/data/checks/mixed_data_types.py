"""Check mixed data types."""

import pandas as pd


def check_mixed_data_types(df: pd.DataFrame) -> dict[str, bool]:
    """Check for columns with mixed data types."""
    mixed_data_types = {}

    for column in df.columns:
        # Get unique data types in the column
        unique_types = set(df[column].map(type))

        # Check if there is more than one unique data type
        if len(unique_types) > 1:
            mixed_data_types[column] = True

    return mixed_data_types
