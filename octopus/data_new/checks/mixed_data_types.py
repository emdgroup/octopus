import pandas as pd


def check_mixed_data_types(df: pd.DataFrame) -> dict[str, bool]:
    """Check for columns with mixed data types in a DataFrame.

    Args:
        df: The DataFrame to check for mixed data types.

    Returns:
        A dictionary with column names as keys and a
        boolean indicating if the column has mixed data types.
    """
    mixed_data_types = {}

    for column in df.columns:
        # Get unique data types in the column
        unique_types = set(df[column].map(type))

        # Check if there is more than one unique data type
        mixed_data_types[column] = len(unique_types) > 1

    return mixed_data_types
