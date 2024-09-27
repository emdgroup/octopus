import pandas as pd


def check_conflicting_labels(df):
    """
    Find columns in the DataFrame that have identical values to other columns.

    Parameters:
    df: The input DataFrame.

    Returns:
    dict: A dictionary where each key is a column name and the value
    is a list of column names with identical values.
    """
    from itertools import combinations

    # Dictionary to store the columns with identical values
    identical_columns = {col: [] for col in df.columns}

    # Get all combinations of two columns
    column_combinations = list(combinations(df.columns, 2))

    for col1, col2 in column_combinations:
        # Check if the values in the two columns are identical
        if df[col1].equals(df[col2]):
            identical_columns[col1].append(col2)
            identical_columns[col2].append(col1)

    # Remove keys with empty lists
    identical_columns = {k: v for k, v in identical_columns.items() if v}

    return identical_columns
