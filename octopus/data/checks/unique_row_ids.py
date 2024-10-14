import pandas as pd


def check_unique_row_id_values(df: pd.DataFrame, row_id_column: str) -> dict:
    """Check if all values in the specified row_id column are unique.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    row_id_column (str): The name of the column to check for unique values.

    Returns:
    dict: A dictionary with the column name as the key and a boolean value
    indicating if all values are unique.
    """
    unique_values = df[row_id_column].is_unique
    if not unique_values:
        return {row_id_column: unique_values}
