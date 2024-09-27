import pandas as pd


def check_column_dtypes(df: pd.DataFrame) -> dict:
    """
    Get the data types of each column in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    dict: A dictionary where each key is a column name and the value is
    the data type of the column.
    """
    return df.dtypes.apply(lambda x: x.name).to_dict()
