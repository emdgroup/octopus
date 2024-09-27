import pandas as pd


def check_string_out_of_bounds(
    df: pd.DataFrame, length_threshold_factor=2
) -> dict[list]:
    """
    Find strings that are significantly longer than the average string length.

    Parameters:
    df: The input DataFrame.
    length_threshold_factor: The factor by which a string's
    length must exceed the average length to be considered long (default is 2).

    Returns:
    dict: A dictionary where each key is a column name and the value is
    a list of long strings.
    """

    long_string = {}
    df
    for column in df.columns:
        if df[column].dtype == object or df[column].dtype.name == "category":
            try:
                column_values = (
                    df[column].dropna().tolist()
                )  # Drop NaN values and convert to list

                # Calculate the average length of strings in the column
                avg_length = sum(len(value) for value in column_values) / len(
                    column_values
                )

                # Identify strings that are significantly longer than the average length
                long_strings = [
                    value
                    for value in column_values
                    if len(value) > length_threshold_factor * avg_length
                ]

                long_string[column] = long_strings
            except:
                long_string[column] = None

    return long_string
