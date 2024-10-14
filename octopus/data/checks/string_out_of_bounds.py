"""Check for long strings."""

import pandas as pd


def check_string_out_of_bounds(
    df: pd.DataFrame, length_threshold_factor=2
) -> dict[list]:
    """Find strings that are significantly longer than the average string length."""
    long_string = {}
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
                if long_strings:
                    long_string[column] = long_strings
            except:  # noqa: E722
                pass

    return long_string
