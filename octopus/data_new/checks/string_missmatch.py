import pandas as pd
from fuzzywuzzy import fuzz


def check_string_mismatch(df: pd.DataFrame, threshold: int = 80) -> dict[list[list]]:
    """
    Find similar strings within each column of a DataFrame.

    Parameters:
    df: The input DataFrame.
    threshold : The similarity threshold (default is 80).

    Returns:
    A dictionary containing similar string pairs and their similarity scores.
    """

    string_mismatch = {}

    for column in df.columns:
        if df[column].dtype == object or df[column].dtype.name == "category":
            try:

                column_values = (
                    df[column].dropna().tolist()
                )  # Drop NaN values and convert to list

                # Initialize a list to keep track of processed strings
                processed = set()
                similar_groups = []

                for value in column_values:
                    if value not in processed:
                        # Find all similar strings to the current value
                        similar = [
                            other
                            for other in column_values
                            if fuzz.ratio(value, other) >= threshold
                        ]
                        if len(similar) > 1:
                            similar_groups.append(similar)
                            processed.update(similar)

                string_mismatch[column] = similar_groups
            except:
                string_mismatch[column] = None

    return string_mismatch
