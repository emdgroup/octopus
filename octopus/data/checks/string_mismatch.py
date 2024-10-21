"""Check string mismatch."""

import pandas as pd
from fuzzywuzzy import fuzz


def check_string_mismatch(df: pd.DataFrame, threshold: int = 95) -> dict:
    """Find similar strings within each column of a DataFrame."""
    string_mismatch = {}

    for column in df.columns:
        if df[column].dtype == object or df[column].dtype.name == "category":
            try:
                column_values = df[column].dropna().tolist()

                # Check if the column has more than one unique value
                if len(set(column_values)) > 2:
                    # Initialize a list to keep track of processed strings
                    processed = set()
                    similar_groups = []

                    for value in column_values:
                        if value not in processed:
                            # Find all similar strings to the current value,
                            # excluding identical strings
                            similar = [
                                other
                                for other in column_values
                                if value != other
                                and fuzz.ratio(value, other) >= threshold
                            ]
                            if similar:
                                similar.append(value)
                                similar_groups.append(similar)
                                processed.update(similar)
                                processed.add(value)
                    if similar_groups:
                        string_mismatch[column] = similar_groups
            except:  # noqa: E722
                pass

    return string_mismatch
