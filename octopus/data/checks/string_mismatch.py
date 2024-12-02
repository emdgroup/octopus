"""Check string mismatch."""

import re

import pandas as pd
from rapidfuzz import fuzz


def check_string_mismatch(df: pd.DataFrame) -> dict:
    """Find unique groups of similar strings, ignoring numeric suffixes."""
    string_mismatch = {}

    def remove_numbers(entry):
        """Remove numbers from the end of a string."""
        return re.sub(r"\d+$", "", entry)

    def determine_threshold(length):
        """Determine the similarity threshold based on the length of the string."""
        if length <= 7:
            return 80  # Lower threshold for shorter strings
        elif 7 <= length <= 12:
            return 85  # Medium threshold for medium-length strings
        else:
            return 90  # Higher threshold for longer strings

    for column in df.columns:
        if df[column].dtype == object or df[column].dtype.name == "category":
            try:
                # Remove numbers from the end of each entry
                column_values = df[column].dropna().apply(remove_numbers).unique()

                # Check if the column has more than one unique value
                if len(column_values) > 2:
                    # Initialize a set to keep track of processed strings
                    processed = set()
                    similar_groups = []

                    for value in column_values:
                        if value not in processed:
                            threshold = determine_threshold(len(value))
                            # Find all similar strings to the current value,
                            # excluding identical strings
                            similar = set(
                                other
                                for other in column_values
                                if value != other
                                and fuzz.ratio(value, other) >= threshold
                            )
                            if similar:
                                similar.add(value)
                                similar_groups.append(list(similar))
                                processed.update(similar)
                    if similar_groups:
                        string_mismatch[column] = similar_groups
            except Exception as e:  # Catch and log exception if needed
                print(f"An error occurred while processing column {column}: {e}")

    return string_mismatch
