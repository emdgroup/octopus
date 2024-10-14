"""Check for identical features."""

from typing import Dict, List

import pandas as pd


def check_identical_features(
    df: pd.DataFrame, feature_columns: List[str]
) -> Dict[str, List[str]]:
    """Identify features that have identical values but different column names."""
    identical_features = {col: [] for col in feature_columns}

    for col in feature_columns:
        for other_col in feature_columns:
            if col != other_col and df[col].equals(df[other_col]):
                identical_features[col].append(other_col)

    # Remove entries with empty lists
    identical_features = {k: v for k, v in identical_features.items() if v}

    return identical_features
