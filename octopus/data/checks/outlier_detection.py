"""Outlier dection."""

import pandas as pd
from PyNomaly import loop

NEAREST_NEIGHBORS_PERCENT = 0.01
MINIMUM_NUM_NEAREST_NEIGHBORS = 5
EXTEND = 3


def check_outlier_detection(df: pd.DataFrame):
    """Check for outliers."""
    df_temp = df.copy().dropna().select_dtypes(include=["number"])

    num_neighbors = int(
        max(NEAREST_NEIGHBORS_PERCENT * df_temp.shape[0], MINIMUM_NUM_NEAREST_NEIGHBORS)
    )
    m = (
        loop.LocalOutlierProbability(
            df_temp,
            extent=EXTEND,
            n_neighbors=num_neighbors,
        )
        .fit()
        .local_outlier_probabilities
    )
    df_temp["outlier_scores"] = m
    df = df.merge(df_temp, how="left")
    df = df[df["outlier_scores"] > 0.7]
    return df
