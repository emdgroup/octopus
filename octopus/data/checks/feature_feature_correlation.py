"""Check feature feature correlation."""

import pandas as pd


def check_feature_feature_correlation(
    df: pd.DataFrame,
    feature_columns: list,
    method: str = "pearson",
    threshold=0.8,
):
    """Find columns in the DataFrame that are highly correlated."""
    # Filter only numeric columns
    corr_matrix = (
        df[feature_columns].select_dtypes(include=[float, int]).corr(method=method)
    )

    # Dictionary to store the columns with high correlation
    highly_correlated = {col: {} for col in corr_matrix.columns}

    # Iterate over the correlation matrix and find highly correlated columns
    for col in corr_matrix.columns:
        for row in corr_matrix.index:
            if col != row and abs(corr_matrix.loc[row, col]) > threshold:
                highly_correlated[col][row] = corr_matrix.loc[row, col]

    # Remove keys with empty dictionaries
    highly_correlated = {k: v for k, v in highly_correlated.items() if v}

    return highly_correlated
