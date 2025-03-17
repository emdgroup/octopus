"""Impute data."""

from typing import Optional

import miceforest as mf
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class HalfMinImputer(BaseEstimator, TransformerMixin):
    """Impute missing values with half of the minimum value for each feature."""

    def __init__(self):
        self.half_min_ = None

    def fit(self, x: np.array, _y: Optional[np.ndarray] = None) -> "HalfMinImputer":
        """Fit the imputer on x.

        Parameters:
            x: The data to fit.
            _y: Not used, present for API consistency.

        Returns:
            self: Returns the instance itself.
        """
        x = pd.DataFrame(x)
        self.half_min_ = 0.5 * x.min(skipna=True)
        return self

    def transform(self, x: np.array) -> pd.DataFrame:
        """Transform x using the fitted imputer.

        Parameters:
            x: The data to transform.

        Returns:
            pd.DataFrame: The transformed data with missing values imputed.
        """
        x = pd.DataFrame(x)
        return x.fillna(self.half_min_)


def impute_simple(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list,
    imputation_method: str,
) -> tuple:
    """Impute missing values using a specified strategy.

    This function imputes missing values in the training and testing datasets
    based on the specified imputation method.

    Parameters:
        train_df: The training dataset containing features with potential
            missing values.
        test_df: The testing dataset containing features with potential
            missing values.
        feature_columns: A list of feature column names to impute.
        imputation_method: The imputation method to use ("median" or
            "halfmin").

    Returns:
        tuple: A tuple containing the imputed training and testing datasets as
            pandas DataFrames.

    Raises:
        ValueError: If an unknown imputation method is specified.
    """
    # Identify columns with missing values in the training dataset
    train_missing_columns = train_df[feature_columns].columns[
        train_df[feature_columns].isnull().any()
    ]

    # Identify columns with missing values in the test dataset
    test_missing_columns = test_df[feature_columns].columns[
        test_df[feature_columns].isnull().any()
    ]

    # Find common columns with missing values in both datasets
    common_missing_columns = train_missing_columns.union(test_missing_columns)

    if common_missing_columns.empty:
        # If there are no common missing columns, return the original dataframes
        return train_df.copy(), test_df.copy()

    if imputation_method == "median":
        imputer = SimpleImputer(strategy="median")
    elif imputation_method == "halfmin":
        imputer = HalfMinImputer()  # Assuming HalfMinImputer is defined elsewhere
    else:
        raise ValueError(f"Unknown imputation method: {imputation_method}")

    # Fit on training data and transform both train and test data for common
    # missing columns
    imputed_train_features = pd.DataFrame(
        imputer.fit_transform(train_df[common_missing_columns]),
        columns=common_missing_columns,
        index=train_df.index,
    )
    imputed_test_features = pd.DataFrame(
        imputer.transform(test_df[common_missing_columns]),
        columns=common_missing_columns,
        index=test_df.index,
    )

    # Replace original feature columns with imputed values only for the common
    # missing columns
    imputed_train_df = train_df.copy()
    imputed_train_df[common_missing_columns] = imputed_train_features

    imputed_test_df = test_df.copy()
    imputed_test_df[common_missing_columns] = imputed_test_features

    # Check for NaNs in the imputed test dataframe
    assert not imputed_train_df[feature_columns].isnull().any().any(), (
        "NaNs present in imputed train_df"
    )

    # Check for NaNs in the imputed test dataframe
    assert not imputed_test_df[feature_columns].isnull().any().any(), (
        "NaNs present in imputed test_df"
    )

    return imputed_train_df, imputed_test_df


def impute_mice(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute datasets using the mice-forest algorithm.

    Impute training and test datasets using the mice-forest algorithm,
    handling missing values across features.

    Parameters:
        train_df: Training dataset.
        test_df: Testing dataset.
        feature_columns: List of feature column names to impute.

    Returns:
        tuple: A tuple containing the imputed training and testing datasets.
    """
    # Identify variables with missing values in train and test datasets
    missing_in_train = train_df[feature_columns].isna().any()
    missing_in_test = test_df[feature_columns].isna().any()

    # Get lists of columns with missing values
    vars_missing_train = missing_in_train[missing_in_train].index.tolist()
    vars_missing_test = missing_in_test[missing_in_test].index.tolist()

    # Combine lists to get all columns with missing values in either dataset
    vars_with_missing = list(set(vars_missing_train + vars_missing_test))

    if not vars_with_missing:
        # No missing values in both datasets; return them unchanged
        return train_df, test_df

    # Need to impute variables in vars_with_missing
    num_iterations = 10  # Number of MICE iterations

    # Prepare the dataset for imputation
    train_data = train_df[feature_columns].copy()

    # Create the variable schema, excluding the variable itself from its predictors
    variable_schema = {
        var: [col for col in feature_columns if col != var] for var in vars_with_missing
    }

    # Initialize the imputation kernel
    kernel = mf.ImputationKernel(
        train_data,
        random_state=42,
        variable_schema=variable_schema,
    )

    # Run the MICE algorithm for the specified number of iterations
    kernel.mice(num_iterations)

    # Extract the imputed training data
    imputed_train_features = kernel.complete_data(dataset=0)

    # Replace the original feature columns with the imputed values
    imputed_train_df = train_df.copy()
    imputed_train_df[feature_columns] = imputed_train_features[feature_columns]

    # Check if the test dataset has missing values in the feature columns
    if vars_missing_test:
        # Impute the test data using the model fitted on training data
        test_data = test_df[feature_columns].copy()
        imputed_test = kernel.impute_new_data(
            test_data,
            datasets=[0],  # Use dataset index 0
        )
        imputed_test_features = imputed_test.complete_data(dataset=0)

        # Replace the original feature columns with the imputed values
        imputed_test_df = test_df.copy()
        imputed_test_df[feature_columns] = imputed_test_features[feature_columns]
    else:
        # No missing values in test dataset; use it as is
        imputed_test_df = test_df.copy()

    return imputed_train_df, imputed_test_df
