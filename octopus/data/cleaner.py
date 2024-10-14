import attrs
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer


@attrs.define
class DataHealthReport:
    columns: dict
    rows: dict = attrs.field(factory=dict)
    outliers: dict = attrs.field(factory=dict)


@attrs.define
class DataCleaner:
    df: pd.DataFrame
    report: DataHealthReport
    feature_columns: list = attrs.field(factory=list)
    cleaning_report: dict = attrs.field(factory=dict)

    def update_feature_columns(self):
        """
        Update the feature columns based on the current DataFrame columns.
        """
        self.feature_columns = [
            col for col in self.df.columns if col in self.feature_columns
        ]

    def remove_single_value_columns(self):
        """
        Remove columns that contain only a single unique value based on the report.

        Returns:
        pd.DataFrame: The cleaned DataFrame.
        """
        removed_columns = []
        for column, details in self.report.columns.items():
            if column in self.df.columns and details.get("single_values", False):
                self.df.drop(column, axis=1, inplace=True)
                removed_columns.append(column)

        if removed_columns:
            self.cleaning_report["removed_single_value_columns"] = removed_columns
        return self.df

    def remove_high_missing_data_columns(self, threshold=0.1):
        """
        Remove columns with a high percentage of missing data based on the report.

        Parameters:
        threshold (float): The threshold for missing data. Columns with a
                           percentage of missing data higher than this will be removed.
                           Default is 0.5 (50%).

        Returns:
        pd.DataFrame: The cleaned DataFrame.
        """
        removed_columns = []
        for column, details in self.report.columns.items():
            if (
                column in self.df.columns
                and details.get("missing values share", 0) > threshold
            ):
                self.df.drop(column, axis=1, inplace=True)
                removed_columns.append(column)

        if removed_columns:
            self.cleaning_report["removed_high_missing_data_columns"] = removed_columns
        return self.df

    def remove_high_infinity_data_columns(self, threshold=0.1):
        """
        Remove columns with high infinity values

        Parameters:
        threshold (float): The threshold for missing data. Columns with a
                           percentage of missing data higher than this will be removed.
                           Default is 0.5 (50%).

        Returns:
        pd.DataFrame: The cleaned DataFrame.
        """
        removed_columns = []
        for column, details in self.report.columns.items():
            if (
                column in self.df.columns
                and details.get("infinity values share", 0) > threshold
            ):
                self.df.drop(column, axis=1, inplace=True)
                removed_columns.append(column)

        if removed_columns:
            self.cleaning_report["removed_high_inifity_data_columns"] = removed_columns
        return self.df

    def remove_identical_features(self):
        """
        Remove identical features with different labels. Keep only one column.

        Returns:
        pd.DataFrame: The cleaned DataFrame.
        """
        removed_columns = []
        for column, details in self.report.columns.items():
            if "identical_features" in details:
                for item in details["identical_features"]:
                    if column not in removed_columns and column in self.df.columns:
                        self.df.drop(column, axis=1, inplace=True)
                        removed_columns.append(item)

        if removed_columns:
            self.cleaning_report["removed_identical_columns"] = removed_columns
        return self.df

    def one_hot_encode_categorical_features(self):
        """
        One-hot encode categorical features based on the report.

        Returns:
        pd.DataFrame: The DataFrame with categorical features one-hot encoded.
        """

        categorical_columns = []
        for column, details in self.report.columns.items():
            if (
                column in self.df.columns
                and details.get("object/categorical dtype", False)
                and len(details) == 1
            ):
                categorical_columns.append(column)

        self.df = pd.get_dummies(self.df, columns=categorical_columns, drop_first=True)

        if categorical_columns:
            self.cleaning_report["one_hot_encoded_columns"] = categorical_columns
        return self.df

    def impute_missing_data(
        self, imputer_type: str = "simple", threshold: float = 0.1, **imputer_params
    ):
        """
        Impute missing data in columns with low missing values using the specified imputer.

        Parameters:
        imputer_type (str): The type of imputer to use ('simple' or 'knn'). Default is 'simple'.
        **imputer_params: Additional parameters to pass to the imputer.

        Returns:
        pd.DataFrame: The DataFrame with missing data imputed.
        """
        # Filter the feature columns to include only those present in the DataFrame
        feature_columns = [
            col for col in self.feature_columns if col in self.df.columns
        ]

        # Filter the feature columns to include only those present in the DataFrame
        feature_columns = [
            col for col in self.feature_columns if col in self.df.columns
        ]

        # Filter columns based on missing values share threshold
        columns_to_impute = [
            col
            for col in feature_columns
            if "missing values share" in self.report.columns.get(col, {})
            and self.report.columns[col]["missing values share"] < threshold
        ]

        numeric_columns = (
            self.df[columns_to_impute].select_dtypes(include=["number"]).columns
        )
        non_numeric_columns = (
            self.df[columns_to_impute].select_dtypes(exclude=["number"]).columns
        )

        # Ensure numeric columns contain only numeric data
        for col in numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        if list(numeric_columns):
            if imputer_type == "knn":
                numeric_imputer = KNNImputer(**imputer_params)
                self.df[numeric_columns] = numeric_imputer.fit_transform(
                    self.df[numeric_columns]
                )
            else:
                numeric_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
                self.df[numeric_columns] = numeric_imputer.fit_transform(
                    self.df[numeric_columns]
                )

        if list(non_numeric_columns):
            non_numeric_imputer = SimpleImputer(strategy="most_frequent")
            self.df[non_numeric_columns] = non_numeric_imputer.fit_transform(
                self.df[non_numeric_columns]
            )

        if columns_to_impute:
            self.cleaning_report["imputed_columns"] = columns_to_impute
        return self.df

    def clean_data(
        self,
        remove_single_value=True,
        remove_high_missing=True,
        remove_high_infinity=True,
        missing_data_threshold=0.1,
        impute_data=True,
        imputer_type="simple",
        one_hot_encode=True,
        remove_identical_features=True,
    ):
        """
        Clean the DataFrame by removing columns with a single unique value, columns
        with a high percentage of missing data, and/or one-hot encoding categorical features
        based on user selection and the report.

        Parameters:
        remove_single_value (bool): Whether to remove columns with a single unique value.
                                    Default is True.
        remove_high_missing (bool): Whether to remove columns with a high percentage of missing data.
                                    Default is True.
        missing_data_threshold (float): The threshold for missing data. Columns with a
                                        percentage of missing data higher than this will be removed.
                                        Default is 0.5 (50%).
        one_hot_encode (bool): Whether to one-hot encode categorical features.
                               Default is True.

        Returns:
        pd.DataFrame: The cleaned DataFrame.
        """

        self.cleaning_report = {}  # Reset the report for each cleaning operation
        if remove_single_value:
            self.remove_single_value_columns()
        if remove_high_missing:
            self.remove_high_missing_data_columns(threshold=missing_data_threshold)
        if remove_high_infinity:
            self.remove_high_infinity_data_columns(threshold=missing_data_threshold)
        if remove_identical_features:
            self.remove_identical_features()
        if one_hot_encode:
            self.one_hot_encode_categorical_features()
        if impute_data:
            self.impute_missing_data()

        self.update_feature_columns()

        return self.df, self.feature_columns

    def get_cleaning_report(self):
        """
        Get a report of the cleaning operations performed.

        Returns:
        dict: A dictionary containing details of the cleaning operations performed.
        """
        return self.cleaning_report
