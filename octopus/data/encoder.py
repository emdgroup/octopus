"""OctaData Encoder."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from attrs import define

from ..logger import get_logger

logger = get_logger()

MAX_CATEGORIES = 15


@define
class DataEncoder:
    """Encodes categorical features in the data."""

    data: pd.DataFrame
    feature_columns: List[str]
    target_columns: List[str]
    stratification_column: Optional[str]
    target_assignments: Dict[str, str]

    def encode(self):
        """Run all encoding steps."""
        self._categorical_feature_encoding()
        self._categorical_stratification_encoding()
        self._categorical_target_encoding()
        return (
            self.data,
            self.feature_columns,
            self.stratification_column,
            self.target_assignments,
            self.target_columns,
        )

    def _categorical_feature_encoding(self):
        """Process categorical feature columns."""
        # Identify categorical columns in self.relevant_columns
        categorical_columns = [
            col
            for col in self.feature_columns
            if self.data[col].dtype.name == "category"
        ]
        # Split into ordinal and non-ordinal (nominal) categorical columns
        ordinal_columns = [
            col for col in categorical_columns if self.data[col].cat.ordered
        ]
        nominal_columns = [
            col for col in categorical_columns if not self.data[col].cat.ordered
        ]
        # Process non-ordinal categorical columns
        if nominal_columns:
            # (1) Count unique values and check for columns with more than 15 categories
            columns_with_many_categories = [
                col for col in nominal_columns if self.data[col].nunique() > 15
            ]
            if columns_with_many_categories:
                raise ValueError(
                    f"The following nominal categorical columns have more"
                    f" than 15 unique categories: "
                    f"{', '.join(columns_with_many_categories)}"
                )

            # (2) Perform dummy encoding
            dummies = pd.get_dummies(
                self.data[nominal_columns],
                prefix=nominal_columns,
                drop_first=True,  # remove first to avoid redundant information
            )

            # Drop original nominal columns from feature_columns
            # we keep the nominal columns in the data
            self.feature_columns = [
                col for col in self.feature_columns if col not in nominal_columns
            ]

            # Add dummy columns to data
            self.data = pd.concat([self.data, dummies], axis=1)

            # Update feature_columns with new dummy column names
            self.feature_columns.extend(dummies.columns.tolist())

            logger.info("Encoded nominal columns.")

        # Process ordinal categorical columns
        if ordinal_columns:
            # Collect columns where categories are not integers
            problematic_columns = []
            for col in ordinal_columns:
                # Get the categories and check if they are all integers
                categories = self.data[col].cat.categories
                if not all(isinstance(cat, (int, np.integer)) for cat in categories):
                    problematic_columns.append(col)

            # Raise ValueError if there are problematic columns
            if problematic_columns:
                logger.error(
                    "Disallowed characters found in columns: %s", problematic_columns
                )
                raise ValueError(
                    f"The following ordinal categorical columns have"
                    f" non-integer categories: "
                    f"{', '.join(problematic_columns)}"
                )

    def _categorical_stratification_encoding(self):
        """Convert categorical stratification columns to int if needed."""
        if self.stratification_column:
            # Check if stratification column is of type 'category'
            column = self.stratification_column
            if self.data[column].dtype.name == "category":
                # Check if the categories are of integer type
                if not pd.api.types.is_integer_dtype(self.data[column].cat.categories):
                    # Create a new column with integer codes
                    new_column = f"{column}_int"
                    self.data[new_column] = self.data[column].cat.codes
                    # Update self.stratification_columns with the new column name
                    self.stratification_column = new_column

                    logger.info("Encoded stratification column.")

    def _categorical_target_encoding(self):
        """Convert categorical target columns to int if needed."""
        new_target_columns = []

        for column in self.target_columns:
            if self.data[column].dtype.name == "category":
                # Check if the categories are of integer type
                if not pd.api.types.is_integer_dtype(self.data[column].cat.categories):
                    # Create a new column with integer codes
                    new_column = f"{column}_int"
                    self.data[new_column] = self.data[column].cat.codes
                    # Save new columns
                    new_target_columns.append((column, new_column))

                    logger.info("Encoded target columns.")

        # Update target assignments
        for original_column, new_column in new_target_columns:
            for key in self.target_assignments:
                # Replace the original column with the new column
                if self.target_assignments[key] == original_column:
                    self.target_assignments[key] = (
                        new_column  # Directly replace the value
                    )

                    logger.info("Updated target assignments.")

        # update target_columns
        self.target_columns = list(self.target_assignments.values())
