"""Data Health Report."""

import json
from typing import Any, Union

import pandas as pd
from attrs import asdict, define, field


@define
class DataHealthReport:
    """Data Health Report."""

    columns: dict[str, dict] = field(factory=dict)
    rows: dict[str, Any] = field(factory=dict)
    outliers: dict[str, Any] = field(factory=dict)

    def add(self, category: str, key: str, value: Union[dict, Any, list]):
        """Add single item to report."""
        if category == "columns":
            if key in self.columns:
                self.columns[key].update(value)
            else:
                self.columns[key] = value
        elif category == "rows":
            self.rows[key] = value
        elif category == "outliers":
            self.outliers[key] = value

    def add_multiple(self, category: str, items: dict[str, Any]):
        """Add multiple items to report."""
        for key, value in items.items():
            self.add(category, key, value)

    def create_df(self) -> pd.DataFrame:
        """Create report as a DataFrame with columns: type, id, parameter, value."""
        # List to collect DataFrames
        dfs = []

        # Define the attributes to process and their corresponding types
        attributes = {"columns": "column", "rows": "row", "outliers": "outlier"}

        # Process each attribute in self
        for attr_name, type_name in attributes.items():
            data_dict = getattr(self, attr_name)
            if data_dict:
                records = []
                for id_, param_dict in data_dict.items():
                    # Ensure param_dict is a dictionary
                    if not isinstance(param_dict, dict):
                        param_data = {"value": param_dict}
                    else:
                        param_data = param_dict
                    for param_name, value in param_data.items():
                        records.append(
                            {
                                "type": type_name,
                                "id": id_,
                                "parameter": param_name,
                                "value": value,
                            }
                        )
                df = pd.DataFrame(records)
                dfs.append(df)

        # Combine all DataFrames
        if dfs:
            df_combined = pd.concat(dfs, ignore_index=True)
        else:
            # If all data_dicts are empty, return an empty DataFrame
            # with the desired columns
            df_combined = pd.DataFrame(columns=["type", "id", "parameter", "value"])

        return df_combined

    def generate_recommendations(self):
        """Generate recommendation for better data quality."""
        recommendations = {
            "Remove columns with high missing value": any(
                value.get("missing values share", 0) >= 0.1
                for value in self.columns.values()
            ),
            "Impute columns with NaN values": any(
                0 < value.get("missing values share", 0) < 0.1
                for value in self.columns.values()
            ),
            "Remove columns with single values": any(
                value.get("single_values", False) for value in self.columns.values()
            ),
            "Remove columns with infinity values": any(
                value.get("infinity values share", 0) > 0
                for value in self.columns.values()
            ),
            "Remove duplicate columns, keeping only one with identical features.": any(
                value.get("identical_features", False)
                for value in self.columns.values()
            ),
            "One-hot encode object and categorical columns": any(
                value.get("object/categorical dtype", False)
                for value in self.columns.values()
            ),
        }
        return recommendations

    def to_json(self) -> str:
        """Save as json file."""
        return json.dumps(asdict(self), indent=4)

    @staticmethod
    def from_json(json_str: str) -> "DataHealthReport":
        """Load from json."""
        data = json.loads(json_str)
        return DataHealthReport(
            columns=data.get("columns", {}),
            rows=data.get("rows", {}),
            outliers=data.get("outliers", {}),
        )
