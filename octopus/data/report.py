import json
from typing import Any, Union

import pandas as pd
from attrs import asdict, define, field


@define
class DataHealthReport:
    columns: dict[str, dict] = field(factory=dict)
    rows: dict[str, Any] = field(factory=dict)
    outliers: dict[str, Any] = field(factory=dict)

    def add(self, category: str, key: str, value: Union[dict, Any, list]):
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
        for key, value in items.items():
            self.add(category, key, value)

    def get(self, category: str, key: str) -> Union[dict, Any, list]:
        if category == "columns":
            return self.columns.get(key, {})
        elif category == "rows":
            return self.rows.get(key)
        elif category == "outliers":
            return self.outliers.get(key, [])

    def create_issues(self) -> pd.DataFrame:
        data = []
        for column, attributes in self.columns.items():
            if "missing values share" in attributes:
                missing_share = attributes["missing values share"]
                if missing_share > 0.1:
                    data.append(
                        {
                            "Column": column,
                            "Type": "Error",
                            "Details": "High percentage of missing values.",
                            "Recommendation": f"""Missing {missing_share:.2%} of values. 
                                        Usually, imputation does not make sense with so many missing values. 
                                        Please consider deleting this column.""",
                        }
                    )
                elif missing_share > 0:
                    data.append(
                        {
                            "Column": column,
                            "Type": "Warning",
                            "Details": "Some data is missing.",
                            "Recommendation": f"""Missing {missing_share:.2%} of values.
                                    You can either delete this column or continue with imputation in the next step.""",
                        }
                    )
        df = pd.DataFrame(data, columns=["Column", "Type", "Details", "Recommendation"])
        return df

    def create_df(self):
        df_report = (
            pd.DataFrame(self.columns)
            .T.reset_index()
            .rename(columns={"index": "Column"})
            .sort_values("Column")
        )
        return df_report

    def generate_recommendations(self):
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
        return json.dumps(asdict(self), indent=4)

    @staticmethod
    def from_json(json_str: str) -> "DataHealthReport":
        data = json.loads(json_str)
        return DataHealthReport(
            columns=data.get("columns", {}),
            rows=data.get("rows", {}),
            outliers=data.get("outliers", {}),
        )
