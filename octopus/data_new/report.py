import json
from typing import Any, Union

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

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=4)


# @define
# class DataHealthReport:
#     columns: dict[str, dict] = field(factory=dict)

#     def add_col(self, column_name: str, info: dict):
#         if column_name in self.columns:
#             self.columns[column_name].update(info)
#         else:
#             self.columns[column_name] = info

#     def add_multiple_col(self, columns_info: dict[str, dict]):
#         for column_name, info in columns_info.items():
#             self.add_col(column_name, info)

#     def get_col(self, column_name: str) -> dict:
#         return self.columns.get(column_name, {})

#     def to_json(self) -> str:
#         return json.dumps(asdict(self), indent=4)


# class Report:
#     data: dict = field(factory=dict)

#     def add_column_info(self, column_name, info):
#         if column_name not in self.data:
#             self.data[column_name] = {}
#         self.data[column_name].update(info)

#     def add_section(self, section_name, section_data):
#         self.data[section_name] = section_data

#     def get_column_info(self, column_name):
#         return self.data.get(column_name, {})

#     def __str__(self):
#         return str(self.data)
