import os

import numpy as np
import pandas as pd

from octopus.data_new.health_checks import DataHealthChecker

df = pd.DataFrame(
    {
        "row_id": [0, 0, 1, 2],
        "constant_values": [1, 1, 1, 1],
        "none": [None, 2, None, 2],
        "nan": [np.NaN, 2, 3, 4],
        "nan_str": ["None", "noNe", "NaN", "nan"],
        "int": [1, 2, 3, 4],
        "float": [1.02, 2.4, 9.5, 0.6],
        "str_value": ["a", "b", "c", "d"],
        "mixed_data_types": ["1", 1, 2, 5],
        "string_missmatch": ["apple", "appel", "Apple", "grape"],
        "string_out_of_bounds": ["apple", "apple", "apple", "grapegrapegrapegrape"],
        "identical_feature_1": [99, 2, 3, 4],
        "identical_feature_2": [99, 2, 3, 4],
        "inf": [np.Inf, 2, 3, 4],
        "identical_feature_2": ["Inf", "inf", "infinity", 4],
        "Target": [100, 2, 3, 4],
    }
)

print(df)

data_health_check = DataHealthChecker(
    data=df,
    target_columns=["Target"],
    feature_columns=[
        "constant_values",
        "none",
        "nan",
        "int",
        "float",
        "str_value",
        "mixed_data_types",
        "string_missmatch",
        "string_out_of_bounds",
        "identical_feature_1",
        "identical_feature_2",
        "Target",
        "row_id",
    ],
    row_id="row_id",
)

report = data_health_check.generate_report(
    # column_dtypes=True,
    # idendical_feature_target=False,
    # mixed_data_types=False,
    # constant_value=False,
    # missing_values=False,
    # str_missmatch=False,
    # str_out_of_bounds=False,
    # feature_feature_correlation=False,
    # unique_column_names=True,
    # unique_row_id_values=True,
)
print(report.to_json())
