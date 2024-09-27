import os

import numpy as np
import pandas as pd

from octopus.data_new.health_checks import DataHealthChecker

df = pd.DataFrame(
    {
        "constant_values": [1, 1, 1, 1],
        "none": [None, 2, None, 2],
        "nan": [np.NaN, 2, 3, 4],
        "nan_str": ["None", "noNe", "NaN", "nan"],
        "int": [1, 2, 3, 4],
        "float": [1.02, 2.4, 9.5, 0.6],
        "str_value": ["a", "b", "c", "d"],
        "mixed_data_types": ["1", 1, 2, 5],
        "string_missmatch": ["apple", "appel", "Apple", "grape"],
        "string_out_of_bounds": ["apple", "appel", "Apple", "grapegrapegrapegrape"],
        "conflicting_labels_1": [1, 3, 3, 7],
        "conflicting_labels_2": [1, 3, 3, 7],
        "Target": [100, 2, 3, 4],
    }
)

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
    ],
)

report = data_health_check.generate_report()
print(report)
