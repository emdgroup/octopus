"""Compare Experperiments."""

from pathlib import Path

import pandas as pd

from octopus.experiment import OctoExperiment

# datset A
path_studyA = Path("./studies/20240214A_Martin_wf2_octofull_7x6_global_ardreg_single3/")
path_experimentA = path_studyA.joinpath("experiment3", "sequence0", "exp3_0.pkl")
# dataset B
path_studyB = Path("./studies/20240213A_Martin_wf2_octofull_7x6_global_ardreg_15/")
path_experimentB = path_studyB.joinpath("experiment3", "sequence0", "exp3_0.pkl")

# expA
experiment = OctoExperiment.from_pickle(path_experimentA)
expA = dict()
expA["x_traindev"] = experiment.data_traindev[experiment.feature_columns]
expA["y_traindev "] = (
    experiment.data_traindev[experiment.target_assignments.values()].to_numpy().ravel()
)
expA["x_test"] = experiment.data_test[experiment.feature_columns]
expA["y_test"] = experiment.data_test[experiment.target_assignments.values()]
x_test_A = pd.DataFrame(experiment.data_test[experiment.feature_columns])
x_test_A[experiment.row_column] = experiment.data_test[experiment.row_column]

# expB
experiment = OctoExperiment.from_pickle(path_experimentB)
expB = dict()
expB["x_traindev"] = experiment.data_traindev[experiment.feature_columns]
expB["y_traindev "] = (
    experiment.data_traindev[experiment.target_assignments.values()].to_numpy().ravel()
)
expB["x_test"] = experiment.data_test[experiment.feature_columns]
expB["y_test"] = experiment.data_test[experiment.target_assignments.values()]
x_test_B = pd.DataFrame(experiment.data_test[experiment.feature_columns])
x_test_B[experiment.row_column] = experiment.data_test[experiment.row_column]


print(x_test_A.shape)
print(x_test_B.shape)


row_id_A = x_test_A.row_id
row_id_B = x_test_B.row_id

row_id_A.equals(row_id_B)

x_test_A.equals(x_test_B)
