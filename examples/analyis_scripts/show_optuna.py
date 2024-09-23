# %%
"""Script to show optuna results."""

from pathlib import Path

import optuna
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# from optuna.integration.shap import ShapleyImportanceEvaluator

# setup
study_name = "MBOS6_mb_OMORO_fixed_xgb_mb"
experiment_id = 0
sequence_id = 0


study_path = Path("./studies/" + study_name)
# db_files = list(study_path.rglob('*.db'))
db_file = list(study_path.rglob("**/experiment0/sequence0/optuna/*.db"))[0]
print("Database file: ", db_file)

# open optuna study
study = optuna.create_study(
    study_name=db_file.stem, storage="sqlite:///" + str(db_file), load_if_exists=True
)
# df = study.trials_dataframe(attrs=("number", "value","user_attrs",  "params", "state"))
df = study.trials_dataframe()

print(df.shape)
# df.head(20)

# %%
if len(df["value"].unique()) != 1:
    ## list failed experiments
    print(df[df["state"] == "FAIL"])

    # number of trials
    print("Number of completed trials: ", len(df))

    # show best trial
    print(df.loc[df["value"].idxmin()])

    # show 10 best trials
    print(df.sort_values(by="value", ascending=True).head(10))

    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()

    fig = optuna.visualization.plot_slice(study)
    fig.show()

    # evaluator = ShapleyImportanceEvaluator(seed=0)
    # param_importance_without_inf = evaluator.evaluate(study)
    # data = param_importance_without_inf
    # sorted_d = sorted(data.items(), key=lambda x: x[1])
    # display(list(reversed(sorted_d)))


# %%
