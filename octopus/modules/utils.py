"""Helper functions."""
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


def create_innerloop(
    df_train_dev: pd.DataFrame,
    feature_columns: list,
    target_columns: list,
    group_column: list,
) -> list:
    """Create split."""
    inner_loop_ind = []
    gkf = GroupKFold(n_splits=6)
    for cycle, (idx_train, idx_dev) in enumerate(
        gkf.split(
            X=df_train_dev[feature_columns],
            y=df_train_dev[target_columns],
            groups=df_train_dev[group_column].values.ravel(),
        )
    ):
        inner_loop_ind.append(
            {
                "cycle": cycle,
                "train": idx_train.tolist(),
                "dev": idx_dev.tolist(),
            }
        )
    return inner_loop_ind


def model_linear_regression(prediction_task: str, config: dict) -> linear_model:
    """Create linear model."""
    if prediction_task == "classification":
        return linear_model.RidgeClassifier(**config)
    elif prediction_task == "regression":
        return linear_model.Ridge(**config)
    else:
        raise ValueError(
            f"The type {prediction_task} is not supported for Linear Regression"
        )


def get_score(metric: str, y_true: np.array, y_pred: np.array) -> float:
    """Calculate selected metric."""
    if metric == "MAE":
        return mean_absolute_error(y_true, y_pred)
    elif metric == "R2":
        return r2_score(y_true, y_pred)
    elif metric == "MSE":
        return mean_squared_error(y_true, y_pred)


def optuna_direction(metric: str) -> str:
    """Calculate selected metric."""
    if metric == "MAE":
        return "minimize"
    elif metric == "R2":
        return "maximize"
    elif metric == "MSE":
        return "minimize"
