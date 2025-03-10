"""Helper functions."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from octopus.metrics import metrics_inventory

# def get_score(metric: str, y_true: np.array, y_pred: np.array) -> float:
#    """Calculate the specified metric for the given true and predicted values.
#
#    Args:
#        metric: The name of the metric to compute.
#            Valid options are 'MAE', 'R2', and 'MSE'.
#        y_true: An array of true values for the model's predictions
#            to be evaluated against.
#        y_pred: An array of predicted values to evaluate.
#
#    Returns:
#        The computed score of the specified metric.
#
#    Raises:
#        ValueError: If the metric provided is not recognized or supported.
#    """
#    if metric == "MAE":
#        return mean_absolute_error(y_true, y_pred)
#    elif metric == "R2":
#        return r2_score(y_true, y_pred)
#    elif metric == "MSE":
#        return mean_squared_error(y_true, y_pred)
#    else:
#        raise ValueError(
#            f"Unsupported metric '{metric}'. Supported metrics are 'MAE', 'R2', 'MSE'."
#        )


def get_performance(
    model, data, feature_columns, target_metric, target_assignments, threshold=0.5
) -> float:
    """Calculate model performance score on dataset for given metric."""
    input_data = data[feature_columns]

    # Ensure input_data is not empty and contains the required feature columns
    if input_data.empty or not all(
        col in input_data.columns for col in feature_columns
    ):
        raise ValueError(
            "Input data is empty or does not contain the required feature columns."
        )

    # Get target column
    target_col = list(target_assignments.values())[0]
    target = data[target_col]

    metric_config = metrics_inventory.get_metric_config(target_metric)
    metric_function = metrics_inventory.get_metric_function(target_metric)
    ml_type = metric_config.ml_type
    prediction_type = metric_config.prediction_type

    if ml_type == "timetoevent":
        estimate = model.predict(input_data)
        event_time = data[target_assignments["duration"]].astype(float)
        event_indicator = data[target_assignments["event"]].astype(bool)
        performance = metric_function(event_indicator, event_time, estimate)[0]

    if ml_type == "classification":
        if prediction_type == "predict_proba":
            probabilities = model.predict_proba(input_data)
            # Convert to NumPy array if it's a DataFrame
            if isinstance(probabilities, pd.DataFrame):
                probabilities = probabilities.to_numpy()  # Convert to NumPy array

            probabilities = probabilities[:, 1]  # Get probabilities for class 1
            performance = metric_function(target, probabilities)

        else:
            probabilities = model.predict_proba(input_data)
            if isinstance(probabilities, pd.DataFrame):
                probabilities = probabilities.to_numpy()  # Convert to NumPy array

            probabilities = probabilities[:, 1]  # Get probabilities for class 1
            predictions = (probabilities >= threshold).astype(int)
            performance = metric_function(target, predictions)

    if ml_type == "regression":
        if prediction_type == "predict_proba":
            raise ValueError("predict_proba not supported for regression.")

        else:
            predictions = model.predict(input_data)
            if isinstance(predictions, pd.DataFrame):
                predictions = (
                    predictions.to_numpy()
                )  # Convert to NumPy array if it's a DataFrame
            performance = metric_function(target, predictions)

    return performance


def get_performance_score(
    model, data, feature_columns, target_metric, target_assignments
) -> float:
    """Calculate performance value for given metric."""
    performance = get_performance(
        model, data, feature_columns, target_metric, target_assignments
    )

    # convert performance metric to score
    if metrics_inventory.get_direction(target_metric) == "maximize":
        return performance
    else:
        return -performance


def rdc(x, y, f=np.sin, k=20, s=1 / 6.0, n=5):
    """Randomized Dependence Coefficient.

    Computes the Randomized Dependence Coefficient
    x,y: numpy arrays 1-D or 2-D
        If 1-D, size (samples,)
        If 2-D, size (samples, variables)
    f:   function to use for random projection
    k:   number of random projections to use
    s:   scale parameter
    n:   number of times to compute the RDC and
        return the median (for stability)
    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.

    Implements the Randomized Dependence Coefficient
    David Lopez-Paz, Philipp Hennig, Bernhard Schoelkopf
    http://papers.nips.cc/paper/5138-the-randomized-dependence-coefficient.pdf
    """
    if n > 1:
        values = []
        for _ in range(n):
            try:
                values.append(rdc(x, y, f, k, s, 1))
            except np.linalg.linalg.LinAlgError:
                pass
        return np.median(values)

    if len(x.shape) == 1:
        x = x.reshape((-1, 1))
    if len(y.shape) == 1:
        y = y.reshape((-1, 1))

    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method="ordinal") for xc in x.T]) / float(x.size)
    cy = np.column_stack([rankdata(yc, method="ordinal") for yc in y.T]) / float(y.size)

    # Add a vector of ones so that w.x + b is just a dot product
    o = np.ones(cx.shape[0])
    x = np.column_stack([cx, o])
    y = np.column_stack([cy, o])

    # Random linear projections
    rx = (s / x.shape[1]) * np.random.randn(x.shape[1], k)
    ry = (s / y.shape[1]) * np.random.randn(y.shape[1], k)
    x = np.dot(x, rx)
    y = np.dot(y, ry)

    # Apply non-linear function to random projections
    fx = f(x)
    fy = f(y)

    # Compute full covariance matrix
    c = np.cov(np.hstack([fx, fy]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:
        # Compute canonical correlations
        cxx = c[:k, :k]
        cyy = c[k0 : k0 + k, k0 : k0 + k]
        cxy = c[:k, k0 : k0 + k]
        cyx = c[k0 : k0 + k, :k]

        eigs = np.linalg.eigvals(
            np.dot(np.dot(np.linalg.pinv(cxx), cxy), np.dot(np.linalg.pinv(cyy), cyx))
        )

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and 0 <= np.min(eigs) and np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub:
            break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return np.sqrt(np.max(eigs))


def rdc_correlation_matrix(df):
    """Calculate RDC correlation matrix."""
    features = df.columns
    n_features = len(features)
    rdc_matrix = np.zeros((n_features, n_features))

    # Calculate RDC for each pair of features
    for i in range(n_features):
        for j in range(i, n_features):
            if i == j:
                rdc_matrix[i, j] = 1.0
            else:
                rdc_value = rdc(df.iloc[:, i].values, df.iloc[:, j].values)
                rdc_matrix[i, j] = rdc_value
                rdc_matrix[j, i] = rdc_value

    return rdc_matrix
