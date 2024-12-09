"""Helper functions."""

import numpy as np
from scipy.stats import rankdata

# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    if target_metric in ["AUCROC", "LOGLOSS", "AUCPR", "NEGBRIERSCORE"]:
        target_col = list(target_assignments.values())[0]
        target = data[target_col]
        probabilities = model.predict_proba(data[feature_columns])[
            :, 1
        ]  # binary only!!
        performance = metrics_inventory[target_metric]["method"](target, probabilities)
    elif target_metric in ["ACC", "ACCBAL", "F1"]:
        # only soft voting implemented (averaging of prababilites)
        target_col = list(target_assignments.values())[0]
        target = data[target_col]
        probabilities = model.predict_proba(data[feature_columns])[:, 1]  # binary
        predictions = (probabilities >= threshold).astype(int)
        performance = metrics_inventory[target_metric]["method"](target, predictions)
    elif target_metric in ["CI"]:
        estimate = model.predict(data[feature_columns])
        event_time = data[target_assignments["duration"]].astype(float)
        event_indicator = data[target_assignments["event"]].astype(bool)
        performance, _, _, _, _ = metrics_inventory[target_metric]["method"](
            event_indicator, event_time, estimate
        )
    elif target_metric in ["R2", "MSE", "MAE"]:
        target_col = list(target_assignments.values())[0]
        target = data[target_col]
        predictions = model.predict(data[feature_columns])
        performance = metrics_inventory[target_metric]["method"](target, predictions)
    else:
        raise ValueError(f"Unsupported target metric: {target_metric}")

    return performance


def get_performance_score(
    model, data, feature_columns, target_metric, target_assignments
) -> float:
    """Calculate performance value for given metric."""
    performance = get_performance(
        model, data, feature_columns, target_metric, target_assignments
    )

    # convert performance metric to score
    if optuna_direction(target_metric) == "maximize":
        return performance
    else:
        return -performance


def optuna_direction(metric: str) -> str:
    """Determine the direction (minimize or maximize) for the given metric.

    Args:
        metric: The name of the performance metric, which should be one of
                        'MAE', 'MSE', 'R2', 'ACC', 'ACCBAL', 'LOGLOSS', 'AUCROC', 'CI'.

    Returns:
        The optimization direction, either 'minimize' or 'maximize'.

    Raises:
        ValueError: If the metric provided is not recognized.
    """
    direction = {
        "MAE": "minimize",
        "MSE": "minimize",
        "R2": "maximize",
        "ACC": "maximize",
        "ACCBAL": "maximize",
        "LOGLOSS": "minimize",
        "AUCROC": "maximize",
        "CI": "maximize",
        "AUCPR": "maximize",
        "F1": "maximize",
        "NEGBRIERSCORE": "minimize",
    }

    if metric not in direction:
        raise ValueError(
            f"""Unknown metric '{metric}'.
            Available metrics are: {list(direction.keys())}."""
        )

    return direction[metric]


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
