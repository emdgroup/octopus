"""Helper functions."""

import math

import numpy as np
import pandas as pd
import scipy.stats
import shap
from scipy.stats import rankdata

from octopus.metrics import metrics_inventory


def get_performance(
    model, data, feature_columns, target_metric, target_assignments, threshold=0.5, positive_class=None
) -> float:
    """Calculate model performance score on dataset for given metric."""
    input_data = data[feature_columns]

    # Ensure input_data is not empty and contains the required feature columns
    if input_data.empty or not all(col in input_data.columns for col in feature_columns):
        raise ValueError("Input data is empty or does not contain the required feature columns.")

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
        # positive_class is required for classification
        if positive_class is None:
            raise ValueError("positive_class must be provided for classification tasks")

        # Determine positive class index
        classes = model.classes_
        try:
            positive_class_idx = list(classes).index(positive_class)
        except ValueError:
            raise ValueError(f"positive_class {positive_class} not found in model classes {classes}")

        if prediction_type == "predict_proba":
            probabilities = model.predict_proba(input_data)
            # Convert to NumPy array if it's a DataFrame
            if isinstance(probabilities, pd.DataFrame):
                probabilities = probabilities.to_numpy()  # Convert to NumPy array

            probabilities = probabilities[:, positive_class_idx]  # Get probabilities for positive class
            performance = metric_function(target, probabilities)

        else:
            probabilities = model.predict_proba(input_data)
            if isinstance(probabilities, pd.DataFrame):
                probabilities = probabilities.to_numpy()  # Convert to NumPy array

            probabilities = probabilities[:, positive_class_idx]  # Get probabilities for positive class
            predictions = (probabilities >= threshold).astype(int)
            performance = metric_function(target, predictions)

    if ml_type == "regression":
        if prediction_type == "predict_proba":
            raise ValueError("predict_proba not supported for regression.")

        else:
            predictions = model.predict(input_data)
            if isinstance(predictions, pd.DataFrame):
                predictions = predictions.to_numpy()  # Convert to NumPy array if it's a DataFrame
            performance = metric_function(target, predictions)

    return performance


def get_performance_score(
    model, data, feature_columns, target_metric, target_assignments, positive_class=None
) -> float:
    """Calculate performance value for given metric."""
    performance = get_performance(
        model, data, feature_columns, target_metric, target_assignments, positive_class=positive_class
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

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(cxx), cxy), np.dot(np.linalg.pinv(cyy), cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and np.min(eigs) >= 0 and np.max(eigs) <= 1):
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


def get_fi_permutation(experiment, n_repeat, data) -> pd.DataFrame:
    """Calculate permutation feature importances."""
    # fixed confidence level
    confidence_level = 0.95
    feature_columns = experiment["feature_columns"]
    data_traindev = experiment["data_traindev"]
    data_test = experiment["data_test"]
    target_assignments = experiment["target_assignments"]
    target_metric = experiment["target_metric"]
    model = experiment["model"]

    # support prediction on new data as well as test data
    if data is None:  # new data
        data = data_test
    if not set(feature_columns).issubset(data.columns):
        raise ValueError("Features missing in provided dataset.")

    # check that targets are in dataset
    # MISSING

    # calculate baseline score
    baseline_score = get_performance_score(model, data, feature_columns, target_metric, target_assignments)

    # get all data select random feature values
    data_all = pd.concat([data_traindev, data], axis=0)

    results_df = pd.DataFrame(
        columns=[
            "feature",
            "importance",
            "stddev",
            "p-value",
            "n",
            "ci_low_95",
            "ci_high_95",
        ]
    )
    for feature in feature_columns:
        data_pfi = data.copy()
        fi_lst = list()

        for _ in range(n_repeat):
            # replace column with random selection from that column of data_all
            # we use data_all as the validation dataset may be small
            data_pfi[feature] = np.random.choice(data_all[feature], len(data_pfi), replace=False)
            pfi_score = get_performance_score(model, data_pfi, feature_columns, target_metric, target_assignments)
            fi_lst.append(baseline_score - pfi_score)

        # calculate statistics
        pfi_mean = np.mean(fi_lst)
        n = len(fi_lst)
        p_value = np.nan
        stddev = np.std(fi_lst, ddof=1) if n > 1 else np.nan
        if stddev not in (np.nan, 0):
            t_stat = pfi_mean / (stddev / math.sqrt(n))
            p_value = scipy.stats.t.sf(t_stat, n - 1)
        elif stddev == 0:
            p_value = 0.5

        # calculate confidence intervals
        if any(np.isnan(val) for val in [stddev, n, pfi_mean]) or n == 1:
            ci_high = np.nan
            ci_low = np.nan
        else:
            t_val = scipy.stats.t.ppf(1 - (1 - confidence_level) / 2, n - 1)
            ci_high = pfi_mean + t_val * stddev / math.sqrt(n)
            ci_low = pfi_mean - t_val * stddev / math.sqrt(n)

        # save results
        results_df.loc[len(results_df)] = [
            feature,
            pfi_mean,
            stddev,
            p_value,
            n,
            ci_low,
            ci_high,
        ]

    return results_df.sort_values(by="importance", ascending=False)


def get_fi_group_permutation(experiment, n_repeat, data) -> pd.DataFrame:
    """Calculate permutation feature importances."""
    # fixed confidence level
    confidence_level = 0.95
    feature_columns = experiment["feature_columns"]
    data_traindev = experiment["data_traindev"]
    data_test = experiment["data_test"]
    target_assignments = experiment["target_assignments"]
    target_metric = experiment["target_metric"]
    model = experiment["model"]
    feature_groups = experiment["feature_group_dict"]

    print("Number of feature groups found and included: ", len(feature_groups))

    # support prediction on new data as well as test data
    if data is None:  # new data
        data = data_test
    if not set(feature_columns).issubset(data.columns):
        raise ValueError("Features missing in provided dataset.")

    # check that targets are in dataset
    # MISSING

    # keep all features and add group features
    # create features dict
    feature_columns_dict = {x: [x] for x in feature_columns}
    features_dict = {**feature_columns_dict, **feature_groups}

    # calculate baseline score
    baseline_score = get_performance_score(model, data, feature_columns, target_metric, target_assignments)

    # get all data select random feature values
    data_all = pd.concat([data_traindev, data], axis=0)

    results_df = pd.DataFrame(
        columns=[
            "feature",
            "importance",
            "stddev",
            "p-value",
            "n",
            "ci_low_95",
            "ci_high_95",
        ]
    )
    # calculate pfi
    for name, feature in features_dict.items():
        data_pfi = data.copy()
        fi_lst = list()

        for _ in range(n_repeat):
            # replace column with random selection from that column of data_all
            # we use data_all as the validation dataset may be small
            for feat in feature:
                data_pfi[feat] = np.random.choice(data_all[feat], len(data_pfi), replace=False)
            pfi_score = get_performance_score(model, data_pfi, feature_columns, target_metric, target_assignments)
            fi_lst.append(baseline_score - pfi_score)

        # calculate statistics
        pfi_mean = np.mean(fi_lst)
        n = len(fi_lst)
        p_value = np.nan
        stddev = np.std(fi_lst, ddof=1) if n > 1 else np.nan
        if stddev not in (np.nan, 0):
            t_stat = pfi_mean / (stddev / math.sqrt(n))
            p_value = scipy.stats.t.sf(t_stat, n - 1)
        elif stddev == 0:
            p_value = 0.5

        # calculate confidence intervals
        if any(np.isnan(val) for val in [stddev, n, pfi_mean]) or n == 1:
            ci_high = np.nan
            ci_low = np.nan
        else:
            t_val = scipy.stats.t.ppf(1 - (1 - confidence_level) / 2, n - 1)
            ci_high = pfi_mean + t_val * stddev / math.sqrt(n)
            ci_low = pfi_mean - t_val * stddev / math.sqrt(n)

        # save results
        results_df.loc[len(results_df)] = [
            name,
            pfi_mean,
            stddev,
            p_value,
            n,
            ci_low,
            ci_high,
        ]

    return results_df.sort_values(by="importance", ascending=False)


def get_fi_shap(
    experiment: dict,
    data: pd.DataFrame,
    shap_type: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Calculate SHAP feature importances.

    Args:
        experiment: A dictionary containing model, test data, and other info.
        data: Data on which to compute SHAP values; if None, uses test data.
        shap_type: Type of SHAP explainer to use ('exact', 'permutation', or 'kernel').

    Returns:
        Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
            - A dataframe (shap_fi_df) of feature importances.
            - A NumPy array (shap_values) containing the SHAP values.
            - The (processed) data used for SHAP value calculation.

    Raises:
        ValueError: If shap_type is not one of 'exact', 'permutation', or 'kernel'.
    """
    # experiment_id = experiment["id"]
    feature_columns = experiment["feature_columns"]
    data_test = experiment["data_test"][feature_columns]
    model = experiment["model"]
    ml_type = experiment["ml_type"]

    # support prediction on new data as well as test data
    if data is None:  # no external data, use test data
        data = data_test

    if not set(feature_columns).issubset(data.columns):
        raise ValueError("Features missing in provided dataset.")

    data = data[feature_columns]

    def predict_wrapper(data):
        if isinstance(data, pd.Series):
            data = data.values.reshape(1, -1)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=feature_columns)
        return model.predict(data)

    def predict_proba_wrapper(data):
        if isinstance(data, pd.Series):
            data = data.values.reshape(1, -1)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=feature_columns)
        return model.predict_proba(data)

    if ml_type == "classification":
        if shap_type == "exact":
            explainer = shap.explainers.Exact(predict_proba_wrapper, data)
        elif shap_type == "permutation":
            explainer = shap.explainers.Permutation(predict_proba_wrapper, data)
        elif shap_type == "kernel":
            explainer = shap.explainers.Kernel(predict_proba_wrapper, data)
        else:
            raise ValueError(f"Shap type {shap_type} not supported.")

        shap_values = explainer.shap_values(data)
        # only use pos class
        shap_values = shap_values[:, :, 1]  # pylint: disable=E1126
    else:
        if shap_type == "exact":
            explainer = shap.explainers.Exact(predict_wrapper, data)
        elif shap_type == "permutation":
            explainer = shap.explainers.Permutation(predict_wrapper, data)
        elif shap_type == "kernel":
            explainer = shap.explainers.Kernel(predict_wrapper, data)
        else:
            raise ValueError(f"Shap type {shap_type} not supported.")

        shap_values = explainer.shap_values(data)

    # save fi to table
    shap_fi_df = pd.DataFrame(shap_values, columns=data.columns)
    shap_fi_df = shap_fi_df.abs().mean().to_frame().reset_index()
    shap_fi_df.columns = ["feature", "importance"]
    shap_fi_df = shap_fi_df.sort_values(by="importance", ascending=False).reset_index(drop=True)
    # remove features with extremely small fi
    # shap feature importances are always non-zero due to round-off errors
    threshold = shap_fi_df["importance"].max() / 1000
    shap_fi_df = shap_fi_df[shap_fi_df["importance"] > threshold]

    return shap_fi_df, shap_values, data


def get_fi_group_shap(experiment, data, shap_type) -> pd.DataFrame:
    """Calculate SHAP feature importances for feature groups."""
    # experiment_id = experiment["id"]
    feature_columns = experiment["feature_columns"]
    data_test = experiment["data_test"][feature_columns]
    model = experiment["model"]
    ml_type = experiment["ml_type"]
    feature_groups = experiment["feature_group_dict"]

    # Support prediction on new data as well as test data
    if data is None:  # No external data, use test data
        data = data_test

    if not set(feature_columns).issubset(data.columns):
        raise ValueError("Features missing in provided dataset.")

    data = data[feature_columns]

    def predict_wrapper(data):
        if isinstance(data, pd.Series):
            data = data.values.reshape(1, -1)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=feature_columns)
        return model.predict(data)

    def predict_proba_wrapper(data):
        if isinstance(data, pd.Series):
            data = data.values.reshape(1, -1)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=feature_columns)
        return model.predict_proba(data)

    if ml_type == "classification":
        if shap_type == "exact":
            explainer = shap.explainers.Exact(predict_proba_wrapper, data)
        elif shap_type == "permutation":
            explainer = shap.explainers.Permutation(predict_proba_wrapper, data)
        elif shap_type == "kernel":
            explainer = shap.explainers.Kernel(predict_proba_wrapper, data)
        else:
            raise ValueError(f"SHAP type {shap_type} not supported.")

        shap_values = explainer.shap_values(data)
        # Only use positive class
        shap_values = shap_values[:, :, 1]  # pylint: disable=E1126
    else:
        if shap_type == "exact":
            explainer = shap.explainers.Exact(predict_wrapper, data)
        elif shap_type == "permutation":
            explainer = shap.explainers.Permutation(predict_wrapper, data)
        elif shap_type == "kernel":
            explainer = shap.explainers.Kernel(predict_wrapper, data)
        else:
            raise ValueError(f"SHAP type {shap_type} not supported.")

        shap_values = explainer.shap_values(data)

    # Calculate mean absolute SHAP values for individual features
    individual_shap = pd.DataFrame(shap_values, columns=data.columns)
    individual_shap = individual_shap.abs().mean().to_frame().reset_index()
    individual_shap.columns = ["feature", "importance"]

    # Calculate mean absolute SHAP values for each feature group
    group_shap = {}
    for group_name, features in feature_groups.items():
        # Ensure the features are in the data
        if not set(features).issubset(data.columns):
            raise ValueError(f"Features in group '{group_name}' are missing in provided dataset.")
        # Create a boolean mask for the features in this group
        feature_mask = np.isin(data.columns, features)
        group_shap_value = np.sum(shap_values[:, feature_mask], axis=1)
        group_shap[f"{group_name}:{features}"] = group_shap_value

    # Create a DataFrame for group SHAP values
    group_shap_df = pd.DataFrame(
        {
            "feature": list(group_shap.keys()),
            "importance": [np.mean(np.abs(feature_imp)) for _, feature_imp in group_shap.items()],
        }
    )

    # Create a new array for combined SHAP values
    # combined_shap_values = np.concatenate(
    #    [
    #        shap_values,
    #        np.array([feature_imp for _, feature_imp in group_shap.items()]).T,
    #    ],
    #    axis=1,
    # )

    # Create a DataFrame for the combined SHAP values
    # combined_feature_names = list(data.columns) + list(group_shap.keys())
    # combined_df = pd.DataFrame(combined_shap_values, columns=combined_feature_names)

    # results_path = self.path_results

    # (A) Bar plot
    # save_path = results_path.joinpath(
    #    f"model_groupshap_fi_barplot_exp{experiment_id}.pdf",
    # )
    # with PdfPages(save_path) as pdf:
    #    shap.summary_plot(
    #        combined_shap_values, combined_df, plot_type="bar", show=False
    #    )
    #    plt.title("Combined SHAP Feature Importance (Individual + Group)")
    #    plt.tight_layout()
    #    pdf.savefig(plt.gcf(), orientation="portrait")
    #    plt.close()
    #
    # (B) Beeswarm plot
    # save_path = results_path.joinpath(
    #    f"model_groupshap_fi_beeswarm_exp{experiment_id}.pdf",
    # )
    # with PdfPages(save_path) as pdf:
    #    plt.figure(figsize=(8.27, 11.69))  # portrait orientation (A4)
    #    shap.summary_plot(
    #        combined_df.values,
    #        np.concatenate(
    #            [data.values, np.zeros((data.shape[0], len(group_shap)))], axis=1
    #        ),
    #        plot_type="dot",
    #        feature_names=combined_feature_names,
    #        max_display=20,
    #        show=False,
    #    )
    #    plt.title("Combined SHAP Feature Importance (Individual + Group)")
    #   plt.tight_layout()
    #    pdf.savefig(plt.gcf(), orientation="portrait")
    #   plt.close()

    # Combine individual and group SHAP values
    combined_shap_df = pd.concat([individual_shap, group_shap_df], ignore_index=True)
    combined_shap_df = combined_shap_df.sort_values(by="importance", ascending=False).reset_index(drop=True)

    # Remove features with extremely small importance
    # threshold = combined_shap_df["importance"].max() / 1000
    # combined_shap_df = combined_shap_df[combined_shap_df["importance"] > threshold]

    return combined_shap_df
