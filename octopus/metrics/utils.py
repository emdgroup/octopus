"""Metrics utility functions."""

import pandas as pd

from octopus.metrics import metrics_inventory


def add_pooling_performance(pool, scores, target_metric, target_assignments, threshold=0.5, positive_class=None):
    """Add pooling performance scores to scores dict."""
    # calculate pooling scores (soft and hard)

    metric_config = metrics_inventory.get_metric_config(target_metric)
    metric_function = metrics_inventory.get_metric_function(target_metric)
    ml_type = metric_config.ml_type
    prediction_type = metric_config.prediction_type  # type of input required for metric

    if ml_type == "timetoevent":
        for part in pool:
            estimate = pool[part]["prediction"]
            event_time = pool[part][target_assignments["duration"]].astype(float)
            event_indicator = pool[part][target_assignments["event"]].astype(bool)
            ci = metric_function(event_indicator, event_time, estimate)[0]
            scores[part + "_pool_hard"] = float(ci)

    elif ml_type == "classification":
        # positive_class is required for classification
        if positive_class is None:
            raise ValueError("positive_class must be provided for classification tasks")

        if prediction_type == "predict_proba":
            for part in pool:
                target_col = list(target_assignments.values())[0]

                # Use positive_class to get the correct probability column
                probabilities = pool[part][positive_class]
                predictions = pool[part]["prediction"]
                target = pool[part][target_col]
                scores[part + "_pool"] = metric_function(target, probabilities)  # input: probabilities
                # scores[part + "_pool_hard"] = metric_function(target, predictions)

        else:  # "predict"
            for part in pool:
                target_col = list(target_assignments.values())[0]

                # Use positive_class to get the correct probability column
                probabilities = (pool[part][positive_class] >= threshold).astype(int)
                predictions = (pool[part]["prediction"] >= threshold).astype(int)
                target = pool[part][target_col]
                # scores[part + "_pool"] = metric_function(target, probabilities)
                scores[part + "_pool"] = metric_function(target, predictions)  # input: predictions

    elif ml_type == "multiclass":
        if prediction_type == "predict_proba":
            for part in pool:
                target_col = list(target_assignments.values())[0]

                # For multiclass, get all class probabilities
                # Extract all probability columns (exclude prediction and target columns)
                prob_columns = [
                    col
                    for col in pool[part].columns
                    if isinstance(col, (int, float)) and col not in [target_col, "prediction"]
                ]
                probabilities = pool[part][prob_columns].values
                target = pool[part][target_col]

                scores[part + "_pool"] = metric_function(target, probabilities)

        else:  # "predict"
            for part in pool:
                target_col = list(target_assignments.values())[0]
                predictions = pool[part]["prediction"].astype(int)
                target = pool[part][target_col]
                scores[part + "_pool"] = metric_function(target, predictions)

    elif ml_type == "regression":
        for part in pool:
            target_col = list(target_assignments.values())[0]
            predictions = pool[part]["prediction"]
            target = pool[part][target_col]
            scores[part + "_pool"] = metric_function(target, predictions)


def get_performance(
    model, data, feature_columns, target_metric, target_assignments, threshold=0.5, positive_class=None
) -> float:
    """Calculate model performance on dataset for given metric."""
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

    elif ml_type == "classification":
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

    elif ml_type == "multiclass":
        if prediction_type == "predict_proba":
            probabilities = model.predict_proba(input_data)
            # Convert to NumPy array if it's a DataFrame
            if isinstance(probabilities, pd.DataFrame):
                probabilities = probabilities.to_numpy()  # Convert to NumPy array
            performance = metric_function(target, probabilities)

        else:
            predictions = model.predict(input_data)
            if isinstance(predictions, pd.DataFrame):
                predictions = predictions.to_numpy()  # Convert to NumPy array if it's a DataFrame
            performance = metric_function(target, predictions)

    elif ml_type == "regression":
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
    """Calculate model performance score on dataset for given metric and optimizaion direction."""
    performance = get_performance(
        model, data, feature_columns, target_metric, target_assignments, positive_class=positive_class
    )

    # convert performance metric to score
    if metrics_inventory.get_direction(target_metric) == "maximize":
        return performance
    else:
        return -performance
