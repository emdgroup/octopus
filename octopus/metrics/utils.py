"""Metrics utility functions."""

import pandas as pd

from octopus.metrics import metrics_inventory


def get_performance_from_model(
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
        estimate = model.predict(data[feature_columns])
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
        except ValueError as e:
            raise ValueError(f"positive_class {positive_class} not found in model classes {classes}") from e

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


def get_performance_from_predictions(
    predictions: dict,
    target_metric: str,
    target_assignments: dict,
    threshold: float = 0.5,
    positive_class: int | str | None = None,
) -> dict:
    """Calculate model performance from predictions dict (from bag.get_predictions()).

    Args:
        predictions: Dictionary with predictions from bag.get_predictions().
                    Expected structure: {training_id: {partition: df}, 'ensemble': {partition: df}}
                    where partition is 'train', 'dev', or 'test'
        target_metric: Name of the metric to calculate
        target_assignments: Dictionary mapping target types to column names
        threshold: Classification threshold (default: 0.5)
        positive_class: Positive class for binary classification (required for classification)

    Returns:
        Dictionary with performance values for each training and ensemble

    Raises:
        ValueError: If positive_class is not provided for classification tasks,
                   if predict_proba is used for regression, or if ml_type is unknown
    """
    metric_config = metrics_inventory.get_metric_config(target_metric)
    metric_function = metrics_inventory.get_metric_function(target_metric)
    ml_type = metric_config.ml_type
    prediction_type = metric_config.prediction_type

    performance = {}

    for training_id, partitions in predictions.items():
        performance[training_id] = {}

        for part, pred_df in partitions.items():
            target_col = list(target_assignments.values())[0]
            target = pred_df[target_col]

            if ml_type == "timetoevent":
                estimate = pred_df["prediction"]
                event_time = pred_df[target_assignments["duration"]].astype(float)
                event_indicator = pred_df[target_assignments["event"]].astype(bool)
                perf_value = metric_function(event_indicator, event_time, estimate)[0]

            elif ml_type == "classification":
                if positive_class is None:
                    raise ValueError("positive_class must be provided for classification tasks")

                if prediction_type == "predict_proba":
                    probabilities = pred_df[positive_class]
                    perf_value = metric_function(target, probabilities)
                else:
                    probabilities = pred_df[positive_class]
                    predictions_binary = (probabilities >= threshold).astype(int)
                    perf_value = metric_function(target, predictions_binary)

            elif ml_type == "multiclass":
                if prediction_type == "predict_proba":
                    prob_columns = [
                        col
                        for col in pred_df.columns
                        if isinstance(col, (int | float)) and col not in [target_col, "prediction"]
                    ]
                    probabilities = pred_df[prob_columns].values
                    perf_value = metric_function(target, probabilities)
                else:
                    predictions_class = pred_df["prediction"].astype(int)
                    perf_value = metric_function(target, predictions_class)

            elif ml_type == "regression":
                if prediction_type == "predict_proba":
                    raise ValueError("predict_proba not supported for regression")
                predictions_reg = pred_df["prediction"]
                perf_value = metric_function(target, predictions_reg)

            else:
                raise ValueError(f"Unknown ml_type: {ml_type}")

            performance[training_id][part] = float(perf_value)

    return performance


def get_score_from_prediction(
    predictions: dict,
    target_metric: str,
    target_assignments: dict,
    threshold: float = 0.5,
    positive_class: int | str | None = None,
) -> dict:
    """Calculate model performance scores from predictions dict with optimization direction applied.

    This function calls get_performance_from_predictions and then applies the optimization
    direction (maximize/minimize) to convert performance values to scores, similar to
    get_score_from_model.

    Args:
        predictions: Dictionary with predictions from bag.get_predictions().
                    Expected structure: {training_id: {partition: df}, 'ensemble': {partition: df}}
                    where partition is 'train', 'dev', or 'test'
        target_metric: Name of the metric to calculate
        target_assignments: Dictionary mapping target types to column names
        threshold: Classification threshold (default: 0.5)
        positive_class: Positive class for binary classification (required for classification)

    Returns:
        Dictionary with score values for each training and ensemble, with direction applied
    """
    # Get performance values
    performance = get_performance_from_predictions(
        predictions=predictions,
        target_metric=target_metric,
        target_assignments=target_assignments,
        threshold=threshold,
        positive_class=positive_class,
    )

    # Convert performance to score based on optimization direction
    scores = {}
    direction = metrics_inventory.get_direction(target_metric)

    for training_id, partitions in performance.items():
        scores[training_id] = {}
        for part, perf_value in partitions.items():
            if direction == "maximize":
                scores[training_id][part] = perf_value
            else:
                scores[training_id][part] = -perf_value

    return scores


def get_score_from_model(model, data, feature_columns, target_metric, target_assignments, positive_class=None) -> float:
    """Calculate model performance score on dataset for given metric and optimization direction."""
    performance = get_performance_from_model(
        model, data, feature_columns, target_metric, target_assignments, positive_class=positive_class
    )

    # convert performance metric to score
    if metrics_inventory.get_direction(target_metric) == "maximize":
        return performance
    else:
        return -performance
