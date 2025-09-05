"""OctoFull Module."""

from octopus.metrics import metrics_inventory


def add_pooling_scores(pool, scores, target_metric, target_assignments, threshold=0.5):
    """Add pooling scores to scores dict."""
    # calculate pooling scores (soft and hard)

    metric_config = metrics_inventory.get_metric_config(target_metric)
    metric_function = metrics_inventory.get_metric_function(target_metric)
    ml_type = metric_config.ml_type
    prediction_type = metric_config.prediction_type

    if ml_type == "timetoevent":
        for part in pool.keys():
            estimate = pool[part]["prediction"]
            event_time = pool[part][target_assignments["duration"]].astype(float)
            event_indicator = pool[part][target_assignments["event"]].astype(bool)
            ci = metric_function(event_indicator, event_time, estimate)[0]
            scores[part + "_pool_hard"] = float(ci)

    elif ml_type == "classification":
        if prediction_type == "predict_proba":
            for part in pool.keys():
                target_col = list(target_assignments.values())[0]
                probabilities = pool[part][1]  # binary only!!
                predictions = pool[part]["prediction"]
                target = pool[part][target_col]
                scores[part + "_pool_soft"] = metric_function(target, probabilities)
                scores[part + "_pool_hard"] = metric_function(target, predictions)

        else:
            for part in pool.keys():
                target_col = list(target_assignments.values())[0]
                probabilities = (pool[part][1] >= threshold).astype(int)  # binary only!!
                predictions = (pool[part]["prediction"] >= threshold).astype(int)
                target = pool[part][target_col]
                scores[part + "_pool_soft"] = metric_function(target, probabilities)
                scores[part + "_pool_hard"] = metric_function(target, predictions)

    elif ml_type == "regression":
        for part in pool.keys():
            target_col = list(target_assignments.values())[0]
            predictions = pool[part]["prediction"]
            target = pool[part][target_col]
            scores[part + "_pool_hard"] = metric_function(target, predictions)
