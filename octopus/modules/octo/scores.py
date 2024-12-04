"""OctoFull Module."""

from octopus.metrics import metrics_inventory


def add_pooling_scores(pool, scores, target_metric, target_assignments):
    """Add pooling scores to scores dict."""
    # calculate pooling scores (soft and hard)
    if target_metric in ["AUCROC", "LOGLOSS"]:
        for part in pool.keys():
            target_col = list(target_assignments.values())[0]
            probabilities = pool[part][1]  # binary only!!
            predictions = pool[part]["prediction"]
            target = pool[part][target_col]
            scores[part + "_pool_soft"] = metrics_inventory[target_metric]["method"](
                target, probabilities
            )
            scores[part + "_pool_hard"] = metrics_inventory[target_metric]["method"](
                target, predictions
            )
    elif target_metric in ["ACC", "ACCBAL"]:
        for part in pool.keys():
            target_col = list(target_assignments.values())[0]
            predictions = pool[part]["prediction"].astype(int)
            target = pool[part][target_col]
            scores[part + "_pool_hard"] = metrics_inventory[target_metric]["method"](
                target, predictions
            )
    elif target_metric in ["CI"]:
        for part in pool.keys():
            estimate = pool[part]["prediction"]
            event_time = pool[part][target_assignments["duration"]].astype(float)
            event_indicator = pool[part][target_assignments["event"]].astype(bool)
            ci, _, _, _, _ = metrics_inventory[target_metric]["method"](
                event_indicator, event_time, estimate
            )
            scores[part + "_pool_hard"] = float(ci)
    elif target_metric in ["R2", "MSE", "MAE"]:
        for part in pool.keys():
            target_col = list(target_assignments.values())[0]
            predictions = pool[part]["prediction"]
            target = pool[part][target_col]
            scores[part + "_pool_hard"] = metrics_inventory[target_metric]["method"](
                target, predictions
            )
    else:
        raise ValueError(f"Unsupported target metric: {target_metric}")
