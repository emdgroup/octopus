"""Regression metrics."""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Constants for metric names
MAE = "MAE"
MSE = "MSE"
R2 = "R2"

regression_metrics = {
    MAE: mean_absolute_error,
    MSE: mean_squared_error,
    R2: r2_score,
}
