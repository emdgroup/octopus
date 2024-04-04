"""Ensemble selection."""

# description:
# - (first) based on global HP selection only
# - look at Optuna optimization and re-train N best bags,
#   see def _optimize_splits(self, splits):
# - ?? filter bags that meeting feature constraints, see existing code
# - sort best bags (target-value, not val only) and ensemble best bags
#   till optimum is reached - M best bags
# - ensemble selection using M best bags as a starting point
#   + max_iterations=100
#   + start with first best (M) bags
#   + select new bags from M or full set?


# input parameters:
# - run ensemble selection?
# - number of best models to be saved
# - maximum number of ensemble iterations
