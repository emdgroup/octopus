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

from pathlib import Path
from attrs import define, field, validators
from octopus.modules.octo.bag import Bag


@define
class EnSel:
    """Ensemble Selection."""

    target_metric: str = field(validator=[validators.instance_of(str)])
    target_assignments: dict = field(validator=[validators.instance_of(dict)])
    path_trials: Path = field(validator=[validators.instance_of(Path)])
    max_n_best_models: int = field(validator=[validators.instance_of(int)])
    max_n_iterations: int = field(validator=[validators.instance_of(int)])

    bags: dict = field(init=False, validator=[validators.instance_of(dict)])

    def __attrs_post_init__(self):
        # initialization here due to "Python immutable default"
        self.bags = dict()
        # get a trials existing in path_trials
        self._collect_trials()

    def _collect_trials(self):
        """Get all trials save in path_trials and store properties in self.bags."""
        # Get all .pkl files in the directory
        pkl_files = [
            file
            for file in self.path_trials.iterdir()
            if file.is_file() and file.suffix == ".pkl"
        ]

        # fill bags dict
        for file in pkl_files:
            bag = Bag.from_pickle(file)
            self.bags[file] = {
                "id": bag.bag_id,
                "scores": bag.get_scores(),
                "predictions": bag.get_predictions(),
                "n_features_used_mean": bag.n_features_used_mean,
            }

    def get_ens_input(self):
        """Get ensemble dict."""
        return self.bags
