"""RFE2."""

import copy

import numpy as np
import pandas as pd
from attrs import Factory, define, field, validators

from octopus.modules.octo.bag import Bag
from octopus.modules.octo.core import OctoCore
from octopus.results import ModuleResults

# TOBEDONE
# - fix selected features = empty list ?????
# - remove least important feature
#   -- automatically remove not used features
#   -- deal with group features, needs to be intelligent
#   -- deal with negative feature importances
#   -- consider count information (included in feature/mean)
# - pfi: consider groups, consider counts
# - do we need the step input?
# - jason output in results, see rfe
# - how to override/disable OcoCore inputs hat are not needed?
# - model retraining after n removal, or start use module several times
# - autogluon: add 3-5 random feature and remove all feature below the lowest random


@define
class Rfe2Core(OctoCore):
    """Rfe2 Core."""

    # Optional attribute (with default value)
    results: pd.DataFrame = field(
        init=False,
        default=Factory(pd.DataFrame),
        validator=[validators.instance_of(pd.DataFrame)],
    )
    """RFE results dataframe."""

    @property
    def config(self) -> dict:
        """Module configuration."""
        return self.experiment.ml_config

    @property
    def feature_groups(self) -> dict:
        """Feature groups."""
        return self.experiment.feature_groups

    @property
    def fi_method(self) -> str:
        """Feature importance method."""
        return self.config.fi_method_rfe

    @property
    def step(self) -> int:
        """Number of features to be removed in RFE step."""
        return self.config.step

    @property
    def min_features_to_select(self) -> int:
        """Minimum number of features to select."""
        return self.config.min_features_to_select

    @property
    def selection_method(self) -> str:
        """Method for selection final solution (best/parsimonious)."""
        return self.config.selection_method

    def __attrs_post_init__(self):
        # run OctoCore post_init() to create directory, etc...
        super().__attrs_post_init__()

        # Initialize results DataFrame
        self.results = pd.DataFrame(
            columns=[
                "step",
                "performance_mean",
                "performance_sem",
                "n_features",
                "features",
                "feature_importances",
                "model",
            ]
        )

    def run_experiment(self):
        """Run experiment."""
        # (1) train and optimize model
        self._optimize_splits(self.data_splits)
        # create best bag
        self._create_best_bag()
        bag_results = copy.deepcopy(self.experiment.results["best"])
        bag = bag_results.model
        bag_scores = bag_results.scores
        bag_selected_features = bag_results.selected_features

        # record baseline performance
        step = 0
        dev_lst = bag_scores["dev_lst"]
        self.results.loc[len(self.results)] = {
            "step": step,
            "performance_mean": bag_scores["dev_avg"],
            "performance_sem": np.std(dev_lst, ddof=1) / len(dev_lst),  # no np.sqrt
            "n_features": len(bag_selected_features),
            "features": bag_selected_features,
            "feature_importances": self._get_fi(bag),
            "model": copy.deepcopy(bag),
        }

        self._print_step_information()

        # (2) run RFE iterations
        while True:
            step = step + 1
            # calculate new features
            new_features = self._calculate_new_features(bag)

            if len(new_features) < self.min_features_to_select:
                break

            # retrain bag and calculate feature importances
            bag = self._retrain_and_calc_fi(bag, new_features)

            # get scores
            bag_scores = bag.get_scores()

            # record performance
            dev_lst = bag_scores["dev_lst"]
            self.results.loc[len(self.results)] = {
                "step": step,
                "performance_mean": bag_scores["dev_avg"],
                "performance_sem": np.std(dev_lst, ddof=1) / len(dev_lst),  # no np.sqrt
                "n_features": len(new_features),
                "features": new_features,
                "feature_importances": self._get_fi(bag),
                "model": copy.deepcopy(bag),
            }

            # print step results
            self._print_step_information()

        # (3) analyze results and select best model
        #    - create and save results object
        if self.selection_method == "best":
            selected_row = self.results.loc[self.results["performance_mean"].idxmax()]
        elif self.selection_method == "parsimonious":
            # best performance mean and sem
            best_performance_mean = self.results["performance_mean"].max()
            best_performance_sem = self.results.loc[
                self.results["performance_mean"] == best_performance_mean,
                "performance_sem",
            ].values[0]
            # define threshold for accepting solution with less features
            threshold = best_performance_mean - best_performance_sem
            filtered_df = self.results[self.results["performance_mean"] >= threshold]
            if not filtered_df.empty:
                selected_row = filtered_df.loc[filtered_df["n_features"].idxmin()]
            else:
                # take best value if no solution with less features can be found
                selected_row = self.results.loc[
                    self.results["performance_mean"].idxmax()
                ]

        print("Selected solution:", selected_row)

        # save results to experiment
        best_model = selected_row["model"]
        self.experiment.results["Rfe2"] = ModuleResults(
            id="rfe2",
            model=best_model,
            scores=best_model.get_scores(),
            feature_importances={
                "dev": selected_row["feature_importances"],
            },
            selected_features=best_model.get_selected_features(),
        )

        print("RFE solution:")
        print(
            f"Step: {selected_row['step']}, n_features: {selected_row['n_features']}"
            f", Perf_mean: {selected_row['performance_mean']:.4f}"
            f", Perf_sem: {selected_row['performance_sem']:.4f}"
        )
        print("Selected feautures:", best_model.get_selected_features())
        print("Selected feautures:", selected_row["features"])

        return self.experiment

    def _print_step_information(self):
        """Print step performance."""
        last_row = self.results.iloc[-1]
        print(
            f"Step: {last_row['step']}, n_features: {last_row['n_features']}"
            f", Perf_mean: {last_row['performance_mean']:.4f}"
            f", Perf_sem: {last_row['performance_sem']:.4f}"
        )

    def _retrain_and_calc_fi(self, bag: Bag, new_features: list) -> Bag:
        """Retrain bag using new feature set and calculate feature importances."""
        bag = copy.deepcopy(bag)

        # update feature_columns and feature groups
        feature_groups = self.experiment.calculate_feature_groups(new_features)
        for training in bag.trainings:
            training.feature_columns = new_features
            training.feature_groups = feature_groups

        # update feature groups??

        # retrain bag
        bag.fit()

        # calculate feature importances
        bag.calculate_feature_importances([self.fi_method], partitions=["dev"])

        return bag

    def _get_fi(self, bag: Bag) -> pd.DataFrame:
        """Get relevant feature importances."""
        if self.fi_method == "permutation":
            fi_df = bag.feature_importances["permutation_dev_mean"]
        elif self.fi_method == "shap":
            fi_df = bag.feature_importances["shap_dev_mean"]

        return fi_df

    def _calculate_new_features(self, bag: Bag) -> list:
        """Perfrom RFE step and calculate new features."""
        bag = copy.deepcopy(bag)

        fi_df = self._get_fi(bag)

        # only keep nonzero features
        fi_df = fi_df[fi_df["importance"] != 0]

        # remove all group features -> single features
        fi_df = fi_df[~fi_df["feature"].str.startswith("group")]

        # calculate absolute values
        fi_df["importance_abs"] = fi_df["importance"].abs()
        fi_df = fi_df.sort_values(by="importance_abs", ascending=False)

        # drop the row with the lowest value in the 'importance_abs' column
        fi_df_reduced = fi_df.drop(index=fi_df["importance_abs"].idxmin())

        feat_new = fi_df_reduced["feature"]

        return sorted(feat_new, key=lambda x: (len(x), sorted(x)))
