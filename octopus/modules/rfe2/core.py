"""RFE2 core function."""

import copy

from attrs import define

from octopus.modules.octo.bag import Bag
from octopus.modules.octo.core import OctoCore

# RF2 TOBEDONE:
#  - train bag in standard octo way
#  - give bag and hyperparameters to rfe-process
#  - (start) train bag und standard hyperparameters
#  - calculate group_pfi and shap
# - remove least important feature
#   -- automatically remove not used features
#   -- deal with group features, needs to be intelligent
#   -- deal with negative feature importances
#   -- consider count information
# - record performance
# - go back to start and repeat
# - select the model, different approaches
#   -- persimonial
#   -- best model
# - model retraining after n removal, or start use module several times
# - autogluon: add 3-5 random feature and remove all feature below the lowest random


@define
class Rfe2Core(OctoCore):
    """Rfe2 Core."""

    @property
    def config(self) -> dict:
        """Module configuration."""
        return self.experiment.ml_config

    @property
    def feature_groups(self) -> dict:
        """Feaure groups."""
        return self.experiment.feature_groups

    @property
    def fi_method(self) -> str:
        """Feature importance method."""
        return self.config.fi_method_rfe

    @property
    def step(self) -> int:
        """Number of feature to be removed in RFE step."""
        return self.config.step

    @property
    def min_features_to_select(self) -> int:
        """Minimum number of features to select."""
        return self.config.smin_features_to_select

    def run_experiment(self):
        """Run experiment."""
        # (1) train and optimize model
        self._optimize_splits(self.data_splits)
        # create best bag
        self._create_best_bag()
        bag_results = copy.deepcopy(self.experiment.results["best"])
        bag_model = bag_results.model
        bag_scores = bag_results.scores
        # model should be found here: self.experiment.results["best"]

        # show config
        print("config:", self.config)

        # (2) run RFE iterations
        while True:
            # get new features
            new_features = self.get_new_features(bag_results)
            # calculate score + store score

            if len(new_features) < self.min_features_to_select:
                break

        # (3) analyze results and select best model
        #    - create and save results object

        return self.experiment

    def get_new_features(self, bag: Bag) -> list:
        """Calculate new features, rfe step."""
        if self.fi_method == "permutation":
            fi_df = bag.feature_importances["permutation_dev_mean"]
        elif self.fi_method == "shap":
            fi_df = bag.feature_importances["shap_dev_mean"]

        # only keep nonzero features
        fi_df = fi_df[fi_df["importance"] != 0]

        # calculate absolute values
        fi_df["importance_abs"] = fi_df["importance"].abs()

        # store group features
        groups_df = fi_df[fi_df["feature"].str.startswith("group")].copy()

        # remove all group features -> single features
        fi_df = fi_df[~fi_df["feature"].str.startswith("group")]
        feat_single = fi_df["feature"].tolist()

        # For each feature group with positive importance (only),
        # check if any feature is in feat_single. In not, add the
        # one with the largest feature importance
        groups = groups_df[groups_df["importance"] > 0]["feature"].tolist()
        feat_additional = []
        for key in groups:
            features = self.feature_groups.get(key, [])
            if not any(feature in feat_single for feature in features):
                if features:  # Ensure the list is not empty
                    # Find the feature with the highest importance in fi_df
                    feature_importances = fi_df[fi_df["feature"].isin(features)]
                    if not feature_importances.empty:
                        best_feature = feature_importances.loc[
                            feature_importances["importance"].idxmax(), "feature"
                        ]
                        feat_additional.append(best_feature)

        # Add the additional features to feat_single and remove duplicates
        feat_all = list(set(feat_single + feat_additional))
        print("Number of selected features: ", len(feat_all))

        return sorted(feat_all, key=lambda x: (len(x), sorted(x)))
