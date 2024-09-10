"""MRMR core function."""

import numpy as np
import pandas as pd
from attrs import define, field, validators
from scipy.stats import spearmanr
from sklearn.feature_selection import (
    f_classif,
    f_regression,
)

from octopus.experiment import OctoExperiment
from octopus.modules.utils import rdc

# MRMR module
# (1) Inputs:
#     - development feature importances, averaged or for each model (counts, PFI, Shap)
#     - correlation_type: 'pearson', 'rdc'
#     - n_features: number of features to be extracted
# (2) Qubim either uses F1, Smolonogov-stats or Shapley as input
#     https://github.com/smazzanti/mrmr/blob/main/mrmr/pandas.py
#     use this implementation?
# (2) Output:
#     - selected features, saved in experiment
# (2) Feature importances from the development dataset are preferable as they show
#     features that are relevant for model generalization.
# (3) MRMR must not be done on test feature importances to avoid information leakage as
#     the MRMR module may be preprocessing step for later model trainings.
#     In this module the  features are taken from the traindev dataset.
# (4) We ignore selected_features from the previous sequence item. The features used are
#     extracted from the feature importance table

# Literature:
# https://github.com/ThomasBury/arfs?tab=readme-ov-file
# https://ar5iv.labs.arxiv.org/html/1908.05376
# https://github.com/smazzanti/mrmr

# TOBEDONE:
# (1) relevance-type "permutation", importance_type="permutation" ?
# (1) add mutual information to relevance methods
# (2) saving results? any plots?


@define
class MrmrCore:
    """MRMR module."""

    experiment: OctoExperiment = field(
        validator=[validators.instance_of(OctoExperiment)]
    )

    @property
    def data_traindev(self) -> pd.DataFrame:
        """data_traindev."""
        return self.experiment.data_traindev

    @property
    def x_traindev(self) -> pd.DataFrame:
        """x_traindev."""
        return self.experiment.data_traindev[self.experiment.feature_columns]

    @property
    def y_traindev(self) -> pd.DataFrame:
        """y_traindev."""
        return self.experiment.data_traindev[
            self.experiment.target_assignments.values()
        ]

    @property
    def feature_columns(self) -> list:
        """feature_columns."""
        return self.experiment.feature_columns

    @property
    def ml_type(self) -> str:
        """ML type."""
        return self.experiment.ml_type

    @property
    def correlation_type(self) -> str:
        """Correlation type."""
        return self.experiment.ml_config.correlation_type

    @property
    def relevance_type(self) -> str:
        """Relevance type."""
        return self.experiment.ml_config.relevance_type

    @property
    def model_name(self) -> str:
        """Model name."""
        return self.experiment.ml_config.model_name

    @property
    def n_features(self) -> int:
        """Number of features selected by MRMR."""
        return self.experiment.ml_config.n_features

    @property
    def feature_importances(self) -> dict:
        """Feature importances calculated by preceding module."""
        return self.experiment.prior_feature_importances[self.model_name]

    @property
    def feature_importance_key(self) -> str:
        """Feature importance key."""
        fi_type = self.experiment.ml_config.feature_importance_type
        fi_method = self.experiment.ml_config.feature_importance_method
        if fi_method == "internal":
            key = "internal" + "_" + fi_type
        else:
            key = fi_method + "_dev" + "_" + fi_type
        return key

    def __attrs_post_init__(self):
        # checks performed when feature importances are used
        if self.relevance_type == "permutation":
            # MRMR should not be the first sequence item
            if self.experiment.sequence_item_id == 0:
                raise ValueError("MRMR module should not be the first sequence item.")
            # check if model_name exists
            if self.model_name not in self.experiment.prior_feature_importances:
                raise ValueError(
                    f"Specified model name not found: {self.model_name}. "
                    f"Available model names: "
                    f"{list(self.experiment.prior_feature_importances.keys())}"
                )
            # check if feature_importance key exists
            if self.feature_importance_key not in self.feature_importances:
                raise ValueError(
                    f"No feature importances available for "
                    f"key {self.feature_importance_key} "
                    f"Available keys: {self.feature_importances.keys()}"
                )

    def run_experiment(self):
        """Run mrmr module on experiment."""
        # return updated experiment object

        # (1) get FI from experiment, check if exist

        # display information
        print("MRMR-Module")
        print(f"Experiment: {self.experiment.experiment_id}")
        print(f"Sequence item: {self.experiment.sequence_item_id}")
        print(f"Number of features selected by MRMR: {self.n_features}")
        print(f"Correlation type used by MRMR: {self.correlation_type}")
        print(f"Relevance type used by MRMR: {self.relevance_type}")
        print(f"Specified model name: {self.model_name}")
        print(
            f"Available model names: "
            f"{list(self.experiment.prior_feature_importances.keys())}"
        )

        # select relevance information
        if self.relevance_type == "permutation":
            # (a) get feature importances
            # (b) only use feaures with positive importances
            re_df = self.feature_importances[self.feature_importance_key]
            print("Number of features in provided fi table: ", len(re_df))
            re_df = re_df[re_df["importance"] > 0].reset_index()
            print("Number features with positive importance: ", len(re_df))
            # remove all group features
            re_df = re_df[~re_df["feature"].str.startswith("group")]
            print("Number of non-group features with positive importance: ", len(re_df))
        elif self.relevance_type == "f-statistics":
            re_df = pd.DataFrame(columns=["feature", "importance"])
            re_df["feature"] = self.feature_columns

            if self.ml_type == "classification":
                values, _ = f_classif(
                    self.x_traindev, self.y_traindev.to_numpy().ravel()
                )
            elif self.ml_type == "regression":
                values, _ = f_regression(
                    self.x_traindev, self.y_traindev.to_numpy().ravel()
                )
            else:
                raise ValueError(f"ML-type {self.ml_type} not supported.")
            re_df["importance"] = values
        else:
            raise ValueError(f"Relevance type  {self.relevance_type} not supported.")

        # calculate MRMR features
        selected_mrmr_features = self._maxrminr(
            re_df,
            n_features=self.n_features,
            correlation_type=self.correlation_type,
        )

        # save features selected by mrmr
        self.experiment.selected_features = sorted(
            selected_mrmr_features, key=lambda s: (len(s), s)
        )
        print("Selected features: ", self.experiment.selected_features)

        return self.experiment

    def _maxrminr(self, fi_df, n_features=30, correlation_type="pearson"):
        """MRMR function.

        Computes maximum relevant and minimum redundant features.
        FI_df: data frame with feature importances
        correlation_type: 'pearson', 'rdc'
        n_features: number of features to be extracted
        """
        FLOOR = 0.001

        # extract features from feature importance table
        fi_features = fi_df["feature"].tolist()

        # number of features requested by MRMR compatible with fi table
        if n_features > len(fi_features):
            n_features = len(fi_features)

        # feature dataframe
        features_df = self.data_traindev[fi_features].copy()

        # start MRMR
        f_df = fi_df.copy(deep=True)

        # initialize correlation matrices
        corr = pd.DataFrame(
            0.00001, index=features_df.columns, columns=features_df.columns
        )

        # initialize list of selected features and list of excluded features
        selected = []
        not_selected = features_df.columns.tolist()

        # repeat n_features times:
        # compute FCQ score for all the features that are currently excluded,
        # then find the best one, add it to selected, and remove it from not_selected
        for i in range(n_features):
            # setup score dataframe
            score_df = f_df[f_df["feature"].isin(not_selected)].copy()

            # compute (absolute) correlations between the last selected feature and
            # all the (currently) excluded features
            if i > 0:
                last_selected = selected[-1]

                if correlation_type == "pearson":
                    # calculate correlation (pearson)
                    corr.loc[not_selected, last_selected] = (
                        features_df[not_selected]
                        .corrwith(features_df[last_selected])
                        .fillna(FLOOR)
                        .abs()
                        .clip(FLOOR)
                    )
                elif correlation_type == "rdc":
                    # calculate  RDC correlation
                    # corr_rdc.loc[not_selected, last_selected] =
                    for ns in not_selected:
                        corr.loc[ns, last_selected] = np.clip(
                            rdc(
                                features_df[ns].to_numpy(),
                                features_df[last_selected].to_numpy(),
                            ),
                            FLOOR,
                            None,
                        )
                elif correlation_type == "spearmanr":
                    # Calculate correlation (Spearman)
                    for col in not_selected:
                        corr_value, _ = spearmanr(
                            features_df[col], features_df[last_selected]
                        )
                        corr_value = max(
                            abs(corr_value), FLOOR
                        )  # Ensure non-negative correlation with a floor value
                        corr.loc[col, last_selected] = corr_value
                else:
                    raise ValueError(
                        f"Correlation type {correlation_type} not supported."
                    )
                # add "corr" column to score_df
                score_df["corr"] = (
                    corr.loc[not_selected, selected]
                    .mean(axis=1)
                    .fillna(FLOOR)
                    .replace(1.0, float("Inf"))
                ).to_numpy()

            else:
                # the selection of the first feature is only based on feature importance
                score_df["corr"] = 1

            # compute FCQ score for all the (currently) excluded features
            # (this is Formula 2)
            score_df["score"] = score_df["importance"] / score_df["corr"]

            # find best feature, add it to selected and remove it from not_selected
            # Find the index of the row with the highest score
            best = score_df.loc[score_df["score"].idxmax(), "feature"]
            # print("best", best)

            # best_row = score_df.loc[score_df['score'].argmax()]
            selected.append(best)
            not_selected.remove(best)

        return selected
