"""MRMR Module."""

import numpy as np
import pandas as pd
from attrs import define, field, validators

from octopus.experiment import OctoExperiment
from octopus.modules.utils import rdc

# MRMR module
# (1) Inputs:
#     - development feature importances, averaged or for each model (counts, PFI, Shap)
#     - correlation_type: 'pearson', 'rdc'
#     - n_features: number of features to be extracted
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
# (1) use counts instead of pfi
# (2) use combined counts and pfi !!! as input
# (3) how to deal with grouped features ???
# (4) use results from ensemble selection (count + pfi)
# (5) saving results? any plots?
# ()


@define
class Mrmr:
    """MRMR."""

    experiment: OctoExperiment = field(
        validator=[validators.instance_of(OctoExperiment)]
    )

    @property
    def data_traindev(self) -> pd.DataFrame:
        """data_traindev."""
        return self.experiment.data_traindev

    @property
    def correlation_type(self) -> str:
        """Correlation type."""
        return self.experiment.ml_config["correlation_type"]

    @property
    def n_features(self) -> int:
        """Number of features selected by MRMR."""
        return self.experiment.ml_config["n_features"]

    @property
    def feature_importances(self) -> dict:
        """Feature importances calculated by preceding module."""
        return self.experiment.prior_feature_importances

    @property
    def feature_importance_key(self) -> str:
        """Feature importance key."""
        fi_type = self.experiment.ml_config["feature_importance_type"]
        fi_method = self.experiment.ml_config["feature_importance_method"]
        if fi_method == "internal":
            key = "internal" + "_" + fi_type
        else:
            key = fi_method + "_dev" + "_" + fi_type
        return key

    def __attrs_post_init__(self):
        # initialization here due to "Python immutable default"
        # MRMR should not be the first sequence item
        if self.experiment.sequence_item_id == 0:
            raise ValueError("MRMR module should not be the first sequence item.")

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

        # check if feature_importance key exists
        if self.feature_importance_key not in self.experiment.prior_feature_importances:
            raise ValueError(
                f"No feature importances available for "
                f"key {self.feature_importance_key}"
            )

        # (a) get feature importances
        # (b) only use feaures with positive importances
        fi_df = self.experiment.prior_feature_importances[self.feature_importance_key]
        print("Number of features in provided fi table: ", len(fi_df))
        fi_df = fi_df[fi_df["importance"] > 0].reset_index()
        print("Number features with positive importance: ", len(fi_df))

        # calculate MRMR features
        selected_mrmr_features = self._maxrminr(
            fi_df,
            n_features=self.n_features,
            correlation_type=self.correlation_type,
        )

        # save features selected by mrmr
        self.experiment.selected_features = selected_mrmr_features

        return self.experiment

    def _maxrminr(self, fi_df, n_features=30, correlation_type="pearson"):
        """MRMR function.

        Computes maximum relevant and minimum redundant features.
        FI_df: data frame with feature importances
        correlation_type: 'pearson', 'rdc'
        n_features: number of features to be extracted
        """
        FLOOR = 0.001

        # remove all group features
        fi_df = fi_df[~fi_df["feature"].str.startswith("group")]

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
                else:
                    raise ValueError
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

        print("selected: ", selected)
        return selected


@define
class MrmrConfig:
    """MRMR Config."""

    module: str = field(default="mrmr")
    """Module name."""

    load_sequence_item: bool = field(
        init=False, validator=validators.instance_of(bool), default=False
    )
    """Load existing sequence item, fixed, set to False"""

    description: str = field(validator=[validators.instance_of(str)], default=None)
    """Description."""

    n_features: int = field(validator=[validators.instance_of(int)], default=30)
    """Number of features selected by MRMR."""

    correlation_type: str = field(
        validator=[validators.in_(["pearson", "rdc"])], default="pearson"
    )
    """Selection of correlation type."""

    feature_importance_type: str = field(
        validator=[validators.in_(["mean", "count"])], default="mean"
    )
    """Selection of feature importance type."""

    feature_importance_method: str = field(
        validator=[validators.in_(["permutation", "shap", "internal", "lofo"])],
        default="permutation",
    )
    """Selection of feature importance method."""
