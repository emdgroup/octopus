"""MRMR Module."""

from pathlib import Path

import numpy as np
import pandas as pd
from attrs import define, field, validators
from scipy.stats import rankdata

from octopus.experiment import OctoExperiment

# MRMR module
# (1) Inputs:
#     - development feature importances, averaged or for each model (counts, PFI, Shap)
#     - correlation_type: 'pearson', 'rdc'
#     - n_features: number of features to be extracted
# (2) Outpus:
#     - selected features, saved in experiment
# (2) Feature importances from the development dataset are preferable as they show
#     features that are relevant for model generalization.
# (3) MRMR must not be done on test feature importances to avoid information leakage as
#     the MRMR module may be preprocessing step for later model trainings.
#     In this module the  features are taken from the traindev dataset.

# TOBEDONE:
# (2) check inputs, especially feature importances
# (5) how to avoid re-running the preceeding training again
# (6) measure time for mrmr procedure
# (7) mean or count -- input
# (8) internal, shape, permutation -- input


@define
class Mrmr:
    """MRMR."""

    experiment: OctoExperiment = field(
        validator=[validators.instance_of(OctoExperiment)]
    )

    @property
    def path_module(self) -> Path:
        """Module path."""
        return self.experiment.path_study.joinpath(self.experiment.path_sequence_item)

    @property
    def path_results(self) -> Path:
        """Results path."""
        return self.path_module.joinpath("results")

    @property
    def x_traindev(self) -> pd.DataFrame:
        """x_train."""
        return self.experiment.data_traindev[self.experiment.feature_columns]

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

    def __attrs_post_init__(self):
        # initialization here due to "Python immutable default"
        # MRMR should not be the first sequence item
        if self.experiment.sequence_item_id == 0:
            raise ValueError("MRMR module should not be first sequence item.")

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

        # calcualte MRMR features
        selected_mrmr_features = self._maxrminr()

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
        if n_features > len(fi_df):
            n_features = len(fi_df)

        features_lst = fi_df.index.astype(str).tolist()
        print("MRMR: Number of important features: ", len(features_lst))

        # print('Number of available features:', features_df.shape[1])
        features_df = self.x_traindev[features_lst].copy()

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

            # compute (absolute) correlations between the last selected feature and
            # all the (currently) excluded features
            if i > 0:
                last_selected = selected[-1]

                if correlation_type == "pearson":
                    # calculate correlation (pearson)
                    corr.loc[not_selected, last_selected] = (
                        features_df[not_selected]
                        .corrwith(features_df[last_selected])
                        .abs()
                        .clip(0.00001)
                    )
                elif correlation_type == "rdc":
                    # calculate  RDC correlation
                    # corr_rdc.loc[not_selected, last_selected] =
                    for ns in not_selected:
                        corr.loc[ns, last_selected] = np.clip(
                            self._rdc(
                                features_df[ns].to_numpy(),
                                features_df[last_selected].to_numpy(),
                            ),
                            0.0001,
                            None,
                        )
                else:
                    raise ValueError

            # compute FCQ score for all the (currently) excluded features
            # (this is Formula 2)
            score_df = f_df.loc[not_selected].copy()
            score_df["corr"] = (
                corr.loc[not_selected, selected].mean(axis=1).fillna(0.00001)
            )
            score_df["score"] = score_df["FeatureImportance"] / score_df["corr"]

            # find best feature, add it to selected and remove it from not_selected
            best = score_df["score"].idxmax()

            # best_row = score_df.loc[score_df['score'].argmax()]
            selected.append(best)
            not_selected.remove(best)

        selected_features_df = pd.DataFrame(selected, columns=["features"])
        selected_features_df["features"] = selected_features_df["features"].astype(str)

        return selected_features_df

    def _rdc(self, x, y, f=np.sin, k=20, s=1 / 6.0, n=1):
        """Randomized Dependence Coefficient.

        Computes the Randomized Dependence Coefficient
        x,y: numpy arrays 1-D or 2-D
            If 1-D, size (samples,)
            If 2-D, size (samples, variables)
        f:   function to use for random projection
        k:   number of random projections to use
        s:   scale parameter
        n:   number of times to compute the RDC and
            return the median (for stability)
        According to the paper, the coefficient should be relatively insensitive to
        the settings of the f, k, and s parameters.

        Implements the Randomized Dependence Coefficient
        David Lopez-Paz, Philipp Hennig, Bernhard Schoelkopf
        http://papers.nips.cc/paper/5138-the-randomized-dependence-coefficient.pdf
        """
        if n > 1:
            values = []
            for _ in range(n):
                try:
                    values.append(self._rdc(x, y, f, k, s, 1))
                except np.linalg.linalg.LinAlgError:
                    pass
            return np.median(values)

        if len(x.shape) == 1:
            x = x.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        # Copula Transformation
        cx = np.column_stack([rankdata(xc, method="ordinal") for xc in x.T]) / float(
            x.size
        )
        cy = np.column_stack([rankdata(yc, method="ordinal") for yc in y.T]) / float(
            y.size
        )

        # Add a vector of ones so that w.x + b is just a dot product
        o = np.ones(cx.shape[0])
        x = np.column_stack([cx, o])
        y = np.column_stack([cy, o])

        # Random linear projections
        rx = (s / x.shape[1]) * np.random.randn(x.shape[1], k)
        ry = (s / y.shape[1]) * np.random.randn(y.shape[1], k)
        x = np.dot(x, rx)
        y = np.dot(y, ry)

        # Apply non-linear function to random projections
        fx = f(x)
        fy = f(y)

        # Compute full covariance matrix
        c = np.cov(np.hstack([fx, fy]).T)

        # Due to numerical issues, if k is too large,
        # then rank(fX) < k or rank(fY) < k, so we need
        # to find the largest k such that the eigenvalues
        # (canonical correlations) are real-valued
        k0 = k
        lb = 1
        ub = k
        while True:

            # Compute canonical correlations
            cxx = c[:k, :k]
            cyy = c[k0 : k0 + k, k0 : k0 + k]
            cxy = c[:k, k0 : k0 + k]
            cyx = c[k0 : k0 + k, :k]

            eigs = np.linalg.eigvals(
                np.dot(
                    np.dot(np.linalg.pinv(cxx), cxy), np.dot(np.linalg.pinv(cyy), cyx)
                )
            )

            # Binary search if k is too large
            if not (
                np.all(np.isreal(eigs)) and 0 <= np.min(eigs) and np.max(eigs) <= 1
            ):
                ub -= 1
                k = (ub + lb) // 2
                continue
            if lb == ub:
                break
            lb = k
            if ub == lb + 1:
                k = ub
            else:
                k = (ub + lb) // 2

        return np.sqrt(np.max(eigs))


@define
class MrmrConfig:
    """MRMR Config."""

    module: str = field(default="mrmr")
    """Models for ML."""

    description: str = field(validator=[validators.instance_of(str)], default=None)
    """Description."""

    n_features: int = field(validator=[validators.instance_of(int)], default=30)
    """Number of features selected by MRMR."""

    correlation_type: bool = field(
        validator=[validators.in_(["pearson", "rdc"])], default="pearson"
    )
    """Selection of correlation type."""
