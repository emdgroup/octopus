"""MRMR core function.

TOBEDONE:
(1) Is there a way to consider groups?
(2) relevance-type "permutation", importance_type="permutation" ?
(3) add mutual information to relevance methods
(4) saving results? any plots?
"""

import pandas as pd
from attrs import define, field, validators
from sklearn.feature_selection import f_classif, f_regression

from octopus.experiment import OctoExperiment
from octopus.logger import LogGroup, get_logger
from octopus.modules.utils import rdc_correlation_matrix

logger = get_logger()


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
    def results_key(self) -> str:
        """Results key."""
        return self.experiment.ml_config.results_key

    @property
    def n_features(self) -> int:
        """Number of features selected by MRMR."""
        return self.experiment.ml_config.n_features

    @property
    def feature_importances(self) -> dict:
        """Feature importances calculated by preceding module."""
        return self.experiment.prior_results[self.results_key].feature_importances

    @property
    def feature_importance_key(self) -> str:
        """Feature importance key."""
        fi_type = self.experiment.ml_config.feature_importance_type
        fi_method = self.experiment.ml_config.feature_importance_method
        return (
            f"{'internal' if fi_method == 'internal' else fi_method + '_dev'}_{fi_type}"
        )

    def __attrs_post_init__(self):
        logger.set_log_group(LogGroup.PROCESSING, "MRMR")
        self._validate_configuration()

    def _validate_configuration(self):
        """Validate MRMR configuration.

        1. MRMR should not be the first sequence item
        2. Check if results_key exists
        3. Check if feature_importance key exists

        """
        if self.relevance_type == "permutation":
            if self.experiment.sequence_id == 0:
                raise ValueError("MRMR module should not be the first sequence item.")
            if self.results_key not in self.experiment.prior_results:
                raise ValueError(
                    f"Specified results key not found: {self.results_key}. "
                    "Available results keys: "
                    f"{list(self.experiment.prior_feature_importances.keys())}"
                )
            if self.feature_importance_key not in self.feature_importances:
                raise ValueError(
                    "No feature importances available for "
                    f"key {self.feature_importance_key}."
                    f"Available keys: {self.feature_importances.keys()}"
                )

    def run_experiment(self):
        """Run mrmr module on experiment."""
        self._log_experiment_info()

        relevance_df = self._get_relevance_data()
        mrmr_dict = self._calculate_mrmr_features(relevance_df)

        # get value of first and only item
        selected_mrmr_features = list(mrmr_dict.values())[0]

        # save features selected by mrmr
        self.experiment.selected_features = sorted(
            selected_mrmr_features, key=lambda s: (len(s), s)
        )
        logger.info(f"Selected features: {self.experiment.selected_features}")

        return self.experiment

    def _log_experiment_info(self):
        """Log basic MRMR Info."""
        logger.info("MRMR-Module")
        logger.info(f"Experiment: {self.experiment.experiment_id}")
        logger.info(f"Sequence item: {self.experiment.sequence_id}")
        logger.info(f"Number of features selected by MRMR: {self.n_features}")
        logger.info(f"Correlation type used by MRMR: {self.correlation_type}")
        logger.info(f"Relevance type used by MRMR: {self.relevance_type}")
        logger.info(f"Specified results key: {self.results_key}")
        logger.info(
            f"Available results keys: {list(self.experiment.prior_results.keys())}"
        )

    def _get_relevance_data(self):
        if self.relevance_type == "permutation":
            return self._get_permutation_relevance()
        elif self.relevance_type == "f-statistics":
            return self._get_fstats_relevance()
        else:
            raise ValueError(
                f"Relevance type {self.relevance_type} not supported for MRMR."
            )

    def _get_permutation_relevance(self):
        """Get permutation relevance.

        Only use features with positive importance
        Reduce fi table to feature_columns (previous selected_features).
        Feature columns do not contain any groups.
        """
        re_df = self.feature_importances[self.feature_importance_key]
        re_df = re_df[re_df["feature"].isin(self.feature_columns)]
        logger.info(
            f"Number of features in fi table "
            f"(based on previous selected features, no groups): {len(re_df)}"
        )
        re_df = re_df[re_df["importance"] > 0].reset_index(drop=True)
        logger.info(f"Number features with positive importance: {len(re_df)}")
        return re_df

    def _get_fstats_relevance(self):
        """Get fstats relevance."""
        return relevance_fstats(
            self.x_traindev, self.y_traindev, self.feature_columns, self.ml_type
        )

    def _calculate_mrmr_features(self, relevance_df):
        """Calculate MRMR features."""
        return maxrminr(
            features=self.x_traindev,
            fi_df=relevance_df,
            n_features_lst=[self.n_features],
            correlation_type=self.correlation_type,
        )


# shared functions
def relevance_fstats(
    features: pd.DataFrame,
    target: pd.DataFrame,
    feature_columns: list,
    ml_type: str,
) -> pd.DataFrame:
    """Calculate f-statistics based relevance."""
    features = features[feature_columns]
    target = target.to_numpy().ravel()

    if ml_type == "classification":
        values, _ = f_classif(features, target)
    elif ml_type == "regression":
        values, _ = f_regression(features, target)
    else:
        raise ValueError(f"ML-type {ml_type} not supported.")

    return pd.DataFrame({"feature": feature_columns, "importance": values})


def maxrminr(
    features: pd.DataFrame,
    relevance: pd.DataFrame,
    requested_feature_counts: list[int],
    correlation_type: str = "pearson",
    method: str = "ratio",
) -> dict[int, list[str]]:
    """Perform mRMR feature selection.

    The followings steps are done:
      1. Determine the relevance of all predictor variables and select the feature
         with the highest relevance.
      2. Determine the mean redundancy between the remaining features and all features
         selected so far.
      3. Calculate an importance score as either ratio or difference between relevance
         and redundancy to select the next feature.
      4. Recalculate importance scores and select the next best feature.
      5. Repeat until the desired number of features (n_features_to_select) is reached.


    Further remarks:
      1. Qubim either uses F1, Smolonogov-stats or Shapley as input
         https://github.com/smazzanti/mrmr/blob/main/mrmr/pandas.py
         use this implementation?

      2. Feature importance from the development dataset are preferable as they show
         features that are relevant for model generalization.

      3. MRMR must not be done on test feature importances to avoid information leakage
         as the MRMR module may be preprocessing step for later model trainings.
         In this module the  features are taken from the traindev dataset.

      4. We ignore selected_features from the previous sequence item. The features
         used are extracted from the feature importance table

    Literature:
        https://github.com/ThomasBury/arfs?tab=readme-ov-file
        https://ar5iv.labs.arxiv.org/html/1908.05376
        https://github.com/smazzanti/mrmr

    Args:
        features: Dataset with columns as feature names.
        relevance: Must contain:
            - "feature": Name of the feature
            - "importance": Numeric measure of its relevance
        requested_feature_counts:
            A list of feature counts (e.g., [1, 3, 5]) for which
            partial selection snapshots will be returned.
        correlation_type:
            Correlation method, e.g., "pearson", "spearman", or "rdc"
            (if implemented). Default is "pearson".
        method: Score method, e.g., "ratio" or "difference".

    Returns:
        dict: A dictionary with the MRMR feature selection for given counts.

    Raises:
        ValueError: If correlation_type is not one of {'pearson', 'spearman', 'rdc'},
                    or if method is not either 'ratio' or 'difference'.
    """

    def _adjust_counts(max_feats: int, counts: list[int]) -> list[int]:
        """Adjust requested counts.

        Ensure requested counts do not exceed available features
        Add length of relevant features if not in list.
        """
        valid_counts = [c for c in counts if c <= max_feats]
        if max_feats not in valid_counts:
            valid_counts.append(max_feats)
        return valid_counts

    # Validate correlation type
    if correlation_type not in ["pearson", "spearman", "rdc"]:
        raise ValueError(
            "Correlation_type must be one of {'pearson', 'spearman', 'rdc'}"
        )

    # Validate method
    if method not in ["ratio", "difference"]:
        raise ValueError("Method must be either 'ratio' or 'difference'.")

    # Extract relevant features
    relevant_features = relevance["feature"].unique().tolist()
    max_relevant_features = len(relevant_features)
    requested_feature_counts = _adjust_counts(
        max_relevant_features, requested_feature_counts
    )

    # Convert features DataFrame to only relevant columns
    features_df = features[relevant_features].copy()
    relevance_df = relevance.copy(deep=True)

    # Precompute correlation matrix with absolute values
    if correlation_type == "pearson":
        corr_matrix = features_df.corr(method="pearson").abs()
    elif correlation_type == "spearman":
        corr_matrix = features_df.corr(method="spearman").abs()
    else:  # "rdc"
        corr_values = rdc_correlation_matrix(features_df)
        corr_matrix = pd.DataFrame(
            corr_values,
            index=features_df.columns,
            columns=features_df.columns,
        ).abs()

    # Prepare iterative selection
    selected_features: list[str] = []
    not_selected = set(features_df.columns)
    results = {}

    for i in range(1, max_relevant_features + 1):
        # Build candidate DataFrame for unselected features
        candidate_df = relevance_df[relevance_df["feature"].isin(not_selected)].copy()

        if i == 1:
            # First feature purely by importance
            candidate_df["score"] = candidate_df["importance"]
        else:
            # Calculate mean redundancy
            # features with correlation 1 should not be selected -> redundancy=Inf
            # set lower boundary to avoid divide-by-zero
            candidate_features = candidate_df["feature"].values
            candidate_corrs = corr_matrix.loc[candidate_features, selected_features]
            mean_redundancies = (
                candidate_corrs.replace(1.0, float("Inf"))
                .mean(axis=1)
                .clip(lower=0.001)
            )
            candidate_df["redundancy"] = mean_redundancies.values

            # calculate score
            if method == "ratio":
                candidate_df["score"] = (
                    candidate_df["importance"] / candidate_df["redundancy"]
                )
            else:  # "difference"
                candidate_df["score"] = (
                    candidate_df["importance"] - candidate_df["redundancy"]
                )

        # Select best feature by score
        best_feature = candidate_df.loc[candidate_df["score"].idxmax(), "feature"]
        selected_features.append(best_feature)
        not_selected.remove(best_feature)

        # Store intermediate results for requested counts
        if i in requested_feature_counts:
            results[i] = selected_features.copy()

    return results
