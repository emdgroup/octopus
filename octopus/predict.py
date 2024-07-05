"""Octopus prediction."""

from pathlib import Path

import pandas as pd
from attrs import define, field, validators

from octopus.data import OctoData
from octopus.experiment import OctoExperiment

# TOBEDONE:
# (1) Inputs:
#    - study, sequence
#    - predict on new dataset or on test
#    - feature importances on new dataset or on test
#    - feature importance method (permutation, shap, shap version)
# (2) Outputs:
#    - predictions on new dataset or test (ensembled)
#    - any performance info?
#    - featur importances per model, averaged -- detailed info on FI.
# (3) Usage:
#     - OctoML.predict() -- create this method
#     - predict-Object
# (4) Other topcis:
#     - OctoML to save study path?
#     - read config from study path
#     - predict-Object: show sequence


# TOBEDONE
# (1) correltly label outputs of probabilities .predict_proba()


@define
class OctoPredict:
    """OctoPredict."""

    study_path: Path = field(validator=[validators.instance_of(Path)])
    """Path to study."""

    sequence_item_id: int = field(
        init=False, default=-1, validator=[validators.instance_of(int)]
    )
    """Sequence item id."""

    experiments: dict = field(init=False, validator=[validators.instance_of(dict)])
    """Dictionary containing model and corresponding test_dataset."""

    @property
    def config(self) -> dict:
        """Study configuration."""
        return OctoData.from_pickle(self.study_path.joinpath("config", "config.pkl"))

    @property
    def n_experiments(self) -> int:
        """Number of experiments."""
        return self.config.n_folds_outer

    def ml_type(self) -> str:
        """ML-type."""
        return self.config.ml_type

    def __attrs_post_init__(self):
        # set last sequence item as default
        if self.sequence_item_id < 0:
            self.sequence_item_id = len(self.config.cfg_sequence) - 1
        # get models
        self.experiments = self._get_models()

    def _get_models(self):
        """Get all models and test data from study path."""
        print("\nLoading available experiments ......")
        experiments = dict()

        for experiment_id in range(self.n_experiments):
            path_exp = self.study_path.joinpath(
                f"experiment{experiment_id}",
                f"sequence{self.sequence_item_id}",
                f"exp{experiment_id}_{self.sequence_item_id}.pkl",
            )
            # extract best model. test dataset, feature columns
            if path_exp.exists():
                print(
                    f"Experiment{experiment_id}, sequence{self.sequence_item_id} found."
                )
                experiment = OctoExperiment.from_pickle(path_exp)
                experiments[experiment_id] = {
                    "model": experiment.models["best"],
                    "data_test": experiment.data_test,
                    "feature_columns": experiment.feature_columns,
                    "row_column": experiment.row_column,
                }
        print(f"{len(experiments)} experiment(s) out of {self.n_experiments} found.")
        return experiments

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict on new data."""
        preds_lst = list()
        for _, experiment in self.experiments.items():

            feature_columns = experiment["feature_columns"]

            if set(feature_columns).issubset(data.columns):
                df = pd.DataFrame(columns=["row_id", "prediction"])
                df["row_id"] = data.columns
                df["prediction"] = experiment["model"].predict(data[feature_columns])
                preds_lst.append(df)
            else:
                raise ValueError("Features missing in provided dataset.")

        print(pd.concat(preds_lst, axis=0))

        grouped_df = (
            pd.concat(preds_lst, axis=0)
            .groupby("row_id")["prediction"]
            .agg(["mean", "std", "count"])
            .rename(
                columns={"mean": "prediction", "std": "prediction_std", "count": "n"},
            )
            .reset_index()
        )

        return grouped_df

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict_proba on new data."""
        preds_lst = list()
        for _, experiment in self.experiments.items():

            feature_columns = experiment["feature_columns"]

            if set(feature_columns).issubset(data.columns):
                df = pd.DataFrame(columns=["row_id", "prediction"])
                df["row_id"] = data.columns
                # only binary predictions are supported
                df["probability"] = experiment["model"].predict_proba(
                    data[feature_columns]
                )[:, 1]
                preds_lst.append(df)
            else:
                raise ValueError("Features missing in provided dataset.")

        print(pd.concat(preds_lst, axis=0))

        grouped_df = (
            pd.concat(preds_lst, axis=0)
            .groupby("row_id")["probability"]
            .agg(["mean", "std", "count"])
            .rename(
                columns={"mean": "probability", "std": "probability_std", "count": "n"},
            )
            .reset_index()
        )
        return grouped_df

    def predict_test(self) -> pd.DataFrame:
        """Predict on available test data."""
        preds_lst = list()
        for _, experiment in self.experiments.items():

            data_test = experiment["data_test"]
            feature_columns = experiment["feature_columns"]
            row_column = experiment["row_column"]

            df = pd.DataFrame(columns=["row_id", "prediction"])
            df["row_id"] = data_test[row_column]
            df["prediction"] = experiment["model"].predict(data_test[feature_columns])
            preds_lst.append(df)

        print(pd.concat(preds_lst, axis=0))

        grouped_df = (
            pd.concat(preds_lst, axis=0)
            .groupby(row_column)["prediction"]
            .agg(["mean", "std", "count"])
            .rename(
                columns={"mean": "prediction", "std": "prediction_std", "count": "n"},
            )
            .reset_index()
        )

        return grouped_df

    def predict_proba_test(self) -> pd.DataFrame:
        """Predict_proba on available test data."""
        preds_lst = list()
        for _, experiment in self.experiments.items():

            data_test = experiment["data_test"]
            feature_columns = experiment["feature_columns"]
            row_column = experiment["row_column"]

            df = pd.DataFrame(columns=["row_id", "probability"])
            df["row_id"] = data_test[row_column]
            # only binary classification!!
            df["probability"] = experiment["model"].predict_proba(
                data_test[feature_columns]
            )[:, 1]
            preds_lst.append(df)

        grouped_df = (
            pd.concat(preds_lst, axis=0)
            .groupby(row_column)["probability"]
            .agg(["mean", "std", "count"])
            .rename(
                columns={"mean": "probability", "std": "probability_std", "count": "n"},
            )
            .reset_index()
        )

        return grouped_df

    def calculate_fi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate feature importances on new data."""

    def calculate_fi_test(self) -> pd.DataFrame:
        """Calculate feature importances on available test data."""
