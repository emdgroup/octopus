"""Octopus prediction."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import shap
from attrs import define, field, validators
from matplotlib.backends.backend_pdf import PdfPages

from octopus.data import OctoData
from octopus.experiment import OctoExperiment
from octopus.modules.metrics import metrics_inventory
from octopus.modules.utils import optuna_direction

# TOBEDONE
# (1) !calculate_fi(data_df)
#     on new data we can use self.predict_proba for calculating fis.
# (2) correltly label outputs of probabilities .predict_proba()
# (3) replace metrics with score, relevant for feature importances
# (4) Permutation importance on group of features
# (5) ? create OctoML.predict(), .calculate_fi()


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

    results: dict = field(init=False, validator=[validators.instance_of(dict)])
    """Results."""

    @property
    def config(self) -> dict:
        """Study configuration."""
        return OctoData.from_pickle(self.study_path.joinpath("config", "config.pkl"))

    @property
    def n_experiments(self) -> int:
        """Number of experiments."""
        return self.config.n_folds_outer

    def __attrs_post_init__(self):
        # initialization here due to "Python immutable default"
        self.results = dict()
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
                    "id": experiment_id,
                    "model": experiment.models["best"],
                    "data_traindev": experiment.data_traindev,
                    "data_test": experiment.data_test,
                    "feature_columns": experiment.feature_columns,
                    "row_column": experiment.row_column,
                    "target_assignments": experiment.target_assignments,
                    "target_metric": experiment.config["target_metric"],
                    "ml_type": experiment.config["ml_type"],
                }
        print(f"{len(experiments)} experiment(s) out of {self.n_experiments} found.")
        return experiments

    def predict(self, data: pd.DataFrame, return_df=False) -> pd.DataFrame:
        """Predict on new data."""
        preds_lst = list()
        for _, experiment in self.experiments.items():
            feature_columns = experiment["feature_columns"]

            if set(feature_columns).issubset(data.columns):
                df = pd.DataFrame(columns=["row_id", "prediction"])
                df["row_id"] = data.index
                df["prediction"] = experiment["model"].predict(data[feature_columns])
                preds_lst.append(df)
            else:
                raise ValueError("Features missing in provided dataset.")

        grouped_df = pd.concat(preds_lst, axis=0).groupby("row_id").mean()

        if return_df is True:
            return grouped_df
        else:
            return grouped_df.to_numpy()

    def predict_proba(self, data: pd.DataFrame, return_df=False) -> pd.DataFrame:
        """Predict_proba on new data."""
        preds_lst = list()
        for _, experiment in self.experiments.items():
            feature_columns = experiment["feature_columns"]
            probabilities = experiment["model"].predict_proba(data[feature_columns])

            if set(feature_columns).issubset(data.columns):
                df = pd.DataFrame()
                df["row_id"] = data.index
                # only binary predictions are supported
                prob_columns = range(probabilities.shape[1])
                for column in prob_columns:
                    df[column] = probabilities[:, column]
                preds_lst.append(df)
            else:
                raise ValueError("Features missing in provided dataset.")

        grouped_df = pd.concat(preds_lst, axis=0).groupby("row_id").mean()

        if return_df is True:
            return grouped_df
        else:
            return grouped_df.to_numpy()

        grouped_df = (
            pd.concat(preds_lst, axis=0)
            .groupby("row_id")["probability"]
            .agg(["mean", "std", "count"])
            .rename(
                columns={"mean": "probability", "std": "probability_std", "count": "n"},
            )
            .reset_index()
        )
        if return_df is True:
            return grouped_df
        else:
            return grouped_df["probability"].to_numpy()

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

    def calculate_fi(
        self,
        data: pd.DataFrame,
        n_repeat: int = 10,
        fi_type: str = "permutation",
        shap_type: str = "exact",
    ) -> pd.DataFrame:
        """Calculate feature importances on new data."""
        if shap_type not in ["exact", "permutation"]:
            raise ValueError("Specified shap_type not supported.")

        # feature importances for every single available experiment/model
        print("Calculating feature importances for every experiment/model.")
        for _, experiment in self.experiments.items():
            exp_id = experiment["id"]
            if fi_type == "permutation":
                results_df = self._get_fi_permutation(experiment, n_repeat, data=data)
                self.results[f"fi_table_permutation_exp{exp_id}"] = results_df
                self._plot_permutation_fi(exp_id, results_df)
            elif fi_type == "shap":
                results_df = self._get_fi_shap(
                    experiment, data=data, shap_type=shap_type
                )
                self.results[f"fi_table_shap_exp{exp_id}"] = results_df
            else:
                raise ValueError("Feature Importance type not supported")

        # feature importances for the combined predictions
        print("Calculating combined feature importances.")
        # create combined experiment
        feature_col_lst = list()
        for exp_id, experiment in self.experiments.items():
            feature_col_lst.extend(experiment["feature_columns"])

        # use last experiment in for loop
        exp_combined = {
            "id": "_all",
            "model": self,
            "data_traindev": pd.concat(
                [experiment["data_traindev"], experiment["data_test"]], axis=0
            ),
            "feature_columns": list(set(feature_col_lst)),
            # same for all experiments
            "data_test": experiment["data_test"],  # not used
            "row_column": experiment["row_column"],
            "target_assignments": experiment["target_assignments"],
            "target_metric": experiment["target_metric"],
            "ml_type": experiment["ml_type"],
        }

        if fi_type == "permutation":
            results_df = self._get_fi_permutation(exp_combined, n_repeat, data=data)
            self.results["fi_table_permutation_ensemble"] = results_df
            self._plot_permutation_fi(exp_combined["id"], results_df)
        elif fi_type == "shap":
            results_df = self._get_fi_shap(exp_combined, data=data, shap_type=shap_type)
            self.results["fi_table_shap_ensemble"] = results_df

    def calculate_fi_test(
        self, n_repeat: int = 10, fi_type: str = "permutation", shap_type: str = "exact"
    ) -> pd.DataFrame:
        """Calculate feature importances on available test data."""
        if shap_type not in ["exact", "permutation"]:
            raise ValueError("Specified shap_type not supported.")

        print("Calculating feature importances for every experiment/model.")
        for _, experiment in self.experiments.items():
            exp_id = experiment["id"]
            if fi_type == "permutation":
                results_df = self._get_fi_permutation(experiment, n_repeat, data=None)
                self.results[f"fi_table_permutation_exp{exp_id}"] = results_df
                self._plot_permutation_fi(exp_id, results_df)
            elif fi_type == "shap":
                results_df = self._get_fi_shap(
                    experiment, data=None, shap_type=shap_type
                )
                self.results[f"fi_table_shap_exp{exp_id}"] = results_df
            else:
                raise ValueError("Feature Importance type not supported")

    def _plot_permutation_fi(self, experiment_id, df):
        """Create plot for permutation fi and save to file."""
        # Calculate error bars
        lower_error = df["importance"] - df["ci_low_95"]
        upper_error = df["ci_high_95"] - df["importance"]
        error = [lower_error.values, upper_error.values]

        save_path = self.study_path.joinpath(
            f"experiment{experiment_id}",
            f"sequence{self.sequence_item_id}",
            "results",
            f"model_permutation_fi_exp{experiment_id}_{self.sequence_item_id}.pdf",
        )
        # create directories if needed, required for id="all"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # plot figure and save to pdf
        with PdfPages(save_path) as pdf:
            plt.figure(figsize=(8.27, 11.69))  # portrait orientation (A4)
            _ = plt.barh(
                df["feature"],
                df["importance"],
                xerr=error,
                capsize=5,
                color="royalblue",
                # edgecolor="black",
            )

            # Adding labels and title
            plt.ylabel("Feature")
            plt.xlabel("Importance")
            plt.title("Feature Importance with Confidence Intervals")
            plt.grid(True, axis="x")

            # Adjust layout to make room for the plot
            plt.tight_layout()

            pdf.savefig(plt.gcf(), orientation="portrait")
            plt.close()

    def _get_performance_score(
        self, model, data, feature_columns, target_metric, target_assignments
    ) -> float:
        """Calculate model performance score on dataset."""
        if target_metric in ["AUCROC", "LOGLOSS"]:
            target_col = list(target_assignments.values())[0]
            target = data[target_col]
            probabilities = model.predict_proba(data[feature_columns])[
                :, 1
            ]  # binary only!!
            score = metrics_inventory[target_metric]["method"](target, probabilities)
        elif target_metric in ["CI"]:
            estimate = model.predict(data)
            event_time = data[target_assignments["duration"]].astype(float)
            event_indicator = data[target_assignments["event"]].astype(bool)
            score, _, _, _, _ = metrics_inventory[target_metric]["method"](
                event_indicator, event_time, estimate
            )
        else:
            target_col = list(target_assignments.values())[0]
            target = data[target_col]
            probabilities = model.predict(data)
            score = metrics_inventory[target_metric]["method"](target, probabilities)

        # make sure that the sign of the feature importances is correct
        if optuna_direction(target_metric) == "maximize":
            return score
        else:
            return -score

    def _get_fi_permutation(self, experiment, n_repeat, data) -> pd.DataFrame:
        """Calculate permutation feature importances."""
        # fixed confidence level
        confidence_level = 0.95
        feature_columns = experiment["feature_columns"]
        data_traindev = experiment["data_traindev"]
        data_test = experiment["data_test"]
        target_assignments = experiment["target_assignments"]
        target_metric = experiment["target_metric"]
        model = experiment["model"]

        # support prediction on new data as well as test data
        if data is None:  # new data
            data = data_test
        if not set(feature_columns).issubset(data.columns):
            raise ValueError("Features missing in provided dataset.")

        # check that targets are in dataset
        # MISSING

        # calculate baseline score
        baseline_score = self._get_performance_score(
            model, data, feature_columns, target_metric, target_assignments
        )

        # get all data select random feature values
        data_all = pd.concat([data_traindev, data], axis=0)

        results_df = pd.DataFrame(
            columns=[
                "feature",
                "importance",
                "stddev",
                "p-value",
                "n",
                "ci_low_95",
                "ci_high_95",
            ]
        )
        for feature in feature_columns:
            data_pfi = data.copy()
            fi_lst = list()

            for _ in range(n_repeat):
                # replace column with random selection from that column of data_all
                # we use data_all as the validation dataset may be small
                data_pfi[feature] = np.random.choice(
                    data_all[feature], len(data_pfi), replace=False
                )
                pfi_score = self._get_performance_score(
                    model, data_pfi, feature_columns, target_metric, target_assignments
                )
                fi_lst.append(baseline_score - pfi_score)

            # calculate statistics
            pfi_mean = np.mean(fi_lst)
            n = len(fi_lst)
            p_value = np.nan
            stddev = np.std(fi_lst, ddof=1) if n > 1 else np.nan
            if stddev not in (np.nan, 0):
                t_stat = pfi_mean / (stddev / math.sqrt(n))
                p_value = scipy.stats.t.sf(t_stat, n - 1)
            elif stddev == 0:
                p_value = 0.5

            # calculate confidence intervals
            if np.nan in (stddev, n, pfi_mean) or n == 1:
                ci_high = np.nan
                ci_low = np.nan
            else:
                t_val = scipy.stats.t.ppf(1 - (1 - confidence_level) / 2, n - 1)
                ci_high = pfi_mean + t_val * stddev / math.sqrt(n)
                ci_low = pfi_mean - t_val * stddev / math.sqrt(n)

            # save results
            results_df.loc[len(results_df)] = [
                feature,
                pfi_mean,
                stddev,
                p_value,
                n,
                ci_low,
                ci_high,
            ]

        return results_df.sort_values(by="importance", ascending=False)

    def _get_fi_shap(self, experiment, data, shap_type) -> pd.DataFrame:
        """Calculate shap feature importances."""
        experiment_id = experiment["id"]
        feature_columns = experiment["feature_columns"]
        data_test = experiment["data_test"][feature_columns]
        model = experiment["model"]
        ml_type = experiment["ml_type"]

        # support prediction on new data as well as test data
        if data is None:  # no external data, use test data
            data = data_test

        if not set(feature_columns).issubset(data.columns):
            raise ValueError("Features missing in provided dataset.")

        data = data[feature_columns]

        if ml_type == "classification":
            if shap_type == "exact":
                explainer = shap.explainers.Exact(model.predict_proba, data)
            else:
                explainer = shap.explainers.Permutation(model.predict_proba, data)
            shap_values = explainer(data)
            # only use pos class
            shap_values = shap_values[:, :, 1]  # pylint: disable=E1126
        else:
            if shap_type == "exact":
                explainer = shap.explainers.Exact(model.predict, data)
            else:
                explainer = shap.explainers.Permutation(model.predict, data)
            shap_values = explainer(data)

        results_path = self.study_path.joinpath(
            f"experiment{experiment_id}",
            f"sequence{self.sequence_item_id}",
            "results",
        )
        # create directories if needed, required for id="all"
        results_path.mkdir(parents=True, exist_ok=True)

        # (A) Bar plot
        save_path = results_path.joinpath(
            f"model_shap_fi_barplot_exp{experiment_id}_{self.sequence_item_id}.pdf",
        )
        with PdfPages(save_path) as pdf:
            plt.figure(figsize=(8.27, 11.69))  # portrait orientation (A4)
            shap.plots.bar(shap_values, show=False)
            plt.tight_layout()
            pdf.savefig(plt.gcf(), orientation="portrait")
            plt.close()

        # (B) Beeswarm plot
        save_path = results_path.joinpath(
            f"model_shap_fi_beeswarm_exp{experiment_id}_{self.sequence_item_id}.pdf",
        )
        with PdfPages(save_path) as pdf:
            plt.figure(figsize=(8.27, 11.69))  # portrait orientation (A4)
            shap.plots.beeswarm(shap_values, max_display=20, show=False)
            plt.tight_layout()
            pdf.savefig(plt.gcf(), orientation="portrait")
            plt.close()

        # (C) save fi to table
        shap_fi_df = pd.DataFrame(shap_values.values, columns=data.columns)
        shap_fi_df = shap_fi_df.abs().mean().to_frame().reset_index()
        shap_fi_df.columns = ["feature", "importance"]
        shap_fi_df = shap_fi_df.sort_values(
            by="importance", ascending=False
        ).reset_index(drop=True)

        return shap_fi_df
