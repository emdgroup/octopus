"""OctoFull Bags."""

import concurrent.futures
import pickle
from statistics import mean

import pandas as pd
from attrs import define, field, validators
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sksurv.metrics import concordance_index_censored


@define
class Bag:
    """Container for Trainings.

    Supports:
    - execution of trainings, sequential/parallel
    - saving/loading
    """

    bag_id: str = field(validator=[validators.instance_of(str)])
    trainings: list = field(validator=[validators.instance_of(list)])
    # same config parameters (execution type, num_workers) also used for
    # parallelization of optuna optimizations of individual inner loop trainings
    parallel_execution: bool = field(validator=[validators.instance_of(bool)])
    num_workers: int = field(validator=[validators.instance_of(int)])
    target_metric: str = field(validator=[validators.instance_of(str)])
    target_assignments: dict = field(validator=[validators.instance_of(dict)])
    row_column: str = field(validator=[validators.instance_of(str)])
    train_status: bool = field(default=False)

    # bag training outputs, initialized in post_init
    feature_importances: dict = field(
        init=False, validator=[validators.instance_of(dict)]
    )
    features_used: list = field(init=False, validator=[validators.instance_of(list)])

    def __attrs_post_init__(self):
        # initialization here due to "Python immutable default"
        self.feature_importances = dict()
        self.features_used = list()

    def fit(self):
        """Run all available trainings."""
        if self.parallel_execution is True:
            # max_tasks_per_child=1 requires Python3.11
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers,
            ) as executor:
                futures = []
                train_results = []
                for i in self.trainings:
                    try:
                        future = executor.submit(i.fit)
                        futures.append(future)
                    except Exception as e:  # pylint: disable=broad-except
                        print(f"Exception occurred while submitting task: {e}")
                for future in concurrent.futures.as_completed(futures):
                    try:
                        train_results.append(future.result())
                        print("Inner parallel training completed")
                    except Exception as e:  # pylint: disable=broad-except
                        print(f"Exception occurred while executing task: {e}")
                        print(f"Exception: {type(e).__name__}")
                # replace trainings with processed trainings
                # order in self.trainings may change!
                self.trainings = train_results

        else:
            for training in self.trainings:
                training.fit()
                print("Inner sequential training completed")

        self.train_status = (True,)

        # get used features in bag
        feat_lst = list()
        for training in self.trainings:
            feat_lst.extend(training.features_used)
        self.features_used = list(set(feat_lst))

    def get_predictions(self):
        """Extract bag test predictions."""
        if not self.train_status:
            print("Running trainings first to be able to get scores")
            self.fit()

        predictions = dict()
        pool = []
        for training in self.trainings:
            # collect all predictions (train/dev/test) from training
            predictions[training.training_id] = training.predictions
            # pool predictions for ensembling
            pool.append(training.predictions["test"])
        pool = pd.concat(pool, axis=0)
        ensemble = pool.groupby(by=self.row_column).mean()

        if self.target_metric in ["AUCROC", "LOGLOSS"]:
            ensemble["probability"] = ensemble[1]  # binary only!!
        predictions["ensemble"] = {"test": ensemble}

        return predictions

    def get_scores(self):
        """Get scores."""
        if not self.train_status:
            print("Running trainings first to be able to get scores")
            self.fit()

        scores = dict()
        metrics_inventory = {
            "AUCROC": roc_auc_score,
            "ACC": accuracy_score,
            "ACCBAL": balanced_accuracy_score,
            "LOGLOSS": log_loss,
            "MAE": mean_absolute_error,
            "MSE": mean_squared_error,
            "R2": r2_score,
            "CI": concordance_index_censored,
        }

        storage = {key: [] for key in ["train", "dev", "test"]}
        pool = {key: [] for key in ["train", "dev", "test"]}

        for training in self.trainings:
            # averaging
            if self.target_metric in ["AUCROC", "LOGLOSS"]:
                for part in storage.keys():
                    target_col = list(self.target_assignments.values())[0]
                    probabilities = training.predictions[part][1]  # binary only!!
                    target = training.predictions[part][target_col]
                    storage[part].append(
                        metrics_inventory[self.target_metric](target, probabilities)
                    )
            elif self.target_metric in ["CI"]:
                for part in storage.keys():
                    estimate = training.predictions[part]["prediction"]
                    event_time = training.predictions[part][
                        self.target_assignments["duration"]
                    ].astype(float)
                    event_indicator = training.predictions[part][
                        self.target_assignments["event"]
                    ].astype(bool)
                    ci, _, _, _, _ = metrics_inventory[self.target_metric](
                        event_indicator, event_time, estimate
                    )
                    storage[part].append(float(ci))
            else:
                for part in storage.keys():
                    target_col = list(self.target_assignments.values())[0]
                    predictions = training.predictions[part]["prediction"]
                    target = training.predictions[part][target_col]
                    storage[part].append(
                        metrics_inventory[self.target_metric](target, predictions)
                    )
            # pooling
            for part in pool.keys():
                pool[part].append(training.predictions[part])

        # calculate averaging scores
        scores["train_avg"] = mean(storage["train"])
        scores["train_lst"] = storage["train"]
        scores["dev_avg"] = mean(storage["dev"])
        scores["dev_lst"] = storage["dev"]
        scores["test_avg"] = mean(storage["test"])
        scores["test_lst"] = storage["test"]
        # stack pooled data and groupby
        for part in pool.keys():
            pool[part] = pd.concat(pool[part], axis=0)
            pool[part] = pool[part].groupby(by=self.row_column).mean()
        # calculate pooling scores (soft and hard)
        if self.target_metric in ["AUCROC", "LOGLOSS"]:
            for part in pool.keys():
                target_col = list(self.target_assignments.values())[0]
                probabilities = pool[part][1]  # binary only!!
                predictions = pool[part]["prediction"]
                target = pool[part][target_col]
                scores[part + "_pool_soft"] = metrics_inventory[self.target_metric](
                    target, probabilities
                )
                scores[part + "_pool_hard"] = metrics_inventory[self.target_metric](
                    target, predictions
                )
        elif self.target_metric in ["CI"]:
            for part in pool.keys():
                estimate = pool[part]["prediction"]
                event_time = pool[part][self.target_assignments["duration"]].astype(
                    float
                )
                event_indicator = pool[part][self.target_assignments["event"]].astype(
                    bool
                )
                ci, _, _, _, _ = metrics_inventory[self.target_metric](
                    event_indicator, event_time, estimate
                )
                scores[part + "_pool_hard"] = float(ci)
        else:
            for part in pool.keys():
                target_col = list(self.target_assignments.values())[0]
                predictions = pool[part]["prediction"]
                target = pool[part][target_col]
                scores[part + "_pool_hard"] = metrics_inventory[self.target_metric](
                    target, predictions
                )

        return scores

    def _calculate_fi(self, fi_type="internal", partition="dev"):
        """Calculate feature importance."""
        for training in self.trainings:
            if fi_type == "internal":
                training.calculate_fi_internal()
            elif fi_type == "shap":
                training.calculate_fi_shap(partition=partition)
            elif fi_type == "permutation":
                training.calculate_fi_permutation(partition=partition)
            else:
                raise ValueError("FI type not supported")

    def get_selected_features(self, fi_methods=[]):
        """Get features selected by model, depending on fi method.

        The list of selected features will be derived only from one feature
        importance method out of the ones specified in fi_methods,
        with the following ranking: (1) permutation (2) shap (3) internal.
        """
        feat_lst = list()
        if "permutation" in fi_methods:
            for training in self.trainings:
                fi_df = training.feature_importances["permutation_dev"]
                feat_lst.extend(fi_df[fi_df["importance"] != 0]["feature"].tolist())
            return list(set(feat_lst))
        elif "shap" in fi_methods:
            for training in self.trainings:
                fi_df = training.feature_importances["shap_dev"]
                feat_lst.extend(fi_df[fi_df["importance"] != 0]["feature"].tolist())
            return list(set(feat_lst))
        elif "internal" in fi_methods:
            for training in self.trainings:
                fi_df = training.feature_importances["internal"]
                feat_lst.extend(fi_df[fi_df["importance"] != 0]["feature"].tolist())
            return list(set(feat_lst))
        else:
            print("No features importances calculated")
            return []

    def get_feature_importances(self, fi_methods=[]):
        """Extract feature importances of all models in bag."""
        # we always extract internal feature importances, if available
        self._calculate_fi(fi_type="internal")

        for method in fi_methods:
            if method == "internal":
                pass  # already done
            elif method == "shap":
                self._calculate_fi(fi_type="shap", partition="dev")
                self._calculate_fi(fi_type="shap", partition="test")
            elif method == "permutation":
                self._calculate_fi(fi_type="permutation", partition="dev")
                self._calculate_fi(fi_type="permutation", partition="test")
            else:
                raise ValueError(f"Feature importance method {method} not supported.")

        for training in self.trainings:
            self.feature_importances[
                training.training_id
            ] = training.feature_importances
        return self.feature_importances

    def to_pickle(self, path):
        """Save Bag using pickle."""
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, path):
        """Load Bag from pickle file."""
        with open(path, "rb") as file:
            return pickle.load(file)
