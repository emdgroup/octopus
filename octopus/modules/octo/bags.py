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


@define
class TrainingsBag:
    """Container for Trainings.

    Supports:
    - execution of trainings, sequential/parallel
    - saving/loading
    """

    trainings: list = field(validator=[validators.instance_of(list)])
    # same config parameters (execution type, num_workers) also used for
    # parallelization of optuna optimizations of individual inner loop trainings
    parallel_execution: bool = field(validator=[validators.instance_of(bool)])
    num_workers: int = field(validator=[validators.instance_of(int)])
    target_metric: str = field(validator=[validators.instance_of(str)])
    row_column: str = field(validator=[validators.instance_of(str)])
    # path: Path = field(default=Path(), validator=[validators.instance_of(Path)])
    feature_importances: dict = field(
        default=dict(), validator=[validators.instance_of(dict)]
    )
    train_status: bool = field(default=False)

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

        self.train_status = True

    def get_test_predictions(self):
        """Extract bag test predictions."""
        if not self.train_status:
            print("Running trainings first to be able to get scores")
            self.fit()

        pool = []
        for training in self.trainings:
            pool.append(training.predictions["test"])
        pool = pd.concat(pool, axis=0)
        ensemble = pool.groupby(by=self.row_column).mean()

        if self.target_metric in ["AUCROC", "LOGLOSS"]:
            ensemble["probability"] = ensemble[1]  # binary only!!

        return ensemble

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
        }

        storage = {key: [] for key in ["train", "dev", "test"]}
        pool = {key: [] for key in ["train", "dev", "test"]}

        for training in self.trainings:
            # averaging
            if self.target_metric in ["AUCROC", "LOGLOSS"]:
                for part in storage.keys():
                    probabilities = training.predictions[part][1]  # binary only!!
                    target = training.predictions[part]["target"]
                    storage[part].append(
                        metrics_inventory[self.target_metric](target, probabilities)
                    )
            else:
                for part in storage.keys():
                    predictions = training.predictions[part]["prediction"]
                    target = training.predictions[part]["target"]
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
                probabilities = pool[part][1]  # binary only!!
                predictions = pool[part]["prediction"]
                target = pool[part]["target"]
                scores[part + "_pool_soft"] = metrics_inventory[self.target_metric](
                    target, probabilities
                )
                scores[part + "_pool_hard"] = metrics_inventory[self.target_metric](
                    target, predictions
                )
        else:
            for part in pool.keys():
                predictions = pool[part]["prediction"]
                target = pool[part]["target"]
                scores[part + "_pool_hard"] = metrics_inventory[self.target_metric](
                    target, predictions
                )

        return scores

    def calculate_fi_internal(self):
        """Feature importances, model internal."""
        fi = dict()
        for training in self.trainings:
            training.calculate_fi_internal()
            fi[training.training_id] = training.feature_importances["internal"]

        self.feature_importances["internal"] = fi

    def calculate_fi_permutation(self):
        """Feature importances, permutation."""
        fi = dict()
        for training in self.trainings:
            training.calculate_fi_permutation()
            fi[training.training_id] = training.feature_importances["permutation"]
        self.feature_importances["permutation"] = fi

    def calculate_fi_shap(self):
        """Feature importances, shap."""
        fi = dict()
        for training in self.trainings:
            training.calculate_fi_shap()
            fi[training.training_id] = training.feature_importances["shap"]
        self.feature_importances["shap"] = fi

    def get_selected_features(self):
        """Extract selected bag features, based on shap."""
        self.calculate_fi_shap()
        df_lst = list()
        for _, value in self.feature_importances["shap"].items():
            df_lst.append(value)
        fi_df = pd.concat(df_lst, axis=0)
        fi_df = fi_df[fi_df["importance"] != 0]
        return fi_df["feature"].unique().tolist()

    def get_feature_importances(self):
        """Extract bag feature importances."""
        # - add shap feature importance
        self.calculate_fi_internal()
        # self.calculate_fi_permutation()
        self.calculate_fi_shap()
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
