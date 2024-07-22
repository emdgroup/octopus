"""OctoFull Bags."""

# import concurrent.futures
# import logging
import gzip
import pickle
from statistics import mean

import numpy as np
import pandas as pd
from attrs import define, field, validators
from joblib import Parallel, delayed

from octopus.modules.metrics import metrics_inventory
from octopus.modules.octo.scores import add_pooling_scores

# logging.basicConfig(
#    filename="logging_bag.log",
#    level=logging.INFO,
#    format="%(asctime)s:%(levelname)s:%(message)s",
# )


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
    n_features_used_mean: float = field(
        init=False, validator=[validators.instance_of(float)]
    )

    def __attrs_post_init__(self):
        # initialization here due to "Python immutable default"
        self.feature_importances = dict()
        self.n_features_used_mean = 0.0

    def fit(self):
        """Run all available trainings."""
        if self.parallel_execution is True:
            # (A) joblib based code that works xgboost, solves issue 46
            # Function to execute each training
            def execute_training(training, idx):
                try:
                    result = training.fit()
                    print(f"Training {idx} completed successfully.")
                    # logging.info(f"Training {idx} completed successfully.")
                    return result
                except Exception as e:  # pylint: disable=broad-except
                    print(f"Exception during training{idx}: {e}")
                    print(f"Exception type: {type(e).__name__}")
                    # logging.error(
                    #    f"Training {idx} failed or returned None. Exception: {e}",
                    #    exc_info=True,
                    # )
                    return None

            # Using joblib's Parallel and delayed functionalities
            # default backend is 'loky'
            with Parallel(n_jobs=self.num_workers) as parallel:
                self.trainings = parallel(
                    delayed(execute_training)(training, idx)
                    for idx, training in enumerate(self.trainings)
                )

            # (B) altern. ProcessPoolExecutor code, incompatible with xgboost, issue46
            ## max_tasks_per_child=1 requires Python3.11
            # with concurrent.futures.ProcessPoolExecutor(
            #    max_workers=self.num_workers
            # ) as executor:
            #    # Map each training to a future and keep an index to retain order
            #    futures = {
            #        executor.submit(training.fit): i
            #        for i, training in enumerate(self.trainings)
            #    }
            #    train_results = [None] * len(
            #        self.trainings
            #    )  # List to hold results in order
            #
            #    # Process completed training tasks as they complete
            #    for future in concurrent.futures.as_completed(futures):
            #        index = futures[future]
            #        try:
            #            result = future.result()
            #            train_results[index] = result
            #            print(f"Training {index} completed successfully.")
            #            logging.info(f"Training {index} completed successfully.")
            #        except Exception as e:
            #            print(f"Exception during training {index}: {e}")
            #            print(f"Exception type: {type(e).__name__}")
            #            logging.error(
            #                f"Exception during training {index}: {str(e)}",
            #                exc_info=True,
            #            )
            #            train_results[index] = None
            # Update trainings with the results
            # self.trainings=[result for result in train_results if result is not None]

        else:
            # Running training sequentially in the current process
            for training in self.trainings:
                try:
                    training.fit()
                    print("Inner sequential training completed")
                except Exception as e:  # pylint: disable=broad-except
                    print(
                        f"Error during training {training}: {e},"
                        f" type: {type(e).__name__}"
                    )

        self.train_status = (True,)

        # get used features in bag
        n_feat_lst = list()
        for training in self.trainings:
            n_feat_lst.append(float(len(training.features_used)))
        self.n_features_used_mean = mean(n_feat_lst)

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
        ensemble = pool.groupby(by=self.row_column).mean().reset_index()

        # set correct dtype for target columns
        for column in list(self.target_assignments.values()):
            ensemble[column] = ensemble[column].astype(
                self.trainings[0].data_train[column].dtype
            )

        predictions["ensemble"] = {"test": ensemble}

        return predictions

    def get_scores(self):
        """Get scores."""
        if not self.train_status:
            print("Running trainings first to be able to get scores")
            self.fit()

        scores = dict()
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
                        metrics_inventory[self.target_metric]["method"](
                            target, probabilities
                        )
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
                    ci, _, _, _, _ = metrics_inventory[self.target_metric]["method"](
                        event_indicator, event_time, estimate
                    )
                    storage[part].append(float(ci))
            else:
                for part in storage.keys():
                    target_col = list(self.target_assignments.values())[0]
                    predictions = training.predictions[part]["prediction"]
                    target = training.predictions[part][target_col]
                    storage[part].append(
                        metrics_inventory[self.target_metric]["method"](
                            target, predictions
                        )
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
        add_pooling_scores(pool, scores, self.target_metric, self.target_assignments)

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
            elif fi_type == "lofo":
                training.calculate_fi_lofo()
            else:
                raise ValueError("FI type not supported")

    def get_selected_features(self, fi_methods=None):
        """Get features selected by model, depending on fi method.

        The list of selected features will be derived only from one feature
        importance method out of the ones specified in fi_methods,
        with the following ranking: (1) permutation (2) shap (3) internal.
        """
        if fi_methods is None:
            fi_methods = []

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

    def get_feature_importances(self, fi_methods=None):
        """Extract feature importances of all models in bag."""
        # we always extract internal feature importances, if available
        if fi_methods is None:
            fi_methods = []

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
            elif method == "lofo":
                self._calculate_fi(fi_type="lofo")
            else:
                raise ValueError(f"Feature importance method {method} not supported.")

        # save feature importances for every training in bag
        for training in self.trainings:
            self.feature_importances[training.training_id] = (
                training.feature_importances
            )

        # summary feature importances for all trainings (mean + count)
        # internal, permutation_dev, shap_dev only
        # save in bag
        for method in fi_methods:
            if method == "internal":
                method_str = "internal"
            else:
                method_str = method + "_dev"
            fi_pool = list()
            for training in self.trainings:
                fi_pool.append(training.feature_importances[method_str])
            # calculate mean and count feature importances
            fi = pd.concat(fi_pool, axis=0)
            self.feature_importances[method_str + "_mean"] = (
                fi[["feature", "importance"]]
                .groupby(by="feature")
                .mean()
                .sort_values(by="importance", ascending=False)
                .reset_index()
            )
            self.feature_importances[method_str + "_count"] = (
                fi[["feature", "importance"]]
                .groupby(by="feature")
                .count()
                .sort_values(by="importance", ascending=False)
                .reset_index()
            )

        return self.feature_importances

    def predict(self, x):
        """Predict."""
        preds_lst = list()
        weights_lst = list()
        for training in self.trainings:
            train_w = training.training_weight
            weights_lst.append(train_w)
            preds_lst.append(train_w * training.predict(x))

        # return mean of weighted predictions
        return np.sum(np.array(preds_lst), axis=0) / sum(weights_lst)

    def predict_proba(self, x):
        """Predict_proba."""
        preds_lst = list()
        weights_lst = list()
        for training in self.trainings:
            train_w = training.training_weight
            weights_lst.append(train_w)
            preds_lst.append(train_w * training.predict_proba(x))

        # return mean of weighted predictions
        return np.sum(np.array(preds_lst), axis=0) / sum(weights_lst)

    def to_pickle(self, file_path: str) -> None:
        """Save object to a compressed pickle file.

        Args:
            file_path: The name of the file to save the pickle data to.
        """
        with gzip.GzipFile(file_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, file_path: str) -> "Bag":
        """Load object to a compressed pickle file.

        Args:
            file_path: The path to the file to load the pickle data from.

        Returns:
            Bag: The loaded instance of Bag.
        """
        with gzip.GzipFile(file_path, "rb") as file:
            return pickle.load(file)
