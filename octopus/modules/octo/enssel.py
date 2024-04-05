"""Ensemble selection."""

# TOBEDONE
# - get FI and counts
# - display results summary
# - save results to experiment
# - create finale ensemble_bag containing all the models
# - ensemble models needs to provide finale predictions (dev, test)


# import copy
from pathlib import Path

import pandas as pd
from attrs import define, field, validators

from octopus.modules.metrics import optuna_direction
from octopus.modules.octo.bag import Bag
from octopus.modules.octo.scores import add_pooling_scores


@define
class EnSel:
    """Ensemble Selection."""

    target_metric: str = field(validator=[validators.instance_of(str)])
    target_assignments: dict = field(validator=[validators.instance_of(dict)])
    path_trials: Path = field(validator=[validators.instance_of(Path)])
    max_n_best_models: int = field(validator=[validators.instance_of(int)])
    max_n_iterations: int = field(validator=[validators.instance_of(int)])
    row_column: str = field(validator=[validators.instance_of(str)])
    model_table: pd.DataFrame = field(
        init=False,
        default=pd.DataFrame(),
        validator=[validators.instance_of(pd.DataFrame)],
    )
    scan_table: pd.DataFrame = field(
        init=False,
        default=pd.DataFrame(),
        validator=[validators.instance_of(pd.DataFrame)],
    )

    bags: dict = field(init=False, validator=[validators.instance_of(dict)])

    def __attrs_post_init__(self):
        # initialization here due to "Python immutable default"
        self.bags = dict()
        # get a trials existing in path_trials
        self._collect_trials()
        self._create_model_table()
        self._ensemble_scan()
        self._ensemble_optimization()

    def _collect_trials(self):
        """Get all trials save in path_trials and store properties in self.bags."""
        # Get all .pkl files in the directory
        pkl_files = [
            file
            for file in self.path_trials.iterdir()
            if file.is_file() and file.suffix == ".pkl"
        ]

        # fill bags dict
        for file in pkl_files:
            bag = Bag.from_pickle(file)
            self.bags[file] = {
                "id": bag.bag_id,
                "scores": bag.get_scores(),
                "predictions": bag.get_predictions(),
                "n_features_used_mean": bag.n_features_used_mean,
            }

    def _create_model_table(self):
        """Create model table."""
        df_lst = list()
        for key, value in self.bags.items():
            s = pd.Series()
            s["id"] = value["id"]
            s["dev_avg"] = value["scores"]["dev_avg"]  # only average is necessary
            s["test_avg"] = value["scores"]["test_avg"]
            s["test_pool_hard"] = value["scores"]["test_pool_hard"]
            s["n_features_used_mean"] = value["n_features_used_mean"]
            s["path"] = key
            if self.target_metric in ["AUCROC", "LOGLOSS"]:
                s["test_pool_soft"] = value["scores"]["test_pool_hard"]
            df_lst.append(s)

        if optuna_direction(self.target_metric) == "maximize":
            ascending = False
        else:
            ascending = True

        self.model_table = (
            pd.concat(df_lst, axis=1)
            .T.sort_values(by="dev_avg", ascending=ascending)
            .reset_index(drop=True)
        )

    def _ensemble_models(self, bags):
        """Esemble using all bags and their corresponding models provided by input."""
        # collect all predictions over inner folds and bags
        scores = dict()
        pool = {key: [] for key in ["dev", "test"]}

        for bag in bags:
            predictions = bag["predictions"]
            # remove 'ensemble'
            predictions.pop("ensemble", 0)
            # concatenate and averag dev and test predictions from inner models
            for pred in predictions.values():
                for part in pool.keys():
                    pool[part].append(pred[part])
        # average all predictions (inner models, bags)
        for part in pool.keys():
            pool[part] = (
                pd.concat(pool[part], axis=0).groupby(by=self.row_column).mean()
            )

        # calculate pooling scores (soft and hard)
        add_pooling_scores(pool, scores, self.target_metric, self.target_assignments)

        return scores

    def _ensemble_scan(self):
        """Scan for highest performing ensemble consisting of best N bags."""
        if self.target_metric in ["AUCROC", "LOGLOSS"]:
            self.scan_table = pd.DataFrame(
                columns=[
                    "#models",
                    "score_dev_hard",
                    "score_test_hard",
                    "score_dev_soft",
                    "score_test_soft",
                ]
            )
        else:
            self.scan_table = pd.DataFrame(
                columns=["#models", "score_dev_hard", "score_test_hard"]
            )

        for i in range(1, len(self.model_table)):
            bag_keys = self.model_table[:i]["path"].tolist()
            list_of_bags = [self.bags[key] for key in bag_keys]

            scores = self._ensemble_models(list_of_bags)
            if self.target_metric in ["AUCROC", "LOGLOSS"]:
                self.scan_table.loc[i] = [
                    i,
                    scores["dev_pool_hard"],
                    scores["test_pool_hard"],
                    scores["dev_pool_soft"],
                    scores["test_pool_soft"],
                ]
            else:
                self.scan_table.loc[i] = [
                    i,
                    scores["dev_pool_hard"],
                    scores["test_pool_hard"],
                ]

    def _ensemble_optimization(self):
        """Ensembling optimization with replacement."""
        # we start with an best N models exammple derived from self.scan_table
        if optuna_direction(self.target_metric) == "maximize":
            start_n = int(self.scan_table["score_dev_hard"].idxmax())
        else:
            start_n = int(self.scan_table["score_dev_hard"].idxmin())
        print(start_n)

    def get_ens_input(self):
        """Get ensemble dict."""
        return self.bags
