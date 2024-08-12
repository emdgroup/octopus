"""Ensemble selection."""

# TOBEDONE
# - get FI and counts
# - display results summary
# - save results to experiment
# - create finale ensemble_bag containing all the models
# - ensemble models needs to provide finale predictions (dev, test)

import copy
from collections import Counter
from pathlib import Path

import pandas as pd
from attrs import define, field, validators

from octopus.modules.octo.bag import Bag
from octopus.modules.octo.scores import add_pooling_scores
from octopus.modules.utils import optuna_direction


@define
class EnSel:
    """Ensemble Selection."""

    target_metric: str = field(validator=[validators.instance_of(str)])
    target_assignments: dict = field(validator=[validators.instance_of(dict)])
    path_trials: Path = field(validator=[validators.instance_of(Path)])
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

    @property
    def direction(self) -> str:
        """Optuna direction."""
        return optuna_direction(self.target_metric)

    @property
    def score_type(self) -> str:
        """Score type."""
        if self.target_metric in ["AUCROC", "LOGLOSS"]:
            return "dev_pool_soft"
        else:
            return "dev_pool_hard"

    bags: dict = field(init=False, validator=[validators.instance_of(dict)])
    optimized_ensemble = dict = field(
        init=False, validator=[validators.instance_of(dict)]
    )

    def __attrs_post_init__(self):
        # initialization here due to "Python immutable default"
        self.bags = dict()
        self.optimized_ensemble = dict()
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
            s["dev_pool_hard"] = value["scores"]["dev_pool_hard"]  # relevant
            if self.score_type == "dev_pool_soft":
                s["dev_pool_soft"] = value["scores"]["dev_pool_soft"]  # relevant
                s["test_pool_soft"] = value["scores"]["test_pool_soft"]
            s["test_pool_hard"] = value["scores"]["test_pool_hard"]
            s["dev_avg"] = value["scores"]["dev_avg"]
            s["test_avg"] = value["scores"]["test_avg"]
            s["n_features_used_mean"] = value["n_features_used_mean"]
            s["path"] = key
            df_lst.append(s)

        self.model_table = pd.concat(df_lst, axis=1).T

        # oder of table is important, depending on metric,
        # (a) direction (b) dev_pool_soft or dev_pool_hard
        if self.direction == "maximize":
            ascending = False
        else:
            ascending = True

        self.model_table = self.model_table.sort_values(
            by=self.score_type, ascending=ascending
        ).reset_index(drop=True)

    def _ensemble_models(self, bag_keys):
        """Esemble using all bags and their corresponding models provided by input."""
        # collect all predictions over inner folds and bags
        scores = dict()
        pool = {key: [] for key in ["dev", "test"]}

        for key in bag_keys:
            predictions = self.bags[key]["predictions"]
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
        if self.score_type == "dev_pool_soft":
            self.scan_table = pd.DataFrame(
                columns=[
                    "#models",
                    "dev_pool_hard",
                    "test_pool_hard",
                    "dev_pool_soft",
                    "test_pool_soft",
                ]
            )
        else:
            self.scan_table = pd.DataFrame(
                columns=["#models", "dev_pool_hard", "test_pool_hard"]
            )

        for i in range(1, len(self.model_table)):
            bag_keys = self.model_table[:i]["path"].tolist()
            scores = self._ensemble_models(bag_keys)
            if self.score_type == "dev_pool_soft":
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
        # we start with an best N models exammple derived from self.scan_table,
        # assuming that is sorted correctly
        if self.direction == "maximize":
            start_n = int(self.scan_table[self.score_type].idxmax())
        else:
            start_n = int(self.scan_table[self.score_type].idxmin())
        print("Ensemble scan, number of included best models: ", start_n)

        # startn_bags dict with path as key and repeats=1 as value
        escan_ensemble = dict()
        for _, row in self.model_table.head(start_n).iterrows():
            escan_ensemble[row["path"]] = 1

        # ensemble_optimmization, reference score
        # we start with the bags found in ensemble scan
        results_df = pd.DataFrame(columns=["model", "performance", "bags_lst"])
        start_bags = list(escan_ensemble.keys())
        scores = self._ensemble_models(start_bags)
        start_perf = scores[self.score_type]
        print("Ensemble optimization")
        print("Start performance:", start_perf)
        # record start performance
        results_df.loc[len(results_df)] = [
            ["ensemble scan"],
            start_perf,
            copy.deepcopy(start_bags),
        ]

        # optimization
        bags_ensemble = copy.deepcopy(start_bags)
        best_global = copy.deepcopy(start_perf)

        for i in range(self.max_n_iterations):
            df = pd.DataFrame(columns=["model", "performance"])
            # find additional model
            for model in self.model_table["path"].tolist():
                bags_lst = copy.deepcopy(bags_ensemble)
                bags_lst.append(model)
                scores = self._ensemble_models(bags_lst)
                df.loc[len(df)] = [model, scores[self.score_type]]

            if self.direction == "maximize":
                best_model = df.loc[df["performance"].idxmax()]["model"]
                best_performance = df.loc[df["performance"].idxmax()]["performance"]
                if best_performance < best_global:
                    break  # stop ensembling
                else:
                    best_global = best_performance
                    print(f"iteration: {i}, performance: {best_performance}")
            else:  # minimize
                best_model = df.loc[df["performance"].idxmin()]["model"]
                best_performance = df.loc[df["performance"].idxmin()]["performance"]
                if best_performance > best_global:
                    break  # stop ensembling
                else:
                    best_global = best_performance
                    print(f"iteration: {i}, performance: {best_performance}")

            # add best model to ensemble
            bags_ensemble.append(best_model)

            # record results
            results_df.loc[len(results_df)] = [
                best_model,
                best_performance,
                copy.deepcopy(bags_ensemble),
            ]

        # store optimization results
        self.optimized_ensemble = dict(Counter(results_df.iloc[-1]["bags_lst"]))
        print("Ensemble selection completed.")

    def get_ens_input(self):
        """Get ensemble dict."""
        return self.bags
