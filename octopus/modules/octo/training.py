"""Octo Training."""

import copy
import math

import numpy as np
import pandas as pd
import scipy
import shap
from attrs import Factory, define, field, validators
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MaxAbsScaler

from octopus.models.inventory import ModelInventory
from octopus.modules.utils import get_performance_score

scorer_string_inventory = {
    "AUCROC": "roc_auc",
    "ACC": "accuracy",
    "ACCBAL": "balanced_accuracy",
    "LOGLOSS": "neg_log_loss",
    "MAE": "neg_mean_absolute_error",
    "MSE": "neg_mean_squared_error",
    "R2": "r2",
}


@define
class Training:
    """Model Training Class."""

    training_id: str = field(validator=[validators.instance_of(str)])
    """Training id."""

    ml_type: str = field(validator=[validators.instance_of(str)])
    """ML-type."""

    target_assignments: dict = field(validator=[validators.instance_of(dict)])
    """Target assignments."""

    feature_columns: list = field(validator=[validators.instance_of(list)])
    """Feature columns."""

    row_column: str = field(validator=[validators.instance_of(str)])
    """Row column."""

    data_train: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    "Data train."

    data_dev: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    "Data dev."

    data_test: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    """Data test."""

    target_metric: str = field(validator=[validators.instance_of(str)])
    """Target metric."""

    max_features: int = field(validator=[validators.instance_of(int)])
    """Maximum number of features."""

    feature_groups: dict = field(validator=[validators.instance_of(dict)])
    """Feature Groups."""

    config_training: dict = field(validator=[validators.instance_of(dict)])
    """Training configuration."""

    training_weight: int = field(default=1, validator=[validators.instance_of(int)])
    """Training weight for ensembling"""

    model = field(default=None)
    """Model."""

    predictions: dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Model predictions."""

    feature_importances: dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Feature importances."""

    features_used: list = field(
        default=Factory(list), validator=[validators.instance_of(list)]
    )
    """Features used."""

    outlier_samples: list = field(
        default=Factory(list), validator=[validators.instance_of(list)]
    )
    """Outlie samples identified."""

    scaler = field(init=False)
    """Scaler."""

    @property
    def dim_reduction(self) -> str:
        """Dimension reduction method."""
        return self.config_training["dim_reduction"]

    @property
    def outl_reduction(self) -> int:
        """Parameter outlier reduction method."""
        return self.config_training["outl_reduction"]

    @property
    def ml_model_type(self) -> str:
        """Dimension reduction method."""
        return self.config_training["ml_model_type"]

    @property
    def ml_model_params(self) -> dict:
        """Dimension reduction method."""
        return self.config_training["ml_model_params"]

    @property
    def x_train(self):
        """x_train."""
        return self.data_train[self.feature_columns]

    @property
    def x_dev(self):
        """x_dev."""
        return self.data_dev[self.feature_columns]

    @property
    def x_test(self):
        """x_test."""
        return self.data_test[self.feature_columns]

    @property
    def y_train(self):
        """y_train."""
        if self.ml_type == "timetoevent":
            duration = self.data_train[self.target_assignments["duration"]]
            event = self.data_train[self.target_assignments["event"]]
            return np.array(
                list(zip(event, duration)),
                dtype={"names": ("c1", "c2"), "formats": ("bool", "f8")},
            )
        else:
            return self.data_train[self.target_assignments.values()]

    @property
    def y_dev(self):
        """y_dev."""
        if self.ml_type == "timetoevent":
            duration = self.data_dev[self.target_assignments["duration"]]
            event = self.data_dev[self.target_assignments["event"]]
            return np.array(
                list(zip(event, duration)),
                dtype={"names": ("c1", "c2"), "formats": ("bool", "f8")},
            )
        else:
            return self.data_dev[self.target_assignments.values()]

    @property
    def y_test(self):
        """y_dev."""
        if self.ml_type == "timetoevent":
            duration = self.data_test[self.target_assignments["duration"]]
            event = self.data_test[self.target_assignments["event"]]
            return np.array(
                list(zip(event, duration)),
                dtype={"names": ("c1", "c2"), "formats": ("bool", "f8")},
            )
        else:
            return self.data_test[self.target_assignments.values()]

    def __attrs_post_init__(self):
        # scaler
        self.scaler = MaxAbsScaler()

    # Training class functionality:
    # (1) outlier removal
    # (2) scaling
    # (3) dim_reduction
    # (4) model training
    # (5) model predictions
    # (6) calculate feature importance, on request

    def fit(self):
        """Preprocess and fit model."""
        # missing:
        # (1) scaling
        # (2) missing dim reduction

        # use copy of all train variables, as they may be change due to outlier detec.
        data_train = self.data_train.copy()
        x_train = self.x_train.copy()
        y_train = self.y_train.copy()

        # (1) outlier removal in x_train
        if self.outl_reduction > 0:
            # IsolationForest for outlier detection
            clf = IsolationForest(
                contamination=self.outl_reduction / len(x_train),
                random_state=42,
                n_jobs=1,
            )
            clf.fit(x_train)

            # Get the outlier prediction labels
            # (-1:outliers, 1:inliers)
            outlier_pred = clf.predict(x_train)
            # sometimes there seems to be a mismatch in the number of outliers
            # assert self.outl_reduction == np.sum(outlier_pred == -1)
            # print("Number of outliers specified:", self.outl_reduction)
            # print("Number of outliers found:", np.sum(outlier_pred == -1))

            # identify outlier samples
            self.outlier_samples = data_train[outlier_pred == -1][
                self.row_column
            ].tolist()
            # print("Outlier samples:", self.outlier_samples)

            # Remove outliers from data_train, x_train, y_train
            data_train = data_train[outlier_pred == 1].copy()
            x_train = x_train[outlier_pred == 1].copy()
            y_train = y_train[outlier_pred == 1].copy()

        # (2) Scaling - missing
        # scaling (!after outlier removal)
        # x_train_scaled = self.scaler.fit_transform(x_train)
        # x_dev_scaled = self.scaler.transform(self.x_dev)
        # x_test_scaled = self.scaler.transform(self.x_test)

        # (3) Dimensionality reduction - missing

        # (4) Model training
        # define ML-model
        # self.model = model_inventory[self.ml_model_type]["model"](
        #     **self.ml_model_params
        # )

        self.model = ModelInventory().get_model_instance(
            self.ml_model_type, self.ml_model_params
        )

        if len(self.target_assignments) == 1:
            # standard sklearn single target models
            self.model.fit(
                # x_train_scaled,
                x_train,
                y_train.squeeze(axis=1),
            )
        else:
            # multi target models, incl. time2event
            # self.model.fit(x_train_scaled, y_train)
            self.model.fit(x_train, y_train)

        # (5) Model prediction
        self.predictions["train"] = pd.DataFrame()
        self.predictions["train"][self.row_column] = data_train[self.row_column]
        # self.predictions["train"]["prediction"] = self.model.predict(x_train_scaled)
        self.predictions["train"]["prediction"] = self.model.predict(x_train)

        self.predictions["dev"] = pd.DataFrame()
        self.predictions["dev"][self.row_column] = self.data_dev[self.row_column]
        # self.predictions["dev"]["prediction"] = self.model.predict(x_dev_scaled)
        self.predictions["dev"]["prediction"] = self.model.predict(self.x_dev)

        self.predictions["test"] = pd.DataFrame()
        self.predictions["test"][self.row_column] = self.data_test[self.row_column]
        # self.predictions["test"]["prediction"] = self.model.predict(x_test_scaled)
        self.predictions["test"]["prediction"] = self.model.predict(self.x_test)

        # special treatment of targets due to sklearn
        if len(self.target_assignments) == 1:
            target_col = list(self.target_assignments.values())[0]
            self.predictions["train"][target_col] = y_train.squeeze(axis=1)
            self.predictions["dev"][target_col] = self.y_dev.squeeze(axis=1)
            self.predictions["test"][target_col] = self.y_test.squeeze(axis=1)
        else:
            for target_col in self.target_assignments.values():
                self.predictions["train"][target_col] = data_train[target_col]
                self.predictions["dev"][target_col] = self.data_dev[target_col]
                self.predictions["test"][target_col] = self.data_test[target_col]

        # add additional predictions for classifications
        if self.ml_type == "classification":
            columns = [int(x) for x in self.model.classes_]  # column names --> int
            # self.predictions["train"][columns] = self.model.predict_proba(
            #    x_train_scaled
            # )
            self.predictions["train"][columns] = self.model.predict_proba(x_train)
            # self.predictions["dev"][columns] = self.model.predict_proba(x_dev_scaled)
            self.predictions["dev"][columns] = self.model.predict_proba(self.x_dev)
            # self.predictions["test"][columns]=self.model.predict_proba(x_test_scaled)
            self.predictions["test"][columns] = self.model.predict_proba(self.x_test)

        # add additional predictions for time to event predictions
        if self.ml_type == "timetoevent":
            pass

        # calculate used features, but only if required for optuna max_features>0
        # (to save time, shap or permutation importances may take a lot of time)
        if self.max_features > 0:
            self.features_used = self._calculate_features_used()
        else:
            self.features_used = []

        return self

    def _calculate_features_used(self):
        """Calculate used features, method based on model type."""
        feature_method = ModelInventory().get_feature_method(self.ml_model_type)

        if feature_method == "internal":
            self.calculate_fi_internal()
            fi_df = self.feature_importances["internal"]
        elif feature_method == "shap":
            self.calculate_fi_featuresused_shap(partition="dev")
            fi_df = self.feature_importances["shap_dev"]
        elif feature_method == "permutation":
            self.calculate_fi_permutation(
                partition="dev", n_repeats=2
            )  # only 2 repeats!
            fi_df = self.feature_importances["permutation_dev"]
        elif feature_method == "constant":
            self.calculate_fi_constant()
            fi_df = self.feature_importances["constant"]
        else:
            raise ValueError("feature method provided in model config not supported")

        return fi_df[fi_df["importance"] != 0]["feature"].tolist()

    def calculate_fi_constant(self):
        """Provide flat feature importance table."""
        fi_df = pd.DataFrame()
        fi_df["feature"] = self.feature_columns
        fi_df["importance"] = 1
        self.feature_importances["constant"] = fi_df

    def calculate_fi_internal(self):
        """Sklearn provided internal feature importance (based on train dataset)."""
        # skurv model throw NotImplementedError when accessing "feature_importances"
        if self.ml_type == "timetoevent":
            fi_df = pd.DataFrame(columns=["feature", "importance"])
            print("Warning: Internal features importances not available.")
            self.feature_importances["internal"] = fi_df
            return
        elif hasattr(self.model, "feature_importances_"):
            fi_df = pd.DataFrame()
            fi_df["feature"] = self.feature_columns
            fi_df["importance"] = self.model.feature_importances_
        else:
            fi_df = pd.DataFrame(columns=["feature", "importance"])
            print("Warning: Internal features importances not available.")
        self.feature_importances["internal"] = fi_df

    def calculate_fi_group_permutation(self, partition="dev", n_repeats=10):
        """Permutation feature importance, group version."""
        print(
            f"Calculating permutation feature importances ({partition})"
            ". This may take a while..."
        )
        # fixed confidence level
        confidence_level = 0.95
        feature_columns = self.feature_columns
        target_assignments = self.target_assignments
        target_metric = self.target_metric
        model = self.model
        feature_groups = self.feature_groups

        if partition == "dev":
            data = self.data_dev
        elif partition == "test":
            data = self.data_test

        if not set(feature_columns).issubset(data.columns):
            raise ValueError("Features missing in provided dataset.")

        # check that targets are in dataset
        # MISSING

        # keep all features and add group features
        # create features dict
        feature_columns_dict = {x: [x] for x in feature_columns}
        features_dict = {**feature_columns_dict, **feature_groups}

        # calculate baseline score
        baseline_score = get_performance_score(
            model, data, feature_columns, target_metric, target_assignments
        )

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
        # calculate pfi
        for name, feature in features_dict.items():
            data_pfi = data.copy()
            fi_lst = list()

            for _ in range(n_repeats):
                # replace column with random selection from that column of data_all
                # we use data_all as the validation dataset may be small
                for feat in feature:
                    data_pfi[feat] = np.random.choice(
                        data[feat], len(data_pfi), replace=False
                    )
                pfi_score = get_performance_score(
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
                name,
                pfi_mean,
                stddev,
                p_value,
                n,
                ci_low,
                ci_high,
            ]

        results_df = results_df.sort_values(by="importance", ascending=False)
        self.feature_importances["permutation" + "_" + partition] = results_df

    def calculate_fi_permutation(self, partition="dev", n_repeats=10):
        """Permutation feature importance."""
        print(
            f"Calculating permutation feature importances ({partition})"
            ". This may take a while..."
        )

        if self.ml_type == "timetoevent":
            # sksurv models only provide inbuilt scorer (CI)
            # more work needed to support other metrics
            scoring_type = None
        else:
            scoring_type = scorer_string_inventory[self.target_metric]

        if partition == "dev":
            x = self.x_dev
            y = self.y_dev
        elif partition == "test":
            x = self.x_test
            y = self.y_test

        perm_importance = permutation_importance(
            self.model,
            X=x,
            y=y,
            n_repeats=n_repeats,
            random_state=0,
            scoring=scoring_type,
        )

        fi_df = pd.DataFrame()
        fi_df["feature"] = self.feature_columns
        fi_df["importance"] = perm_importance.importances_mean
        fi_df["importance_std"] = perm_importance.importances_std
        self.feature_importances["permutation" + "_" + partition] = fi_df

    def calculate_fi_lofo(self):
        """LOFO feature importance."""
        print("Calculating LOFO feature importance. This may take a while...")
        # first, dev only
        feature_columns = self.feature_columns
        # calculate dev+test baseline scores
        baseline_dev = get_performance_score(
            self.model,
            self.data_dev,
            feature_columns,
            self.target_metric,
            self.target_assignments,
        )
        baseline_test = get_performance_score(
            self.model,
            self.data_test,
            feature_columns,
            self.target_metric,
            self.target_assignments,
        )

        # create features dict
        feature_columns_dict = {x: [x] for x in feature_columns}
        lofo_features = {**feature_columns_dict, **self.feature_groups}

        # lofo
        fi_dev_df = pd.DataFrame(columns=["feature", "importance"])
        fi_test_df = pd.DataFrame(columns=["feature", "importance"])
        for name, lofo_feature in lofo_features.items():
            selected_features = copy.deepcopy(feature_columns)
            model = copy.deepcopy(self.model)
            selected_features = [x for x in selected_features if x not in lofo_feature]
            # retrain model
            if len(self.target_assignments) == 1:
                # standard sklearn single target models
                model.fit(
                    self.data_train[selected_features],
                    self.y_train.squeeze(axis=1),
                )
            else:
                # multi target models, incl. time2event
                model.fit(self.data_train[selected_features], self.y_train)

            # get lofo dev + test scores
            score_dev = get_performance_score(
                model,
                self.data_dev,
                selected_features,
                self.target_metric,
                self.target_assignments,
            )
            score_test = get_performance_score(
                model,
                self.data_test,
                selected_features,
                self.target_metric,
                self.target_assignments,
            )

            fi_dev_df.loc[len(fi_dev_df)] = [name, baseline_dev - score_dev]
            fi_test_df.loc[len(fi_test_df)] = [name, baseline_test - score_test]

        self.feature_importances["lofo" + "_dev"] = fi_dev_df
        self.feature_importances["lofo" + "_test"] = fi_test_df

    def calculate_fi_featuresused_shap(self, partition="dev"):
        """Shap feature importance, specifically for calc_features_used."""
        print("Calculating shape feature importances. This may take a while...")
        # Initialize shape explainer using training data
        # improve speed by self.x_train.sample(n=100, replace=True, random_state=0)
        # TreeExplainer(model, data, model_output="probability",
        # feature_perturbation="interventional",)

        if self.ml_type == "timetoevent":
            raise ValueError("Shap feature importance not supported for timetoevent")
        else:
            # this works for linear and tree models
            explainer = shap.Explainer(
                self.model,
                self.x_train,
            )

            # Calculate SHAP values for the dev dataset
            if partition == "dev":
                shap_values = explainer.shap_values(self.x_dev)  # pylint: disable=E1101
            elif partition == "test":
                shap_values = explainer.shap_values(  # pylint: disable=E1101
                    self.x_test
                )
            else:
                raise ValueError("dataset type not supported")

        # Calculate the feature importances as the absolute mean of SHAP values
        if isinstance(shap_values, list):  # shap v.< 0.45, multi-output, e.g. 2 classes
            feature_importances = np.abs(shap_values[0]).mean(axis=0)
        elif isinstance(shap_values, np.ndarray):  # shap v. >= 0.45 or single output
            if shap_values.ndim == 2:  # single output
                feature_importances = np.abs(shap_values).mean(axis=0)
            elif (
                shap_values.ndim == 3
            ):  # multi-output (e.g. 2 classes) for shap >= 0.45
                feature_importances = np.abs(shap_values[:, :, 0]).mean(axis=0)
            else:
                raise TypeError("Type error shape_value")
        else:
            raise TypeError("Type error shape_value")

        fi_df = pd.DataFrame()
        fi_df["feature"] = self.feature_columns
        fi_df["importance"] = feature_importances
        # remove features with extremely small fi
        fi_df = fi_df[fi_df["importance"] > fi_df["importance"].max() / 1000]
        self.feature_importances["shap" + "_" + partition] = fi_df

    def calculate_fi_shap(self, partition="dev", shap_type="kernel"):
        """Shap feature importance."""
        print(
            f"Calculating shape feature importances ({partition})"
            ". This may take a while..."
        )
        # here we use model agnostic methods to estimate shap values
        # methods: (a) kernel (b) permutation (c) exact

        if self.ml_type == "classification":
            model = self.model.predict_proba
        else:
            model = self.model.predict

        # select data
        if partition == "dev":
            data = self.x_dev
        elif partition == "test":
            data = self.x_test
        else:
            raise ValueError("dataset type not supported")

        # select explainer based on shap_type
        if shap_type == "exact":
            explainer = shap.explainers.Exact(model, data)
        elif shap_type == "permutation":
            explainer = shap.explainers.Permutation(model, data)
        elif shap_type == "kernel":
            explainer = shap.explainers.Kernel(model, data)
        else:
            raise ValueError(f"Shap type {shap_type} not supported.")

        # get shap values
        shap_values = explainer(data).values

        # Calculate the feature importances as the absolute mean of SHAP values
        if isinstance(shap_values, list):  # shap v.< 0.45, multi-output, e.g. 2 classes
            feature_importances = np.abs(shap_values[0]).mean(axis=0)
        elif isinstance(shap_values, np.ndarray):  # shap v. >= 0.45 or single output
            if shap_values.ndim == 2:  # single output
                feature_importances = np.abs(shap_values).mean(axis=0)
            elif (
                shap_values.ndim == 3
            ):  # multi-output (e.g. 2 classes) for shap v. >= 0.45
                feature_importances = np.abs(shap_values[:, :, 0]).mean(axis=0)
            else:
                raise TypeError("Type error shape_value")
        else:
            raise TypeError("Type error shape_value")

        fi_df = pd.DataFrame()
        fi_df["feature"] = data.columns.tolist()
        fi_df["importance"] = feature_importances
        # remove features with extremely small fi
        fi_df = fi_df[fi_df["importance"] > fi_df["importance"].max() / 1000]
        self.feature_importances["shap" + "_" + partition] = fi_df

    def predict(self, x):
        """Predict."""
        return self.model.predict(x)

    def predict_proba(self, x):
        """Predict_proba."""
        return self.model.predict_proba(x)

    def to_pickle(self, path):
        """Save training."""

    @classmethod
    def from_pickle(cls, path):
        """Load training."""
