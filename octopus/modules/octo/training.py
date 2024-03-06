"""OctoFull Trainings."""

import numpy as np
import pandas as pd
import shap
from attrs import define, field, validators
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MaxAbsScaler

from octopus.models.config import model_inventory


@define
class Training:
    """Model Training Class."""

    training_id: str = field(validator=[validators.instance_of(str)])
    ml_type: str = field(validator=[validators.instance_of(str)])
    target_assignments: dict = field(validator=[validators.instance_of(dict)])
    feature_columns: list = field(validator=[validators.instance_of(list)])
    row_column: str = field(validator=[validators.instance_of(str)])
    data_train: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    data_dev: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    data_test: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    target_metric: str = field(validator=[validators.instance_of(str)])
    max_features: int = field(validator=[validators.instance_of(int)])
    # configuration for training
    config_training: dict = field(validator=[validators.instance_of(dict)])
    # training outputs, initialized in post_init
    model = field(default=None)
    predictions: dict = field(init=False, validator=[validators.instance_of(dict)])
    feature_importances: dict = field(
        init=False, validator=[validators.instance_of(dict)]
    )
    features_used: list = field(init=False, validator=[validators.instance_of(list)])
    # scaler
    scaler = field(init=False)

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
        # initialization here due to "Python immutable default"
        self.predictions = dict()
        self.feature_importances = dict()
        self.features_used = list()
        # scaler
        self.scaler = MaxAbsScaler()

    # perform:
    # (1) dim_reduction
    # (2) outlier removal
    # (3) training
    # (4) standard feature importance
    # (4) permutation feature importance
    # (5) shapley feature importance

    # output:
    # (1) predictions
    # (2) probabilities in case of classification
    # (3) feature_importances, which
    # (4)

    def fit(self):
        """Run trainings."""
        # missing:
        # (1) missing: outlier removal
        # (2) scaling
        # (3) missinf dim reduction

        # scaling (!after outlier removal)
        # x_train_scaled = self.scaler.fit_transform(self.x_train)
        # x_dev_scaled = self.scaler.transform(self.x_dev)
        # x_test_scaled = self.scaler.transform(self.x_test)
        self.model = model_inventory[self.ml_model_type]["model"](
            **self.ml_model_params
        )

        if len(self.target_assignments) == 1:
            # standard sklearn single target models
            self.model.fit(
                # x_train_scaled,
                self.x_train,
                self.y_train.squeeze(axis=1),
            )
        else:
            # multi target models, incl. time2event
            # self.model.fit(x_train_scaled, self.y_train)
            self.model.fit(self.x_train, self.y_train)

        self.predictions["train"] = pd.DataFrame()
        self.predictions["train"][self.row_column] = self.data_train[self.row_column]
        # self.predictions["train"]["prediction"] = self.model.predict(x_train_scaled)
        self.predictions["train"]["prediction"] = self.model.predict(self.x_train)

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
            self.predictions["train"][target_col] = self.y_train.squeeze(axis=1)
            self.predictions["dev"][target_col] = self.y_dev.squeeze(axis=1)
            self.predictions["test"][target_col] = self.y_test.squeeze(axis=1)
        else:
            for target_col in self.target_assignments.values():
                self.predictions["train"][target_col] = self.data_train[target_col]
                self.predictions["dev"][target_col] = self.data_dev[target_col]
                self.predictions["test"][target_col] = self.data_test[target_col]

        # add additional predictions for classifications
        if self.ml_type == "classification":
            columns = [int(x) for x in self.model.classes_]  # column names --> int
            # self.predictions["train"][columns] = self.model.predict_proba(
            #    x_train_scaled
            # )
            self.predictions["train"][columns] = self.model.predict_proba(self.x_train)
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
        feature_method = model_inventory[self.ml_model_type]["feature_method"]

        if feature_method == "internal":
            self.calculate_fi_internal()
            fi_df = self.feature_importances["internal"]
        elif feature_method == "shap":
            self.calculate_fi_shap(partition="dev")
            fi_df = self.feature_importances["shap_dev"]
        elif feature_method == "permutation":
            self.calculate_fi_permutation(
                partition="dev", n_repeats=2
            )  # only 2 repeats!
            fi_df = self.feature_importances["permutation_dev"]
        else:
            raise ValueError("feature method provided in model config not supported")

        return fi_df[fi_df["importance"] != 0]["feature"].tolist()

    def calculate_fi_internal(self):
        """Sklearn provided internal feature importance (based on train dataset)."""
        if hasattr(self.model, "features_importances_"):
            fi_df = pd.DataFrame()
            fi_df["feature"] = self.feature_columns
            fi_df["importance"] = self.model.features_importances_

        else:
            fi_df = pd.DataFrame(columns=["feature", "importance"])
        self.feature_importances["internal"] = fi_df

    def calculate_fi_permutation(self, partition="dev", n_repeats=10):
        """Permutation feature importance."""
        print("Calculating permutation feature importances. This may take a while...")
        if partition == "dev":
            perm_importance = permutation_importance(
                self.model,
                X=self.x_dev,
                y=self.y_dev,
                n_repeats=n_repeats,
                random_state=0,
            )
        elif partition == "test":
            perm_importance = permutation_importance(
                self.model,
                X=self.x_test,
                y=self.y_test,
                n_repeats=n_repeats,
                random_state=0,
            )
        fi_df = pd.DataFrame()
        fi_df["feature"] = self.feature_columns
        fi_df["importance"] = perm_importance.importances_mean
        fi_df["importance_std"] = perm_importance.importances_std
        self.feature_importances["permutation" + "_" + partition] = fi_df

    def calculate_fi_shap(self, partition="dev"):
        """Shap feature importance."""
        print("Calculating shape feature importances. This may take a while...")
        # Initialize shape explainer using training data
        # improve speed by self.x_train.sample(n=100, replace=True, random_state=0)
        explainer = shap.Explainer(self.model, self.x_train)

        # Calculate SHAP values for the dev dataset
        if partition == "dev":
            # shap_values = explainer(self.x_dev)  # pylint: disable=E1101
            shap_values = explainer.shap_values(self.x_dev)  # pylint: disable=E1101
        elif partition == "test":
            # shap_values = explainer(self.x_test)
            shap_values = explainer.shap_values(self.x_test)  # pylint: disable=E1101
        else:
            raise ValueError("dataset type not supported")

        # Calculate the feature importances as the absolute mean of SHAP values
        feature_importances = np.abs(shap_values).mean(axis=0)

        fi_df = pd.DataFrame()
        fi_df["feature"] = self.feature_columns
        fi_df["importance"] = feature_importances
        self.feature_importances["shap" + "_" + partition] = fi_df

    def to_pickle(self, path):
        """Save training."""

    @classmethod
    def from_pickle(cls, path):
        """Load training."""
