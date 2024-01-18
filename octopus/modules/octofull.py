"""OctoFull Module."""
import warnings

import pandas as pd
from attrs import define, field, validators
from sklearn.exceptions import DataConversionWarning
from sklearn.inspection import permutation_importance

from octopus.datasplit import DataSplit
from octopus.experiment import OctoExperiment
from octopus.models.config import model_inventory

# TOBEDONE:
# - training: if condition for single target_assignments - conversion of input
# - all multi target models are done separately, shap and permutation
#   importance may not work anyways
# - include shapley and permutation importance
# - separate model parameters for optuna and others
#   (seeds, ..... = mostly fixed, not part of optimization)
# - add model parameters together
# - include dimensionality reduction
# - include outlier elimination


@define
class OctoFull:
    """OctoFull."""

    experiment: OctoExperiment = field(
        validator=[validators.instance_of(OctoExperiment)]
    )
    model = field(init=False)
    data_splits = field(default=dict(), validator=[validators.instance_of(dict)])

    def __attrs_post_init__(self):
        # create datasplit during init
        self.data_splits = DataSplit(
            dataset=self.experiment.data_traindev,
            datasplit_col=self.experiment.datasplit_column,
            seed=self.experiment.ml_config["config"]["datasplit_seed_inner"],
            num_folds=self.experiment.ml_config["config"]["k_inner"],
            stratification_col=self.experiment.stratification_column,
        ).get_datasplits()

    def run_experiment(self):
        """Run experiment."""
        # run trainings
        for key, split in self.data_splits.items():
            model_params = {
                "max_depth": 3,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": 0.8,
                "n_estimators": 100,
            }

            training = Training(
                training_id=self.experiment.id + "_" + str(key),
                ml_type=self.experiment.ml_type,
                x_train=split["train"][
                    self.experiment.feature_columns
                ],  # inner datasplit
                x_dev=split["test"][self.experiment.feature_columns],  # inner datasplit
                x_test=self.experiment.data_test[self.experiment.feature_columns],
                y_train=split["train"][self.experiment.target_assignments.values()],
                y_dev=split["test"][self.experiment.target_assignments.values()],
                y_test=self.experiment.data_test[
                    self.experiment.target_assignments.values()
                ],
                dim_reduction="",
                outl_reduction=3,
                ml_seed=self.experiment.ml_config["config"]["ml_seed"],
                ml_model_type="ExtraTreesClassifier",
                ml_model_params=model_params,
                ml_jobs=self.experiment.ml_config["config"]["ml_jobs"],
                class_weights=self.experiment.ml_config["config"]["class_weights"],
                target_metric=self.experiment.config["target_metric"],
            )
            training.run_training()
            # training.to_pickle(path)

        return self.experiment

    def predict(self, dataset: pd.DataFrame):
        """Predict on new dataset."""
        # this is old and not working code
        model = self.experiment.models["model_0"]
        return model.predict(dataset[self.experiment.feature_columns])

    def predict_proba(self, dataset: pd.DataFrame):
        """Predict_proba on new dataset."""
        # this is old and not working code
        if self.experiment.ml_type == "classification":
            self.model = self.experiment.models["model_0"]
            return self.model.predict_proba(dataset[self.experiment.feature_columns])
        else:
            raise ValueError("predict_proba only supported for classifications")


@define
class Training:
    """Model Training Class."""

    training_id: str = field(validator=[validators.instance_of(str)])
    ml_type: str = field(validator=[validators.instance_of(str)])
    x_train: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    x_dev: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    x_test: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    y_train: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    y_dev: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    y_test: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    dim_reduction: str = field(validator=[validators.instance_of(str)])
    outl_reduction: int = field(validator=[validators.instance_of(int)])
    ml_seed: int = field(validator=[validators.instance_of(int)])
    ml_model_type: str = field(validator=[validators.instance_of(str)])
    ml_model_params: dict = field(validator=[validators.instance_of(dict)])
    ml_jobs: int = field(validator=[validators.instance_of(int)])
    class_weights: bool = field(validator=[validators.instance_of(bool)])
    target_metric: str = field(validator=[validators.instance_of(str)])
    model = field(init=False)
    predictions: dict = field(default=dict(), validator=[validators.instance_of(dict)])
    probabilities: dict = field(
        default=dict(), validator=[validators.instance_of(dict)]
    )
    # class weights as optimization parameters

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

    def __attrs_post_init__(self):
        # reset index
        pass

    def run_training(self):
        """Run trainings."""
        # missing: dim reduction
        # missing: outlier removal
        print("training model")
        print("model params:", self.ml_model_params)
        self.model = model_inventory[self.ml_model_type](**self.ml_model_params)

        # Disable the warning
        warnings.filterwarnings("ignore", category=DataConversionWarning)
        self.model.fit(self.x_train, self.y_train)
        # Enable the warning again (optional)
        warnings.filterwarnings("default", category=DataConversionWarning)

        # missing: include row_id
        self.predictions["train"] = self.model.predict(self.x_train)
        self.predictions["dev"] = self.model.predict(self.x_dev)
        self.predictions["test"] = self.model.predict(self.x_test)

        if self.ml_type == "classification":
            # missing: include row_id
            self.probabilities["train"] = self.model.predict_proba(self.x_train)
            self.probabilities["dev"] = self.model.predict_proba(self.x_dev)
            self.probabilities["test"] = self.model.predict_proba(self.x_test)

        # missing: other feature reduction methods
        result = permutation_importance(
            self.model, X=self.x_dev, y=self.y_dev, n_repeats=10, random_state=0
        )
        print(result)

    def to_pickle(self, path):
        """Save training."""

    @classmethod
    def from_pickle(cls, path):
        """Load training."""
