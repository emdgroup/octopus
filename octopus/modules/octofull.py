"""OctoFull Module."""
import pandas as pd
from attrs import define, field, validators

from octopus.experiment import OctoExperiment

# from octopus.ml import DataSplit
from octopus.models.config import model_inventory


@define
class OctoFull:
    """OctoFull."""

    experiment: OctoExperiment = field(
        validator=[validators.instance_of(OctoExperiment)]
    )
    model = field(init=False)

    @property
    def x_train(self) -> pd.DataFrame:
        """x_train."""
        return self.experiment.data_traindev[self.experiment.feature_columns]

    @property
    def y_train(self) -> pd.DataFrame:
        """y_train."""
        return self.experiment.data_traindev[
            self.experiment.target_assignments.values()
        ]

    @property
    def x_test(self) -> pd.DataFrame:
        """x_test."""
        return self.experiment.data_test[self.experiment.feature_columns]

    @property
    def y_test(self) -> pd.DataFrame:
        """y_test."""
        return self.experiment.data_test[self.experiment.target_assignments.values()]

    @property
    def params(self) -> pd.DataFrame:
        """OctoFull parameters."""
        return self.experiment.ml_config["config"]

    def run_experiment(self):
        """Run experiment."""
        # perform and save datasplit
        # train models

        # data_splits = DataSplit(
        #    dataset=self.experiment.data_traindev,
        #    datasplit_col=datasplit_col,
        #    seed=self.experiment.config.,
        #    num_folds=self.oconfig.k_ou,
        #    stratification_col=self.experiment.stratification_col,
        # ).get_datasplits()
        # )

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
    data_train: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    data_dev: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    data_test: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    dim_reduction: str = field(validator=[validators.instance_of(str)])
    outl_reduction: int = field(validator=[validators.instance_of(int)])
    ml_seed: int = field(validator=[validators.instance_of(int)])
    ml_model_type: str = field(validator=[validators.instance_of(str)])
    ml_model_params: dict = field(validator=[validators.instance_of(dict)])
    ml_jobs: int = field(validator=[validators.instance_of(int)])
    class_weights: bool = (field(validator=[validators.instance_of(bool)]),)
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

        self.model = model_inventory[self.ml_model_type](**self.ml_model_params)
        self.model.fit(self.data_train)

        # missing: include row_id
        self.predictions["train"] = self.model.predict(self.data_train)
        self.predictions["dev"] = self.model.predict(self.data_dev)
        self.predictions["test"] = self.model.predict(self.data_test)

        if self.ml_type == "classification":
            # missing: include row_id
            self.probabilities["train"] = self.model.predict_proba(self.data_train)
            self.probabilities["dev"] = self.model.predict_proba(self.data_dev)
            self.probabilities["test"] = self.model.predict_proba(self.data_test)

        # missing: other feature reduction methods

    def to_pickle(self, path):
        """Save training."""

    @classmethod
    def from_pickle(cls, path):
        """Load training."""
