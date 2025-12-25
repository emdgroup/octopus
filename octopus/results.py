"""Results."""

import pandas as pd
from attrs import Factory, define, field, validators

from octopus.models.config import BaseModel


@define
class ModuleResults:
    """Results."""

    id: str = field(validator=[validators.instance_of(str)])
    """Results str id."""

    experiment_id: int | None = field(
        validator=validators.optional(
            validators.and_(
                validators.instance_of(int),  # Ensure it's an int if not None
                validators.ge(0),  # Ensure int is >= 0
            )
        )
    )
    """Experiment id."""

    task_id: int | None = field(
        validator=validators.optional(
            validators.and_(
                validators.instance_of(int),  # Ensure it's an int if not None
                validators.ge(0),  # Ensure int is >= 0
            )
        )
    )
    """Sequence id."""

    model: BaseModel = field()
    """Saved Model."""

    scores: dict = field(default=Factory(dict), validator=[validators.instance_of(dict)])
    """Scores, dictionary."""

    predictions: dict = field(default=Factory(dict), validator=[validators.instance_of(dict)])
    """Predictions, dictionary."""

    feature_importances: dict[str, pd.DataFrame] = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Feature importances, dictionary."""

    selected_features: list[str] = field(default=Factory(list), validator=[validators.instance_of(list)])
    """Selected features, list of strings."""

    results: dict = field(default=Factory(dict), validator=[validators.instance_of(dict)])
    """Other results, dictionary."""

    def create_prediction_df(self):
        """Create prediction dataframe."""
        df_prediction = pd.DataFrame()

        # prediction keys in octo
        # ["0_0_0"]["train/dev/test"] - predictions from trainings
        # ["ensemble"]["test"] - predictions from bag ensemble
        # ["autogluon"]["test"] - autogluon

        for key, value in self.predictions.items():
            for split, df in value.items():
                temp_df = df.copy()
                temp_df["training_id"] = key
                temp_df["split"] = split
                df_prediction = pd.concat([df_prediction, temp_df], ignore_index=True)

        df_prediction["experiment_id"] = self.experiment_id
        df_prediction["task_id"] = self.task_id
        return df_prediction

    def create_feature_importance_df(self):
        """Create feature importance dataframe."""
        df_feature_importance = pd.DataFrame()

        for key, value in self.feature_importances.items():
            for fi_type, df in value.items():
                if fi_type in ["internal", "permutation_dev"]:
                    temp_df = df.copy()
                    temp_df["fi_type"] = fi_type
                    temp_df["experiment_id"] = self.experiment_id
                    temp_df["task_id"] = self.task_id
                    temp_df["training_id"] = key

                    df_feature_importance = pd.concat([df_feature_importance, temp_df], ignore_index=True)
        return df_feature_importance
