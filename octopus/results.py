"""Results."""

import pandas as pd
from attrs import Factory, define, field, validators


@define
class ModuleResults:
    """Results."""

    id: str = field(validator=[validators.instance_of(str)])
    """Results str id."""

    model = field(default="")
    """Saved Model."""

    scores: dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Scores, dictionary."""

    predictions: dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Predictions, dictionary."""

    feature_importances: dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Feature importances, dictionary."""

    selected_features: list = field(
        default=Factory(list), validator=[validators.instance_of(list)]
    )
    """Feature importances, dictionary."""

    results: dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Other results, dictionary."""

    def create_prediction_dataframe(self):
        df_prediction = pd.DataFrame()

        for key, value in self.predictions.items():
            if "_" in key:
                experiment_id, sequence_id, split_id = key.split("_")
                for split, df in value.items():
                    temp_df = df.copy()
                    temp_df["experiment_id"] = experiment_id
                    temp_df["sequence_id"] = sequence_id
                    temp_df["split_id"] = split_id
                    temp_df["split"] = split
                    df_prediction = pd.concat(
                        [df_prediction, temp_df], ignore_index=True
                    )
            elif key == "ensemble":
                # ensemble
                temp_df = value["test"].copy()
                temp_df["split_id"] = "ensemble"
                temp_df["split"] = "test"
                df_prediction = pd.concat([df_prediction, temp_df], ignore_index=True)

            else:
                raise ValueError("Unknown key in prediction dictionary.")
        return df_prediction
