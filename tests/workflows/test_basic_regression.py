"""Test basic regression workflow."""

import os

import pandas as pd

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.modules import Octo


def test_basic_regression():
    """Test basic regression."""
    data_df = (
        pd.read_csv(
            os.path.join(os.getcwd(), "datasets", "california_housing_prices.csv")
        )
        .reset_index()
        .astype(
            {
                "housing_median_age": int,
                "total_rooms": int,
                "population": int,
                "households": int,
                "median_income": int,
                "median_house_value": int,
            }
        )
        .loc[0:100, :]
    )

    octo_data = OctoData(
        data=data_df,
        target_columns=["median_house_value"],
        feature_columns=[
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "population",
            "households",
            "median_income",
        ],
        sample_id="index",
        datasplit_type="sample",
    )

    config_study = ConfigStudy(
        name="test_basic_regression",
        ml_type="regression",
        target_metric="R2",
        silently_overwrite_study=True,
        ignore_data_health_warning=True,
    )

    config_manager = ConfigManager(
        outer_parallelization=False, run_single_experiment_num=0
    )

    config_sequence = ConfigSequence(
        [
            Octo(
                item_id=1,
                input_item_id=0,
                description="step_1",
                models=["RandomForestRegressor", "XGBRegressor"],
                n_trials=1,
            ),
        ]
    )

    octo_ml = OctoML(
        octo_data,
        config_study=config_study,
        config_manager=config_manager,
        config_sequence=config_sequence,
    )

    octo_ml.create_outer_experiments()
    octo_ml.run_outer_experiments()

    success = True

    assert success is True
