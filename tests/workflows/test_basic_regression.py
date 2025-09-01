"""Test basic regression workflow."""

from sklearn.datasets import fetch_california_housing

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.modules import Octo


def test_basic_regression():
    """Test basic regression."""
    data_df = (
        fetch_california_housing(as_frame=True)
        .frame.reset_index()
        .astype(
            {
                "HouseAge": int,
                "AveRooms": float,
                "AveBedrms": float,
                "Population": int,
                "AveOccup": float,
                "MedInc": float,
                "MedHouseVal": float,
            }
        )
        .loc[0:100, :]
    )

    octo_data = OctoData(
        data=data_df,
        target_columns=["MedHouseVal"],
        feature_columns=[
            "Longitude",
            "Latitude",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "MedInc",
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

    config_manager = ConfigManager(outer_parallelization=False, run_single_experiment_num=0)

    config_sequence = ConfigSequence(
        [
            Octo(
                sequence_id=0,
                input_sequence_id=-1,
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
    octo_ml.run_study()
    success = True
    assert success is True
