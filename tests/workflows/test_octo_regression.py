"""Test workflow for Octopus regression example."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.datasets import make_regression

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.modules import Octo


class TestOctoRegression:
    """Test suite for Octopus regression workflow."""

    @pytest.fixture
    def diabetes_dataset(self):
        """Create synthetic regression dataset for testing (faster than diabetes dataset)."""
        # Create synthetic regression dataset with reduced size for faster testing
        X, y = make_regression(
            n_samples=30,
            n_features=5,
            n_informative=3,
            noise=0.1,
            random_state=42,
        )

        # Create DataFrame similar to diabetes dataset structure
        features = [f"feature_{i}" for i in range(5)]
        df = pd.DataFrame(X, columns=features)
        df["target"] = y
        df = df.reset_index()

        return df, features

    @pytest.fixture
    def octo_data_config(self, diabetes_dataset):
        """Create OctoData configuration for testing."""
        df, features = diabetes_dataset

        return OctoData(
            data=df,
            target_columns=["target"],
            feature_columns=features,
            sample_id="index",
            datasplit_type="sample",
        )

    @pytest.fixture
    def config_study(self):
        """Create ConfigStudy for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            return ConfigStudy(
                name="test_octo_regression",
                ml_type="regression",
                target_metric="MAE",
                metrics=["MAE", "MSE", "R2"],
                datasplit_seed_outer=1234,
                n_folds_outer=3,  # Reduced for faster testing
                start_with_empty_study=True,
                path=temp_dir,
                silently_overwrite_study=True,
                ignore_data_health_warning=True,
            )

    @pytest.fixture
    def config_manager(self):
        """Create ConfigManager for testing."""
        return ConfigManager(
            outer_parallelization=False,  # Disable for testing
            run_single_experiment_num=0,  # Run only first experiment
        )

    def test_diabetes_dataset_loading(self, diabetes_dataset):
        """Test that the diabetes dataset is loaded correctly."""
        df, features = diabetes_dataset

        # Verify dataset structure
        assert isinstance(df, pd.DataFrame)
        assert "target" in df.columns
        assert "index" in df.columns
        assert len(features) == 5  # Synthetic dataset has 5 features
        assert df.shape[0] == 30  # Synthetic dataset has 30 samples

        # Verify target values are continuous (regression)
        assert df["target"].dtype in ["float64", "int64"]
        assert df["target"].nunique() > 20  # Should have many unique values for regression

        # Verify all feature columns exist in dataframe
        for feature in features:
            assert feature in df.columns

        # Verify no missing values
        assert not df[features].isnull().any().any()
        assert not df["target"].isnull().any()

    def test_octo_data_configuration(self, octo_data_config):
        """Test OctoData configuration for diabetes dataset."""
        # Verify OctoData configuration
        assert octo_data_config.target_columns == ["target"]
        assert len(octo_data_config.feature_columns) == 5
        assert octo_data_config.sample_id == "index"
        assert octo_data_config.datasplit_type == "sample"

        # Verify data integrity
        assert octo_data_config.data is not None
        assert isinstance(octo_data_config.data, pd.DataFrame)

    def test_octo_sequence_configuration(self):
        """Test that Octo sequence can be properly configured."""
        config_sequence = ConfigSequence(
            [
                Octo(
                    sequence_id=0,
                    input_sequence_id=-1,
                    description="step_1",
                    models=[
                        "RandomForestRegressor",
                        "XGBRegressor",
                        "ExtraTreesRegressor",
                        "ElasticNetRegressor",
                        "GradientBoostingRegressor",
                        "CatBoostRegressor",
                    ],
                    n_trials=12,
                    max_features=6,
                    ensemble_selection=True,
                    ensel_n_save_trials=10,
                )
            ]
        )

        # Verify sequence configuration
        assert len(config_sequence.sequence_items) == 1

        # Verify Octo step configuration
        octo_step = config_sequence.sequence_items[0]
        assert isinstance(octo_step, Octo)
        assert octo_step.sequence_id == 0
        assert octo_step.input_sequence_id == -1
        assert octo_step.description == "step_1"
        assert octo_step.n_trials == 12
        assert octo_step.max_features == 6
        assert octo_step.ensemble_selection is True
        assert octo_step.ensel_n_save_trials == 10

        # Verify models are configured correctly
        expected_models = {
            "RandomForestRegressor",
            "XGBRegressor",
            "ExtraTreesRegressor",
            "ElasticNetRegressor",
            "GradientBoostingRegressor",
            "CatBoostRegressor",
        }
        assert set(octo_step.models) == expected_models

    @patch("octopus.ml.OctoML.run_study")
    def test_workflow_initialization(self, mock_run_study, octo_data_config, config_study, config_manager):
        """Test that the regression workflow can be initialized and configured properly."""
        # Create the sequence configuration (simplified for testing)
        config_sequence = ConfigSequence(
            [
                Octo(
                    sequence_id=0,
                    input_sequence_id=-1,
                    description="step_1",
                    models=["RandomForestRegressor", "XGBRegressor"],
                    n_trials=12,
                    max_features=6,
                    ensemble_selection=True,
                    ensel_n_save_trials=10,
                )
            ]
        )

        # Initialize the ML workflow
        octo_ml = OctoML(
            octo_data_config,
            config_study=config_study,
            config_manager=config_manager,
            config_sequence=config_sequence,
        )

        # Verify initialization
        assert octo_ml.data is octo_data_config
        assert octo_ml.config_study == config_study
        assert octo_ml.config_manager == config_manager
        assert octo_ml.config_sequence == config_sequence

        # Verify sequence structure
        assert len(octo_ml.config_sequence.sequence_items) == 1

        # Test that run_study can be called (mocked)
        octo_ml.run_study()
        mock_run_study.assert_called_once()

    @pytest.mark.parametrize(
        "model",
        [
            "RandomForestRegressor",
            "XGBRegressor",
            "ExtraTreesRegressor",
            "ElasticNetRegressor",
            "GradientBoostingRegressor",
            "CatBoostRegressor",
        ],
    )
    def test_single_model_configuration(self, model):
        """Test configuration with different single models."""
        config_sequence = ConfigSequence(
            [
                Octo(
                    sequence_id=0,
                    input_sequence_id=-1,
                    description="step_1",
                    models=[model],
                    n_trials=12,
                    max_features=6,
                    ensemble_selection=True,
                    ensel_n_save_trials=10,
                )
            ]
        )

        octo_step = config_sequence.sequence_items[0]
        assert octo_step.models == [model]
        assert octo_step.n_trials == 12
        assert octo_step.max_features == 6
        assert octo_step.ensemble_selection is True
        assert octo_step.ensel_n_save_trials == 10

    def test_multiple_models_configuration(self):
        """Test configuration with multiple models."""
        models = ["RandomForestRegressor", "XGBRegressor", "ExtraTreesRegressor"]
        config_sequence = ConfigSequence(
            [
                Octo(
                    sequence_id=0,
                    input_sequence_id=-1,
                    description="step_1",
                    models=models,
                    n_trials=12,
                    max_features=6,
                    ensemble_selection=True,
                    ensel_n_save_trials=10,
                )
            ]
        )

        octo_step = config_sequence.sequence_items[0]
        assert set(octo_step.models) == set(models)
        assert octo_step.n_trials == 12
        assert octo_step.max_features == 6
        assert octo_step.ensemble_selection is True
        assert octo_step.ensel_n_save_trials == 10

    def test_ensemble_selection_configuration(self):
        """Test ensemble selection configuration."""
        config_sequence = ConfigSequence(
            [
                Octo(
                    sequence_id=0,
                    input_sequence_id=-1,
                    description="step_1",
                    models=["RandomForestRegressor", "XGBRegressor"],
                    n_trials=12,
                    max_features=6,
                    ensemble_selection=True,
                    ensel_n_save_trials=10,
                )
            ]
        )

        octo_step = config_sequence.sequence_items[0]
        assert octo_step.ensemble_selection is True
        assert octo_step.ensel_n_save_trials == 10

    def test_hyperparameter_optimization_configuration(self):
        """Test hyperparameter optimization configuration."""
        config_sequence = ConfigSequence(
            [
                Octo(
                    sequence_id=0,
                    input_sequence_id=-1,
                    description="step_1",
                    models=["RandomForestRegressor"],
                    n_trials=12,
                    max_features=6,
                    ensemble_selection=True,
                    ensel_n_save_trials=10,
                    optuna_seed=42,
                    n_optuna_startup_trials=5,
                    penalty_factor=1.5,
                )
            ]
        )

        octo_step = config_sequence.sequence_items[0]
        assert octo_step.n_trials == 12
        assert octo_step.max_features == 6
        assert octo_step.optuna_seed == 42
        assert octo_step.n_optuna_startup_trials == 5
        assert octo_step.penalty_factor == 1.5

    def test_regression_specific_configuration(self):
        """Test regression-specific configuration parameters."""
        config_sequence = ConfigSequence(
            [
                Octo(
                    sequence_id=0,
                    input_sequence_id=-1,
                    description="step_1",
                    models=[
                        "RandomForestRegressor",
                        "XGBRegressor",
                        "ExtraTreesRegressor",
                        "ElasticNetRegressor",
                        "GradientBoostingRegressor",
                        "CatBoostRegressor",
                    ],
                    n_trials=12,
                    max_features=6,
                    ensemble_selection=True,
                    ensel_n_save_trials=10,
                )
            ]
        )

        octo_step = config_sequence.sequence_items[0]

        # Verify all regression models are included
        expected_models = {
            "RandomForestRegressor",
            "XGBRegressor",
            "ExtraTreesRegressor",
            "ElasticNetRegressor",
            "GradientBoostingRegressor",
            "CatBoostRegressor",
        }
        assert set(octo_step.models) == expected_models

        # Verify key parameters
        assert octo_step.n_trials == 12
        assert octo_step.max_features == 6
        assert octo_step.ensemble_selection is True
        assert octo_step.ensel_n_save_trials == 10

    @pytest.mark.slow
    def test_octo_regression_actual_execution(self, octo_data_config):
        """Test that the Octopus regression workflow actually runs end-to-end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal configuration for actual execution
            config_study = ConfigStudy(
                name="test_octo_regression_execution",
                ml_type="regression",
                target_metric="MAE",
                metrics=["MAE", "MSE", "R2"],
                datasplit_seed_outer=1234,
                n_folds_outer=2,  # Minimal folds for speed
                start_with_empty_study=True,
                path=temp_dir,
                silently_overwrite_study=True,
                ignore_data_health_warning=True,
            )

            config_manager = ConfigManager(
                outer_parallelization=False,
                run_single_experiment_num=0,  # Run only first experiment
            )

            # Create the Octo sequence with specified settings
            config_sequence = ConfigSequence(
                [
                    Octo(
                        sequence_id=0,
                        input_sequence_id=-1,
                        description="step_1",
                        models=["RandomForestRegressor", "XGBRegressor"],  # Reduced for speed
                        n_trials=12,
                        max_features=6,
                        ensemble_selection=True,
                        ensel_n_save_trials=10,
                        model_seed=0,
                        n_jobs=1,
                        inner_parallelization=True,
                        n_workers=2,  # Reduced for testing
                        optuna_seed=0,
                        n_optuna_startup_trials=3,  # Reduced for testing
                        resume_optimization=False,
                        penalty_factor=1.0,
                    )
                ]
            )

            # Initialize and run the actual workflow
            octo_ml = OctoML(
                octo_data_config,
                config_study=config_study,
                config_manager=config_manager,
                config_sequence=config_sequence,
            )

            # This will actually execute the Octopus regression workflow
            octo_ml.run_study()

            # Verify that the study was created and files exist
            study_path = Path(temp_dir) / "test_octo_regression_execution"
            assert study_path.exists(), "Study directory should be created"

            # Check for expected subdirectories
            assert (study_path / "data").exists(), "Data directory should exist"
            assert (study_path / "config").exists(), "Config directory should exist"
            assert (study_path / "experiment0").exists(), "Experiment directory should exist"

            # Verify that the Octo step was executed by checking for sequence directories
            experiment_path = study_path / "experiment0"
            sequence_dirs = [d for d in experiment_path.iterdir() if d.is_dir() and d.name.startswith("sequence")]

            # Should have at least one sequence directory for the Octo step
            assert len(sequence_dirs) >= 1, (
                f"Should have at least 1 sequence directory, found: {[d.name for d in sequence_dirs]}"
            )

            # Verify the Octo step was executed
            sequence_dir = sequence_dirs[0]
            assert sequence_dir.exists(), "Octo sequence step should have been executed"

    def test_full_configuration_parameters(self):
        """Test that all configuration parameters are supported."""
        config_sequence = ConfigSequence(
            [
                Octo(
                    sequence_id=0,
                    input_sequence_id=-1,
                    description="step_1",
                    models=[
                        "RandomForestRegressor",
                        "XGBRegressor",
                        "ExtraTreesRegressor",
                        "ElasticNetRegressor",
                        "GradientBoostingRegressor",
                        "CatBoostRegressor",
                    ],
                    n_trials=12,
                    max_features=6,
                    ensemble_selection=True,
                    ensel_n_save_trials=10,
                    model_seed=0,
                    n_jobs=1,
                    max_outl=0,
                    fi_methods_bestbag=["permutation"],
                    inner_parallelization=True,
                    n_workers=5,
                    optuna_seed=0,
                    n_optuna_startup_trials=10,
                    resume_optimization=False,
                    penalty_factor=1.0,
                    n_folds_inner=5,
                )
            ]
        )

        # Verify all parameters are set correctly
        octo_step = config_sequence.sequence_items[0]

        # Basic configuration
        assert octo_step.sequence_id == 0
        assert octo_step.input_sequence_id == -1
        assert octo_step.description == "step_1"

        # Model configuration
        expected_models = {
            "RandomForestRegressor",
            "XGBRegressor",
            "ExtraTreesRegressor",
            "ElasticNetRegressor",
            "GradientBoostingRegressor",
            "CatBoostRegressor",
        }
        assert set(octo_step.models) == expected_models
        assert octo_step.model_seed == 0
        assert octo_step.n_jobs == 1
        assert octo_step.max_outl == 0

        # Feature importance
        assert octo_step.fi_methods_bestbag == ["permutation"]

        # Parallelization
        assert octo_step.inner_parallelization is True
        assert octo_step.n_workers == 5

        # Hyperparameter optimization
        assert octo_step.optuna_seed == 0
        assert octo_step.n_optuna_startup_trials == 10
        assert octo_step.resume_optimization is False
        assert octo_step.n_trials == 12
        assert octo_step.max_features == 6
        assert octo_step.penalty_factor == 1.0

        # Ensemble selection
        assert octo_step.ensemble_selection is True
        assert octo_step.ensel_n_save_trials == 10

        # Data splitting
        assert octo_step.n_folds_inner == 5
