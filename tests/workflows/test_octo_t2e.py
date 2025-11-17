"""Test workflow for Octopus time-to-event (survival analysis) example."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigStudy, ConfigWorkflow
from octopus.modules import Octo


class TestOctoTimeToEvent:
    """Test suite for Octopus time-to-event workflow."""

    @pytest.fixture
    def survival_dataset(self):
        """Create synthetic time-to-event dataset for testing."""
        np.random.seed(42)

        # Create synthetic survival dataset with sufficient size for concordance index calculation
        n_samples = 100
        n_features = 5

        # Generate features
        X = np.random.randn(n_samples, n_features)
        features = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=features)

        # Generate survival times (duration) - exponential distribution
        # Higher feature values lead to longer survival times
        risk_score = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
        baseline_hazard = 0.1
        duration = np.random.exponential(scale=1.0 / (baseline_hazard * np.exp(risk_score)))

        # Generate censoring times
        censoring_time = np.random.exponential(scale=15, size=n_samples)

        # Observed time is minimum of event time and censoring time
        observed_time = np.minimum(duration, censoring_time)

        # Event indicator: 1 if event occurred, 0 if censored
        event = (duration <= censoring_time).astype(int)

        # Add to dataframe
        df["duration"] = observed_time
        df["event"] = event
        df = df.reset_index()

        return df, features

    @pytest.fixture
    def octo_data_config(self, survival_dataset):
        """Create OctoData configuration for time-to-event testing."""
        df, features = survival_dataset

        return OctoData(
            data=df,
            target_columns=["duration", "event"],
            feature_columns=features,
            sample_id="index",
            datasplit_type="sample",
            target_assignments={"duration": "duration", "event": "event"},
        )

    @pytest.fixture
    def config_study(self):
        """Create ConfigStudy for time-to-event testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            return ConfigStudy(
                name="test_octo_t2e",
                ml_type="timetoevent",
                target_metric="CI",
                metrics=["CI"],
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

    def test_survival_dataset_loading(self, survival_dataset):
        """Test that the survival dataset is loaded correctly."""
        df, features = survival_dataset

        # Verify dataset structure
        assert isinstance(df, pd.DataFrame)
        assert "duration" in df.columns
        assert "event" in df.columns
        assert "index" in df.columns
        assert len(features) == 5  # Synthetic dataset has 5 features
        assert df.shape[0] == 100  # Synthetic dataset has 100 samples

        # Verify duration values are positive
        assert (df["duration"] > 0).all()
        assert df["duration"].dtype in ["float64", "int64"]

        # Verify event values are binary (0 or 1)
        assert df["event"].dtype in ["int64", "int32", "bool"]
        assert set(df["event"].unique()).issubset({0, 1})

        # Verify all feature columns exist in dataframe
        for feature in features:
            assert feature in df.columns

        # Verify no missing values
        assert not df[features].isnull().any().any()
        assert not df["duration"].isnull().any()
        assert not df["event"].isnull().any()

        # Verify we have both censored and uncensored observations
        assert df["event"].sum() > 0, "Should have at least one event"
        assert (df["event"] == 0).sum() > 0, "Should have at least one censored observation"

    def test_octo_data_configuration(self, octo_data_config):
        """Test OctoData configuration for survival dataset."""
        # Verify OctoData configuration
        assert octo_data_config.target_columns == ["duration", "event"]
        assert len(octo_data_config.feature_columns) == 5
        assert octo_data_config.sample_id == "index"
        assert octo_data_config.datasplit_type == "sample"

        # Verify data integrity
        assert octo_data_config.data is not None
        assert isinstance(octo_data_config.data, pd.DataFrame)

    def test_octo_sequence_configuration(self):
        """Test that Octo sequence can be properly configured for time-to-event."""
        config_workflow = ConfigWorkflow(
            [
                Octo(
                    task_id=0,
                    depends_on_task=-1,
                    description="step_1",
                    models=["ExtraTreesSurv"],
                    n_trials=12,
                    max_features=6,
                    ensemble_selection=True,
                    ensel_n_save_trials=10,
                )
            ]
        )

        # Verify sequence configuration
        assert len(config_workflow.tasks) == 1

        # Verify Octo step configuration
        octo_step = config_workflow.tasks[0]
        assert isinstance(octo_step, Octo)
        assert octo_step.task_id == 0
        assert octo_step.depends_on_task == -1
        assert octo_step.description == "step_1"
        assert octo_step.n_trials == 12
        assert octo_step.max_features == 6
        assert octo_step.ensemble_selection is True
        assert octo_step.ensel_n_save_trials == 10

        # Verify models are configured correctly
        assert octo_step.models == ["ExtraTreesSurv"]

    @patch("octopus.ml.OctoML.run_study")
    def test_workflow_initialization(self, mock_run_study, octo_data_config, config_study, config_manager):
        """Test that the time-to-event workflow can be initialized and configured properly."""
        # Create the sequence configuration
        config_workflow = ConfigWorkflow(
            [
                Octo(
                    task_id=0,
                    depends_on_task=-1,
                    description="step_1",
                    models=["ExtraTreesSurv"],
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
            config_workflow=config_workflow,
        )

        # Verify initialization
        assert octo_ml.data is octo_data_config
        assert octo_ml.config_study == config_study
        assert octo_ml.config_manager == config_manager
        assert octo_ml.config_workflow == config_workflow

        # Verify sequence structure
        assert len(octo_ml.config_workflow.tasks) == 1

        # Test that run_study can be called (mocked)
        octo_ml.run_study()
        mock_run_study.assert_called_once()

    def test_single_model_configuration(self):
        """Test configuration with ExtraTreesSurv model."""
        config_workflow = ConfigWorkflow(
            [
                Octo(
                    task_id=0,
                    depends_on_task=-1,
                    description="step_1",
                    models=["ExtraTreesSurv"],
                    n_trials=12,
                    max_features=6,
                    ensemble_selection=True,
                    ensel_n_save_trials=10,
                )
            ]
        )

        octo_step = config_workflow.tasks[0]
        assert octo_step.models == ["ExtraTreesSurv"]
        assert octo_step.n_trials == 12
        assert octo_step.max_features == 6
        assert octo_step.ensemble_selection is True
        assert octo_step.ensel_n_save_trials == 10

    def test_ensemble_selection_configuration(self):
        """Test ensemble selection configuration."""
        config_workflow = ConfigWorkflow(
            [
                Octo(
                    task_id=0,
                    depends_on_task=-1,
                    description="step_1",
                    models=["ExtraTreesSurv"],
                    n_trials=12,
                    max_features=6,
                    ensemble_selection=True,
                    ensel_n_save_trials=10,
                )
            ]
        )

        octo_step = config_workflow.tasks[0]
        assert octo_step.ensemble_selection is True
        assert octo_step.ensel_n_save_trials == 10

    def test_hyperparameter_optimization_configuration(self):
        """Test hyperparameter optimization configuration."""
        config_workflow = ConfigWorkflow(
            [
                Octo(
                    task_id=0,
                    depends_on_task=-1,
                    description="step_1",
                    models=["ExtraTreesSurv"],
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

        octo_step = config_workflow.tasks[0]

        # Verify time-to-event model is included
        assert "ExtraTreesSurv" in octo_step.models
        # Verify key parameters
        assert octo_step.n_trials == 12
        assert octo_step.max_features == 6
        assert octo_step.ensemble_selection is True
        assert octo_step.ensel_n_save_trials == 10
        assert octo_step.optuna_seed == 42
        assert octo_step.n_optuna_startup_trials == 5
        assert octo_step.penalty_factor == 1.5

    @pytest.mark.slow
    def test_octo_timetoevent_actual_execution(self, octo_data_config):
        """Test that the Octopus time-to-event workflow actually runs end-to-end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal configuration for actual execution
            config_study = ConfigStudy(
                name="test_octo_t2e_execution",
                ml_type="timetoevent",
                target_metric="CI",
                metrics=["CI"],
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
            config_workflow = ConfigWorkflow(
                [
                    Octo(
                        task_id=0,
                        depends_on_task=-1,
                        description="step_1",
                        models=["ExtraTreesSurv"],
                        n_trials=12,
                        max_features=6,
                        ensemble_selection=True,
                        ensel_n_save_trials=10,
                        model_seed=0,
                        n_jobs=1,
                        fi_methods_bestbag=["shap"],  # Use SHAP for feature importance
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
                config_workflow=config_workflow,
            )

            # This will actually execute the Octopus time-to-event workflow
            octo_ml.run_study()

            # Verify that the study was created and files exist
            study_path = Path(temp_dir) / "test_octo_t2e_execution"
            assert study_path.exists(), "Study directory should be created"

            # Check for expected subdirectories
            assert (study_path / "data").exists(), "Data directory should exist"
            assert (study_path / "config").exists(), "Config directory should exist"
            assert (study_path / "experiment0").exists(), "Experiment directory should exist"

            # Verify that the Octo step was executed by checking for workflow directories
            experiment_path = study_path / "experiment0"
            workflow_dirs = [d for d in experiment_path.iterdir() if d.is_dir() and d.name.startswith("workflowtask")]

            # Should have at least one workflow directory for the Octo step
            assert len(workflow_dirs) >= 1, (
                f"Should have at least 1 workflow directory, found: {[d.name for d in workflow_dirs]}"
            )

            # Verify the Octo step was executed
            workflow_dir = workflow_dirs[0]
            assert workflow_dir.exists(), "Octo workflow step should have been executed"

    def test_full_configuration_parameters(self):
        """Test that all configuration parameters are supported."""
        config_workflow = ConfigWorkflow(
            [
                Octo(
                    task_id=0,
                    depends_on_task=-1,
                    description="step_1",
                    models=["ExtraTreesSurv"],
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
        octo_step = config_workflow.tasks[0]

        # Basic configuration
        assert octo_step.task_id == 0
        assert octo_step.depends_on_task == -1
        assert octo_step.description == "step_1"

        # Model configuration
        assert octo_step.models == ["ExtraTreesSurv"]
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
