"""Test workflow for Octopus intro classification example."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigStudy, ConfigWorkflow
from octopus.modules import Octo


class TestOctoIntroClassification:
    """Test suite for Octopus intro classification workflow."""

    @pytest.fixture
    def breast_cancer_dataset(self):
        """Create synthetic binary classification dataset for testing (faster than breast cancer dataset)."""
        # Create synthetic binary classification dataset with reduced size for faster testing
        X, y = make_classification(
            n_samples=30,
            n_features=5,
            n_informative=3,
            n_redundant=2,
            n_classes=2,
            random_state=42,
        )

        # Create DataFrame similar to breast cancer dataset structure
        feature_names = [f"feature_{i}" for i in range(5)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        df = df.reset_index()

        return df, feature_names

    @pytest.fixture
    def octo_data_config(self, breast_cancer_dataset):
        """Create OctoData configuration for testing."""
        df, features = breast_cancer_dataset

        return OctoData(
            data=df,
            target_columns=["target"],
            feature_columns=features,
            sample_id="index",
            datasplit_type="sample",
            stratification_column="target",
        )

    @pytest.fixture
    def config_study(self):
        """Create ConfigStudy for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            return ConfigStudy(
                name="test_octo_intro_classification",
                ml_type="classification",
                target_metric="ACCBAL",
                metrics=["AUCROC", "ACCBAL", "ACC", "LOGLOSS"],
                datasplit_seed_outer=1234,
                n_folds_outer=2,  # Reduced for faster testing
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

    def test_breast_cancer_dataset_loading(self, breast_cancer_dataset):
        """Test that the breast cancer dataset is loaded correctly."""
        df, features = breast_cancer_dataset

        # Verify dataset structure
        assert isinstance(df, pd.DataFrame)
        assert "target" in df.columns
        assert "index" in df.columns
        assert len(features) == 5  # Reduced to 5 features for faster testing
        assert df.shape[0] == 30  # Reduced to 30 samples for faster testing

        # Verify target values are binary (0 and 1)
        unique_targets = df["target"].unique()
        assert len(unique_targets) == 2
        assert set(unique_targets) == {0, 1}

        # Verify all feature columns exist in dataframe
        for feature in features:
            assert feature in df.columns

        # Verify no missing values
        assert not df[features].isnull().any().any()
        assert not df["target"].isnull().any()

    def test_octo_data_configuration(self, octo_data_config):
        """Test OctoData configuration for breast cancer dataset."""
        # Verify OctoData configuration
        assert octo_data_config.target_columns == ["target"]
        assert len(octo_data_config.feature_columns) == 5  # Reduced to 5 features
        assert octo_data_config.sample_id == "index"
        assert octo_data_config.datasplit_type == "sample"
        assert octo_data_config.stratification_column == "target"

        # Verify data integrity
        assert octo_data_config.data is not None
        assert isinstance(octo_data_config.data, pd.DataFrame)

    def test_octo_sequence_configuration(self):
        """Test that Octo sequence can be properly configured."""
        config_workflow = ConfigWorkflow(
            [
                Octo(
                    description="step_1_octo",
                    task_id=0,
                    depends_on_task=-1,
                    load_task=False,
                    n_folds_inner=3,  # Reduced for faster testing
                    models=[
                        "ExtraTreesClassifier",
                        "RandomForestClassifier",
                    ],
                    fi_methods_bestbag=["permutation"],
                    inner_parallelization=True,
                    n_workers=3,  # Match n_folds_inner
                    optuna_seed=0,
                    n_optuna_startup_trials=5,  # Reduced for faster testing
                    resume_optimization=False,
                    n_trials=6,  # Reduced for faster testing
                    max_features=5,  # Reduced to match feature count
                    penalty_factor=1.0,
                    ensemble_selection=True,
                    ensel_n_save_trials=5,  # Reduced for faster testing
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
        assert octo_step.description == "step_1_octo"
        assert octo_step.n_folds_inner == 3
        assert set(octo_step.models) == {"ExtraTreesClassifier", "RandomForestClassifier"}
        assert octo_step.model_seed == 0
        assert octo_step.n_jobs == 1
        assert octo_step.max_outl == 3
        assert octo_step.fi_methods_bestbag == ["permutation"]
        assert octo_step.inner_parallelization is True
        assert octo_step.n_workers == 3
        assert octo_step.optuna_seed == 0
        assert octo_step.n_optuna_startup_trials == 5
        assert octo_step.resume_optimization is False
        assert octo_step.n_trials == 6
        assert octo_step.max_features == 5
        assert octo_step.penalty_factor == 1.0
        assert octo_step.ensemble_selection is True
        assert octo_step.ensel_n_save_trials == 5

    @patch("octopus.ml.OctoML.run_study")
    def test_workflow_initialization(self, mock_run_study, octo_data_config, config_study, config_manager):
        """Test that the classification workflow can be initialized and configured properly."""
        # Create the sequence configuration (simplified for testing)
        config_workflow = ConfigWorkflow(
            [
                Octo(
                    description="step_1_octo",
                    task_id=0,
                    depends_on_task=-1,
                    n_folds_inner=3,  # Reduced for testing
                    models=["ExtraTreesClassifier"],
                    model_seed=0,
                    n_jobs=1,
                    n_trials=5,  # Further reduced for testing
                    fi_methods_bestbag=["permutation"],
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

    @pytest.mark.parametrize("model", ["ExtraTreesClassifier", "RandomForestClassifier"])
    def test_single_model_configuration(self, model):
        """Test configuration with different single models."""
        config_workflow = ConfigWorkflow(
            [
                Octo(
                    description="step_1_octo",
                    task_id=0,
                    depends_on_task=-1,
                    models=[model],
                    n_trials=3,  # Further reduced for testing
                    n_folds_inner=3,
                )
            ]
        )

        octo_step = config_workflow.tasks[0]
        assert octo_step.models == [model]

    def test_multiple_models_configuration(self):
        """Test configuration with multiple models."""
        models = ["ExtraTreesClassifier", "RandomForestClassifier"]
        config_workflow = ConfigWorkflow(
            [
                Octo(
                    description="step_1_octo",
                    task_id=0,
                    depends_on_task=-1,
                    models=models,
                    n_trials=5,  # Reduced for testing
                    n_folds_inner=3,
                )
            ]
        )

        octo_step = config_workflow.tasks[0]
        assert set(octo_step.models) == set(models)

    def test_feature_importance_configuration(self):
        """Test feature importance method configuration."""
        fi_methods = ["permutation"]
        config_workflow = ConfigWorkflow(
            [
                Octo(
                    description="step_1_octo",
                    task_id=0,
                    depends_on_task=-1,
                    models=["ExtraTreesClassifier"],
                    fi_methods_bestbag=fi_methods,
                    n_trials=3,  # Further reduced for testing
                )
            ]
        )

        octo_step = config_workflow.tasks[0]
        assert octo_step.fi_methods_bestbag == fi_methods

    def test_ensemble_selection_configuration(self):
        """Test ensemble selection configuration."""
        config_workflow = ConfigWorkflow(
            [
                Octo(
                    description="step_1_octo",
                    task_id=0,
                    depends_on_task=-1,
                    models=["ExtraTreesClassifier", "RandomForestClassifier"],
                    ensemble_selection=True,
                    ensel_n_save_trials=15,
                    n_trials=5,  # Reduced for testing
                )
            ]
        )

        octo_step = config_workflow.tasks[0]
        assert octo_step.ensemble_selection is True
        assert octo_step.ensel_n_save_trials == 15

    def test_hyperparameter_optimization_configuration(self):
        """Test hyperparameter optimization configuration."""
        config_workflow = ConfigWorkflow(
            [
                Octo(
                    description="step_1_octo",
                    task_id=0,
                    depends_on_task=-1,
                    models=["ExtraTreesClassifier"],
                    optuna_seed=42,
                    n_optuna_startup_trials=5,
                    n_trials=5,  # Reduced for testing
                    max_features=5,  # Reduced to match feature count
                    penalty_factor=1.5,
                )
            ]
        )

        octo_step = config_workflow.tasks[0]
        assert octo_step.optuna_seed == 42
        assert octo_step.n_optuna_startup_trials == 5
        assert octo_step.n_trials == 5
        assert octo_step.max_features == 5
        assert octo_step.penalty_factor == 1.5

    @pytest.mark.slow
    def test_octo_intro_classification_actual_execution(self, octo_data_config):
        """Test that the Octopus intro classification workflow actually runs end-to-end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal configuration for actual execution
            config_study = ConfigStudy(
                name="test_octo_intro_execution",
                ml_type="classification",
                target_metric="ACCBAL",
                metrics=["AUCROC", "ACCBAL", "ACC", "LOGLOSS"],
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

            # Create the Octo sequence with minimal settings for speed
            config_workflow = ConfigWorkflow(
                [
                    Octo(
                        description="step_1_octo",
                        task_id=0,
                        depends_on_task=-1,
                        load_task=False,
                        n_folds_inner=3,  # Reduced for testing
                        models=["ExtraTreesClassifier"],  # Single model for speed
                        model_seed=0,
                        n_jobs=1,
                        max_outl=0,
                        fi_methods_bestbag=["permutation"],
                        inner_parallelization=True,
                        n_workers=2,  # Reduced for testing
                        optuna_seed=0,
                        n_optuna_startup_trials=3,  # Reduced for testing
                        resume_optimization=False,
                        n_trials=5,  # Further reduced for testing
                        max_features=5,  # Reduced to match feature count
                        penalty_factor=1.0,
                        ensemble_selection=True,
                        ensel_n_save_trials=5,  # Reduced for testing
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

            # This will actually execute the Octopus intro classification workflow
            octo_ml.run_study()

            # Verify that the study was created and files exist
            study_path = Path(temp_dir) / "test_octo_intro_execution"
            assert study_path.exists(), "Study directory should be created"

            # Check for expected subdirectories
            assert (study_path / "data").exists(), "Data directory should exist"
            assert (study_path / "config").exists(), "Config directory should exist"
            assert (study_path / "outerdatasplit0").exists(), "Experiment directory should exist"

            # Verify that the Octo step was executed by checking for workflow directories
            experiment_path = study_path / "outerdatasplit0"
            workflow_dirs = [d for d in experiment_path.iterdir() if d.is_dir() and d.name.startswith("workflowtask")]

            # Should have at least one workflow directory for the Octo step
            assert len(workflow_dirs) >= 1, (
                f"Should have at least 1 workflow directory, found: {[d.name for d in workflow_dirs]}"
            )

            # Verify the Octo step was executed
            workflow_dir = workflow_dirs[0]
            assert workflow_dir.exists(), "Octo workflow step should have been executed"

    def test_full_configuration_parameters(self):
        """Test that all configuration parameters from the original workflow are supported."""
        config_workflow = ConfigWorkflow(
            [
                Octo(
                    description="step_1_octo",
                    task_id=0,
                    depends_on_task=-1,
                    load_task=False,
                    n_folds_inner=5,
                    models=["ExtraTreesClassifier", "RandomForestClassifier"],
                    model_seed=0,
                    n_jobs=1,
                    max_outl=0,
                    fi_methods_bestbag=["permutation"],
                    inner_parallelization=True,
                    n_workers=5,
                    optuna_seed=0,
                    n_optuna_startup_trials=10,
                    resume_optimization=False,
                    n_trials=5,  # Reduced for testing
                    max_features=5,  # Reduced to match feature count
                    penalty_factor=1.0,
                    ensemble_selection=True,
                    ensel_n_save_trials=10,
                )
            ]
        )

        # Verify all parameters are set correctly
        octo_step = config_workflow.tasks[0]

        # Basic configuration
        assert octo_step.description == "step_1_octo"
        assert octo_step.task_id == 0
        assert octo_step.depends_on_task == -1
        assert octo_step.load_task is False

        # Data splitting
        assert octo_step.n_folds_inner == 5

        # Model configuration
        assert set(octo_step.models) == {"ExtraTreesClassifier", "RandomForestClassifier"}
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
        assert octo_step.n_trials == 5
        assert octo_step.max_features == 5
        assert octo_step.penalty_factor == 1.0

        # Ensemble selection
        assert octo_step.ensemble_selection is True
        assert octo_step.ensel_n_save_trials == 10
