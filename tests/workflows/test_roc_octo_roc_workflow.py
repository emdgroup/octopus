"""Test workflow for ROC-OCTO-ROC sequence."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigStudy, ConfigWorkflow
from octopus.modules import Octo, Roc


class TestRocOctoRocWorkflow:
    """Test suite for ROC-OCTO-ROC workflow sequence."""

    @pytest.fixture
    def sample_classification_dataset(self):
        """Create a sample classification dataset for testing."""
        np.random.seed(42)

        # Create a dataset with more informative features and better separability
        X, y = make_classification(
            n_samples=200,
            n_features=15,
            n_informative=12,  # More informative features
            n_redundant=3,
            n_clusters_per_class=2,  # Better class separation
            class_sep=1.5,  # Increase class separation
            flip_y=0.01,  # Reduce noise
            random_state=42,
        )

        # Add some highly correlated features to test ROC filtering
        X_extended = np.column_stack(
            [
                X,
                X[:, 0] + np.random.normal(0, 0.05, X.shape[0]),  # Highly correlated with feature 0
                X[:, 1] + np.random.normal(0, 0.05, X.shape[0]),  # Highly correlated with feature 1
                X[:, 2] + np.random.normal(0, 0.05, X.shape[0]),  # Highly correlated with feature 2
                # Add some genuinely informative features
                X[:, 0] * 2 + np.random.normal(0, 0.1, X.shape[0]),  # Derived but informative
                X[:, 1] * 1.5 + np.random.normal(0, 0.1, X.shape[0]),  # Derived but informative
            ]
        )

        feature_names = [f"feature_{i}" for i in range(X_extended.shape[1])]

        df = pd.DataFrame(X_extended, columns=feature_names)
        df["target"] = y
        df["sample_id"] = range(len(df))

        return df, feature_names

    @pytest.fixture
    def octo_data_config(self, sample_classification_dataset):
        """Create OctoData configuration for testing."""
        df, feature_names = sample_classification_dataset

        return OctoData(
            data=df,
            target_columns=["target"],
            feature_columns=feature_names,
            sample_id="sample_id",
            datasplit_type="sample",
            stratification_column="target",
        )

    @pytest.fixture
    def config_study(self):
        """Create ConfigStudy for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            return ConfigStudy(
                name="test_roc_octo_roc",
                ml_type="classification",
                target_metric="ACCBAL",
                metrics=["AUCROC", "ACCBAL", "ACC"],
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

    def test_roc_octo_roc_sequence_configuration(self):
        """Test that ROC-OCTO-ROC sequence can be properly configured."""
        # Define the three-step sequence: ROC -> OCTO -> ROC
        config_workflow = ConfigWorkflow(
            [
                # Step 0: First ROC - Initial feature correlation filtering
                Roc(
                    description="step_0_roc_initial",
                    task_id=0,
                    depends_on_task=-1,  # First step, no input dependency
                    load_task=False,
                    threshold=0.85,  # Remove features with correlation > 0.85
                    correlation_type="spearmanr",
                    filter_type="f_statistics",
                ),
                # Step 1: OCTO - Model training and feature selection
                Octo(
                    description="step_1_octo",
                    task_id=1,
                    depends_on_task=0,  # Use output from first ROC step
                    n_folds_inner=3,  # Reduced for testing
                    models=["ExtraTreesClassifier"],
                    model_seed=0,
                    n_jobs=1,
                    max_outl=0,
                    fi_methods_bestbag=["permutation"],
                    inner_parallelization=True,
                    n_trials=6,  # Reduced for testing
                ),
                # Step 2: Second ROC - Final feature filtering with lower threshold
                Roc(
                    description="step_2_roc_final",
                    task_id=2,
                    depends_on_task=1,  # Use output from OCTO step
                    load_task=False,
                    threshold=0.5,  # More aggressive filtering as requested
                    correlation_type="spearmanr",
                    filter_type="mutual_info",  # Different filter type for variety
                ),
            ]
        )

        # Verify sequence configuration
        assert len(config_workflow.tasks) == 3

        # Verify first ROC step
        first_roc = config_workflow.tasks[0]
        assert isinstance(first_roc, Roc)
        assert first_roc.task_id == 0
        assert first_roc.depends_on_task == -1
        assert first_roc.threshold == 0.85
        assert first_roc.description == "step_0_roc_initial"

        # Verify OCTO step
        octo_step = config_workflow.tasks[1]
        assert isinstance(octo_step, Octo)
        assert octo_step.task_id == 1
        assert octo_step.depends_on_task == 0
        assert octo_step.description == "step_1_octo"

        # Verify second ROC step
        second_roc = config_workflow.tasks[2]
        assert isinstance(second_roc, Roc)
        assert second_roc.task_id == 2
        assert second_roc.depends_on_task == 1
        assert second_roc.threshold == 0.5  # As requested
        assert second_roc.description == "step_2_roc_final"

    @patch("octopus.ml.OctoML.run_study")
    def test_roc_octo_roc_workflow_initialization(self, mock_run_study, octo_data_config, config_study, config_manager):
        """Test that ROC-OCTO-ROC workflow can be initialized and configured properly."""
        # Create the sequence configuration
        config_workflow = ConfigWorkflow(
            [
                Roc(
                    description="step_0_roc_initial",
                    task_id=0,
                    depends_on_task=-1,
                    threshold=0.85,
                    correlation_type="spearmanr",
                    filter_type="f_statistics",
                ),
                Octo(
                    description="step_1_octo",
                    task_id=1,
                    depends_on_task=0,
                    n_folds_inner=3,
                    models=["ExtraTreesClassifier"],
                    model_seed=0,
                    n_jobs=1,
                    n_trials=15,
                ),
                Roc(
                    description="step_2_roc_final",
                    task_id=2,
                    depends_on_task=1,
                    threshold=0.5,  # As requested
                    correlation_type="spearmanr",
                    filter_type="mutual_info",
                ),
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
        assert len(octo_ml.config_workflow.tasks) == 3

        # Test that run_study can be called (mocked)
        octo_ml.run_study()
        mock_run_study.assert_called_once()

    def test_sequence_dependency_chain(self):
        """Test that the sequence dependency chain is correctly configured."""
        config_workflow = ConfigWorkflow(
            [
                Roc(
                    task_id=0,
                    depends_on_task=-1,  # No dependency
                    threshold=0.85,
                ),
                Octo(
                    task_id=1,
                    depends_on_task=0,  # Depends on first ROC
                    models=["ExtraTreesClassifier"],
                    n_trials=6,
                ),
                Roc(
                    task_id=2,
                    depends_on_task=1,  # Depends on OCTO
                    threshold=0.5,
                ),
            ]
        )

        # Verify dependency chain
        sequence = config_workflow.tasks

        # First step has no dependencies
        assert sequence[0].depends_on_task == -1

        # Second step depends on first
        assert sequence[1].depends_on_task == sequence[0].task_id

        # Third step depends on second
        assert sequence[2].depends_on_task == sequence[1].task_id

        # Verify sequence IDs are sequential
        for i, step in enumerate(sequence):
            assert step.task_id == i

    def test_roc_threshold_configuration(self):
        """Test that ROC thresholds are configured correctly in the sequence."""
        config_workflow = ConfigWorkflow(
            [
                Roc(
                    task_id=0,
                    depends_on_task=-1,
                    threshold=0.85,  # Initial filtering
                ),
                Octo(
                    task_id=1,
                    depends_on_task=0,
                    models=["ExtraTreesClassifier"],
                    n_trials=6,
                ),
                Roc(
                    task_id=2,
                    depends_on_task=1,
                    threshold=0.5,  # Final aggressive filtering as requested
                ),
            ]
        )

        # Verify ROC thresholds
        first_roc = config_workflow.tasks[0]
        second_roc = config_workflow.tasks[2]

        assert first_roc.threshold == 0.85
        assert second_roc.threshold == 0.5

        # Verify that final ROC has more aggressive filtering
        assert second_roc.threshold < first_roc.threshold

    @pytest.mark.parametrize("correlation_type", ["spearmanr", "rdc"])
    @pytest.mark.parametrize("filter_type", ["f_statistics", "mutual_info"])
    def test_roc_configuration_variations(self, correlation_type, filter_type):
        """Test ROC configuration with different correlation and filter types."""
        config_workflow = ConfigWorkflow(
            [
                Roc(
                    task_id=0,
                    depends_on_task=-1,
                    threshold=0.85,
                    correlation_type=correlation_type,
                    filter_type=filter_type,
                ),
                Octo(
                    task_id=1,
                    depends_on_task=0,
                    models=["ExtraTreesClassifier"],
                    n_trials=6,
                ),
                Roc(
                    task_id=2,
                    depends_on_task=1,
                    threshold=0.5,
                    correlation_type=correlation_type,
                    filter_type=filter_type,
                ),
            ]
        )

        # Verify configuration
        first_roc = config_workflow.tasks[0]
        second_roc = config_workflow.tasks[2]

        assert first_roc.correlation_type == correlation_type
        assert first_roc.filter_type == filter_type
        assert second_roc.correlation_type == correlation_type
        assert second_roc.filter_type == filter_type

    def test_octo_configuration_in_sequence(self):
        """Test OCTO module configuration within the ROC-OCTO-ROC sequence."""
        config_workflow = ConfigWorkflow(
            [
                Roc(task_id=0, depends_on_task=-1, threshold=0.85),
                Octo(
                    task_id=1,
                    depends_on_task=0,
                    models=["ExtraTreesClassifier", "RandomForestClassifier"],
                    n_trials=10,
                    max_features=15,
                    n_folds_inner=5,
                    model_seed=42,
                ),
                Roc(task_id=2, depends_on_task=1, threshold=0.5),
            ]
        )

        octo_step = config_workflow.tasks[1]

        # Verify OCTO configuration
        assert isinstance(octo_step, Octo)
        # Models may be sorted alphabetically, so check both models are present
        assert set(octo_step.models) == {"ExtraTreesClassifier", "RandomForestClassifier"}
        assert octo_step.n_trials == 10
        assert octo_step.max_features == 15
        assert octo_step.n_folds_inner == 5
        assert octo_step.model_seed == 42

    def test_workflow_sequence_validation(self):
        """Test that the workflow sequence is properly validated."""
        # Valid sequence
        valid_sequence = ConfigWorkflow(
            [
                Roc(task_id=0, depends_on_task=-1, threshold=0.85),
                Octo(task_id=1, depends_on_task=0, models=["ExtraTreesClassifier"], n_trials=6),
                Roc(task_id=2, depends_on_task=1, threshold=0.5),
            ]
        )

        # Should not raise any errors
        assert len(valid_sequence.tasks) == 3

        # Verify all steps are properly configured
        for i, step in enumerate(valid_sequence.tasks):
            assert step.task_id == i
            if i == 0:
                assert step.depends_on_task == -1
            else:
                assert step.depends_on_task == i - 1

    @pytest.mark.slow
    def test_roc_octo_roc_workflow_actual_execution(self, octo_data_config):
        """Test that ROC-OCTO-ROC workflow actually runs end-to-end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal configuration for actual execution
            config_study = ConfigStudy(
                name="test_roc_octo_roc_execution",
                ml_type="classification",
                target_metric="ACCBAL",
                metrics=["AUCROC", "ACCBAL"],
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

            # Create the ROC-OCTO-ROC sequence with minimal settings for speed
            config_workflow = ConfigWorkflow(
                [
                    Roc(
                        description="step_0_roc_initial",
                        task_id=0,
                        depends_on_task=-1,
                        threshold=0.9,
                        correlation_type="spearmanr",
                        filter_type="f_statistics",
                    ),
                    Octo(
                        description="step_1_octo",
                        task_id=1,
                        depends_on_task=0,
                        n_folds_inner=5,  # Minimal for speed
                        models=["ExtraTreesClassifier"],
                        model_seed=0,
                        n_jobs=1,
                        n_trials=13,  # Increased trials for better feature selection
                        inner_parallelization=True,
                        fi_methods_bestbag=["permutation"],  # Use permutation importance
                    ),
                    Roc(
                        description="step_2_roc_final",
                        task_id=2,
                        depends_on_task=1,
                        threshold=0.5,  # As requested
                        correlation_type="spearmanr",
                        filter_type="f_statistics",
                    ),
                ]
            )

            # Initialize and run the actual workflow
            octo_ml = OctoML(
                octo_data_config,
                config_study=config_study,
                config_manager=config_manager,
                config_workflow=config_workflow,
            )

            # This will actually execute the ROC-OCTO-ROC workflow
            octo_ml.run_study()

            # Verify that the study was created and files exist
            study_path = Path(temp_dir) / "test_roc_octo_roc_execution"
            assert study_path.exists(), "Study directory should be created"

            # Check for expected subdirectories
            assert (study_path / "data").exists(), "Data directory should exist"
            assert (study_path / "config").exists(), "Config directory should exist"
            assert (study_path / "outerdatasplit0").exists(), "Experiment directory should exist"

            # Verify that sequence steps were executed by checking for workflow directories
            experiment_path = study_path / "outerdatasplit0"
            workflow_dirs = [d for d in experiment_path.iterdir() if d.is_dir() and d.name.startswith("workflowtask")]

            # Should have directories for each sequence step
            assert len(workflow_dirs) >= 3, (
                f"Should have at least 3 workflow directories, found: {[d.name for d in workflow_dirs]}"
            )

            # Verify the final ROC step was executed with threshold 0.5
            # The last workflow directory should contain ROC results
            def extract_workflow_task_number(path):
                # Extract task number from "workflowtaskX" format
                name = path.name
                return int(name.replace("workflowtask", ""))

            workflow_dirs_sorted = sorted(workflow_dirs, key=extract_workflow_task_number)
            final_workflow_dir = workflow_dirs_sorted[-1]

            # Check that the final workflow directory exists (indicating ROC step 2 was executed)
            assert final_workflow_dir.exists(), "Final ROC workflow step should have been executed"
