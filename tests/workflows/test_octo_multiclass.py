"""Test workflow for Octopus multiclass classification using Wine dataset."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.modules import Octo


class TestOctoMulticlass:
    """Test suite for Octopus multiclass classification workflow."""

    @pytest.fixture
    def wine_dataset(self):
        """Create synthetic multiclass dataset for testing (faster than Wine dataset)."""
        # Create synthetic multiclass dataset with reduced size for faster testing
        X, y = make_classification(
            n_samples=30,
            n_features=5,
            n_informative=3,
            n_redundant=2,
            n_classes=3,
            random_state=42,
        )

        # Create DataFrame similar to Wine dataset structure
        feature_names = [f"feature_{i}" for i in range(5)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        df = df.reset_index()

        # Create mock wine object for compatibility
        class MockWine:
            target_names = ["class_0", "class_1", "class_2"]

        wine = MockWine()

        return df, feature_names, wine

    @pytest.fixture
    def octo_data_config(self, wine_dataset):
        """Create OctoData configuration for testing."""
        df, features, wine = wine_dataset

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
        """Create ConfigStudy for multiclass testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            return ConfigStudy(
                name="test_octo_multiclass",
                ml_type="multiclass",
                target_metric="AUCROC_MACRO",
                metrics=["AUCROC_MACRO", "AUCROC_WEIGHTED", "ACCBAL_MC"],
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

    def test_wine_dataset_loading(self, wine_dataset):
        """Test that the synthetic dataset is loaded correctly."""
        df, features, wine = wine_dataset

        # Verify dataset structure
        assert isinstance(df, pd.DataFrame)
        assert "target" in df.columns
        assert "index" in df.columns
        assert len(features) == 5  # Synthetic dataset has 5 features
        assert df.shape[0] == 30  # Synthetic dataset has 30 samples

        # Verify target values are multiclass (0, 1, 2)
        unique_targets = df["target"].unique()
        assert len(unique_targets) == 3
        assert set(unique_targets) == {0, 1, 2}

        # Verify all feature columns exist in dataframe
        for feature in features:
            assert feature in df.columns

        # Verify no missing values
        assert not df[features].isnull().any().any()
        assert not df["target"].isnull().any()

        # Verify target distribution
        target_counts = df["target"].value_counts().sort_index()
        assert len(target_counts) == 3
        assert all(count > 0 for count in target_counts.values)

        # Verify target names
        assert len(wine.target_names) == 3
        assert all(isinstance(name, str) for name in wine.target_names)

    def test_octo_data_configuration(self, octo_data_config):
        """Test OctoData configuration for synthetic dataset."""
        # Verify OctoData configuration
        assert octo_data_config.target_columns == ["target"]
        assert len(octo_data_config.feature_columns) == 5
        assert octo_data_config.sample_id == "index"
        assert octo_data_config.datasplit_type == "sample"
        assert octo_data_config.stratification_column == "target"

        # Verify data integrity
        assert octo_data_config.data is not None
        assert isinstance(octo_data_config.data, pd.DataFrame)

    def test_multiclass_sequence_configuration(self):
        """Test that multiclass Octo sequence can be properly configured."""
        config_sequence = ConfigSequence(
            [
                Octo(
                    description="step_1_octo_multiclass",
                    sequence_id=0,
                    input_sequence_id=-1,
                    load_sequence_item=False,
                    n_folds_inner=5,
                    models=[
                        "ExtraTreesClassifier",
                        "RandomForestClassifier",
                        "XGBClassifier",
                        "CatBoostClassifier",
                    ],
                    model_seed=0,
                    n_jobs=1,
                    max_outl=0,
                    fi_methods_bestbag=["permutation"],
                    inner_parallelization=True,
                    n_workers=5,
                    n_trials=20,
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
        assert octo_step.description == "step_1_octo_multiclass"
        assert octo_step.n_folds_inner == 5
        assert set(octo_step.models) == {
            "ExtraTreesClassifier",
            "RandomForestClassifier",
            "XGBClassifier",
            "CatBoostClassifier",
        }
        assert octo_step.model_seed == 0
        assert octo_step.n_jobs == 1
        assert octo_step.max_outl == 0
        assert octo_step.fi_methods_bestbag == ["permutation"]
        assert octo_step.inner_parallelization is True
        assert octo_step.n_workers == 5
        assert octo_step.n_trials == 20

    def test_multiclass_study_configuration(self, config_study):
        """Test ConfigStudy configuration for multiclass."""
        # Verify multiclass-specific configuration
        assert config_study.ml_type == "multiclass"
        assert config_study.target_metric == "AUCROC_MACRO"
        assert set(config_study.metrics) == {"AUCROC_MACRO", "AUCROC_WEIGHTED", "ACCBAL_MC"}
        assert config_study.datasplit_seed_outer == 1234
        assert config_study.n_folds_outer == 3
        assert config_study.start_with_empty_study is True
        assert config_study.silently_overwrite_study is True
        assert config_study.ignore_data_health_warning is True

    @patch("octopus.ml.OctoML.run_study")
    def test_multiclass_workflow_initialization(self, mock_run_study, octo_data_config, config_study, config_manager):
        """Test that the multiclass workflow can be initialized and configured properly."""
        # Create the sequence configuration (simplified for testing)
        config_sequence = ConfigSequence(
            [
                Octo(
                    description="step_1_octo_multiclass",
                    sequence_id=0,
                    input_sequence_id=-1,
                    n_folds_inner=3,  # Reduced for testing
                    models=["ExtraTreesClassifier"],
                    model_seed=0,
                    n_jobs=1,
                    n_trials=10,  # Reduced for testing
                    fi_methods_bestbag=["permutation"],
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
        "model", ["ExtraTreesClassifier", "RandomForestClassifier", "XGBClassifier", "CatBoostClassifier"]
    )
    def test_multiclass_single_model_configuration(self, model):
        """Test configuration with different single multiclass models."""
        config_sequence = ConfigSequence(
            [
                Octo(
                    description="step_1_octo_multiclass",
                    sequence_id=0,
                    input_sequence_id=-1,
                    models=[model],
                    n_trials=5,  # Minimal for testing
                    n_folds_inner=3,
                )
            ]
        )

        octo_step = config_sequence.sequence_items[0]
        assert octo_step.models == [model]

    def test_multiclass_multiple_models_configuration(self):
        """Test configuration with multiple multiclass models."""
        models = ["ExtraTreesClassifier", "RandomForestClassifier", "XGBClassifier", "CatBoostClassifier"]
        config_sequence = ConfigSequence(
            [
                Octo(
                    description="step_1_octo_multiclass",
                    sequence_id=0,
                    input_sequence_id=-1,
                    models=models,
                    n_trials=10,
                    n_folds_inner=3,
                )
            ]
        )

        octo_step = config_sequence.sequence_items[0]
        assert set(octo_step.models) == set(models)

    def test_multiclass_metrics_configuration(self):
        """Test multiclass-specific metrics configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_study = ConfigStudy(
                name="test_multiclass_metrics",
                ml_type="multiclass",
                target_metric="AUCROC_MACRO",
                metrics=["AUCROC_MACRO", "AUCROC_WEIGHTED", "ACCBAL_MC"],
                path=temp_dir,
            )

            # Verify multiclass metrics
            assert config_study.target_metric == "AUCROC_MACRO"
            assert "AUCROC_MACRO" in config_study.metrics
            assert "AUCROC_WEIGHTED" in config_study.metrics
            assert "ACCBAL_MC" in config_study.metrics

    def test_feature_importance_configuration(self):
        """Test feature importance method configuration for multiclass."""
        fi_methods = ["permutation"]
        config_sequence = ConfigSequence(
            [
                Octo(
                    description="step_1_octo_multiclass",
                    sequence_id=0,
                    input_sequence_id=-1,
                    models=["ExtraTreesClassifier"],
                    fi_methods_bestbag=fi_methods,
                    n_trials=5,
                )
            ]
        )

        octo_step = config_sequence.sequence_items[0]
        assert octo_step.fi_methods_bestbag == fi_methods

    def test_hyperparameter_optimization_configuration(self):
        """Test hyperparameter optimization configuration for multiclass."""
        config_sequence = ConfigSequence(
            [
                Octo(
                    description="step_1_octo_multiclass",
                    sequence_id=0,
                    input_sequence_id=-1,
                    models=["ExtraTreesClassifier"],
                    n_trials=25,
                    n_folds_inner=5,
                )
            ]
        )

        octo_step = config_sequence.sequence_items[0]
        assert octo_step.n_trials == 25
        assert octo_step.n_folds_inner == 5

    @pytest.mark.slow
    def test_multiclass_workflow_actual_execution(self, octo_data_config):
        """Test that the multiclass workflow actually runs end-to-end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal configuration for actual execution
            config_study = ConfigStudy(
                name="test_multiclass_execution",
                ml_type="multiclass",
                target_metric="AUCROC_MACRO",
                metrics=["AUCROC_MACRO", "AUCROC_WEIGHTED", "ACCBAL_MC"],
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
            config_sequence = ConfigSequence(
                [
                    Octo(
                        description="step_1_octo_multiclass",
                        sequence_id=0,
                        input_sequence_id=-1,
                        load_sequence_item=False,
                        n_folds_inner=3,  # Reduced for testing
                        models=["ExtraTreesClassifier"],  # Single model for speed
                        model_seed=0,
                        n_jobs=1,
                        max_outl=0,
                        fi_methods_bestbag=["permutation"],
                        inner_parallelization=True,
                        n_workers=3,  # Match n_folds_inner
                        n_trials=12,  # Reduced for testing
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

            # This will actually execute the multiclass workflow
            octo_ml.run_study()

            # Verify that the study was created and files exist
            study_path = Path(temp_dir) / "test_multiclass_execution"
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
            assert sequence_dir.exists(), "Multiclass Octo sequence step should have been executed"

    def test_synthetic_dataset_properties(self, wine_dataset):
        """Test specific properties of the synthetic dataset."""
        df, features, wine = wine_dataset

        # Test feature names
        expected_features = ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]
        assert set(features) == set(expected_features)

        # Test target distribution (synthetic dataset with 30 samples)
        target_counts = df["target"].value_counts().sort_index()
        assert len(target_counts) == 3
        assert sum(target_counts.values) == 30  # Total 30 samples
        assert all(count > 0 for count in target_counts.values)

        # Test that all features are numeric
        for feature in features:
            assert pd.api.types.is_numeric_dtype(df[feature]), f"Feature {feature} should be numeric"

    def test_full_multiclass_configuration_parameters(self):
        """Test that all configuration parameters from the multiclass workflow are supported."""
        config_sequence = ConfigSequence(
            [
                Octo(
                    description="step_1_octo_multiclass",
                    sequence_id=0,
                    input_sequence_id=-1,
                    load_sequence_item=False,
                    n_folds_inner=5,
                    models=[
                        "ExtraTreesClassifier",
                        "RandomForestClassifier",
                        "XGBClassifier",
                        "CatBoostClassifier",
                    ],
                    model_seed=0,
                    n_jobs=1,
                    max_outl=0,
                    fi_methods_bestbag=["permutation"],
                    inner_parallelization=True,
                    n_workers=5,
                    n_trials=20,
                )
            ]
        )

        # Verify all parameters are set correctly
        octo_step = config_sequence.sequence_items[0]

        # Basic configuration
        assert octo_step.description == "step_1_octo_multiclass"
        assert octo_step.sequence_id == 0
        assert octo_step.input_sequence_id == -1
        assert octo_step.load_sequence_item is False

        # Data splitting
        assert octo_step.n_folds_inner == 5

        # Model configuration
        assert set(octo_step.models) == {
            "ExtraTreesClassifier",
            "RandomForestClassifier",
            "XGBClassifier",
            "CatBoostClassifier",
        }
        assert octo_step.model_seed == 0
        assert octo_step.n_jobs == 1
        assert octo_step.max_outl == 0

        # Feature importance
        assert octo_step.fi_methods_bestbag == ["permutation"]

        # Parallelization
        assert octo_step.inner_parallelization is True
        assert octo_step.n_workers == 5

        # Hyperparameter optimization
        assert octo_step.n_trials == 20

    def test_multiclass_target_metric_options(self):
        """Test different target metrics suitable for multiclass classification."""
        target_metrics = ["AUCROC_MACRO", "AUCROC_WEIGHTED", "ACCBAL_MC"]

        for target_metric in target_metrics:
            with tempfile.TemporaryDirectory() as temp_dir:
                config_study = ConfigStudy(
                    name=f"test_multiclass_{target_metric.lower()}",
                    ml_type="multiclass",
                    target_metric=target_metric,
                    metrics=["AUCROC_MACRO", "AUCROC_WEIGHTED", "ACCBAL_MC"],
                    path=temp_dir,
                )

                assert config_study.target_metric == target_metric
                assert config_study.ml_type == "multiclass"
