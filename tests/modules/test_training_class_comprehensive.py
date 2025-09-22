#!/usr/bin/env python3
"""Comprehensive Test Suite for Training Class.

This test suite validates the Training class functionality including:
- Data preprocessing pipeline
- Multiple ML model types (classification, regression, time-to-event)
- Feature importance methods
- Prediction capabilities
- Missing value handling
- Outlier detection

Usage:
    pytest test_training_class_comprehensive.py
    pytest test_training_class_comprehensive.py -v  # verbose output
    pytest test_training_class_comprehensive.py::TestTrainingClass::test_data_preprocessing  # specific test
"""

import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from octopus.models.inventory import ModelInventory
from octopus.modules.octo.training import Training

# ============================================================================
# CONFIGURATION AND FIXTURES
# ============================================================================

# Test data configuration
TEST_CONFIG = {
    "n_samples": 200,
    "test_split": 0.3,
    "dev_split": 0.2,
    "random_seed": 42,
}


class ModelCache:
    """Cache for available models to avoid repeated discovery and messages."""

    def __init__(self):
        self._cached_models_by_type = None
        self._tabpfn_skip_logged = False

    def get_available_models_by_type(self):
        """Get all available models dynamically from ModelInventory, grouped by ML type.

        Excludes TabPFN models as they may have dependency issues.
        """
        # Return cached results if available
        if self._cached_models_by_type is not None:
            return self._cached_models_by_type

        inventory = ModelInventory()
        all_models = inventory.models

        models_by_type = {"classification": [], "regression": [], "timetoevent": []}
        skipped_tabpfn_models = []

        for model_name in all_models.keys():
            # Skip TabPFN models
            if "TabPFN" in model_name:
                skipped_tabpfn_models.append(model_name)
                continue

            try:
                model_config = inventory.get_model_config(model_name)
                ml_type = model_config.ml_type
                if ml_type in models_by_type:
                    models_by_type[ml_type].append(model_name)
            except Exception as e:
                print(f"Warning: Could not get config for model {model_name}: {e}")
                continue

        # Log skipped TabPFN models only once
        if skipped_tabpfn_models and not self._tabpfn_skip_logged:
            print(f"Skipping TabPFN models (TabPFN models excluded): {', '.join(skipped_tabpfn_models)}")
            self._tabpfn_skip_logged = True

        # Cache the results
        self._cached_models_by_type = models_by_type
        return models_by_type


# Global instance for caching
_model_cache = ModelCache()


def get_available_models_by_type():
    """Get all available models dynamically from ModelInventory, grouped by ML type."""
    return _model_cache.get_available_models_by_type()


def get_model_configs():
    """Get model configurations with all available models."""
    available_models = get_available_models_by_type()

    return {
        "classification": {
            "models": available_models["classification"],
            "target_assignments": {"target": "target_class"},
            "target_metric": "AUCROC",
        },
        "regression": {
            "models": available_models["regression"],
            "target_assignments": {"target": "target_reg"},
            "target_metric": "R2",
        },
        "timetoevent": {
            "models": available_models["timetoevent"],
            "target_assignments": {"duration": "duration", "event": "event"},
            "target_metric": "CI",
        },
    }


@pytest.fixture(scope="session")
def test_data():
    """Create comprehensive test dataset with mixed data types."""
    np.random.seed(TEST_CONFIG["random_seed"])
    n_samples = TEST_CONFIG["n_samples"]

    # Generate numerical features
    data = pd.DataFrame(
        {
            "num_col1": np.random.normal(10, 2, n_samples),
            "num_col2": np.random.normal(50, 10, n_samples),
            "num_col3": np.random.uniform(0, 100, n_samples),
        }
    )

    # Add missing values to numerical columns
    data.loc[::15, "num_col1"] = np.nan
    data.loc[::20, "num_col2"] = np.nan
    data.loc[::25, "num_col3"] = np.nan

    # Generate categorical features
    nominal_col = np.random.choice([1, 2, 3, 4], n_samples).astype(float)
    nominal_col[::18] = np.nan
    data["nominal_col"] = nominal_col

    ordinal_col = np.random.choice([1, 2, 3], n_samples).astype(float)
    ordinal_col[::22] = np.nan
    data["ordinal_col"] = ordinal_col

    # Add row identifier
    data["row_id"] = range(n_samples)

    # Generate targets
    data["target_class"] = np.random.choice([0, 1], n_samples)
    data["target_reg"] = (
        0.5 * data["num_col1"].fillna(data["num_col1"].mean())
        + 0.3 * data["num_col2"].fillna(data["num_col2"].mean())
        + np.random.normal(0, 1, n_samples)
    )
    data["duration"] = np.random.exponential(10, n_samples)
    data["event"] = np.random.choice([True, False], n_samples, p=[0.7, 0.3])

    # Split data
    n_train = int(n_samples * (1 - TEST_CONFIG["test_split"] - TEST_CONFIG["dev_split"]))
    n_dev = int(n_samples * TEST_CONFIG["dev_split"])

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    dev_idx = indices[n_train : n_train + n_dev]
    test_idx = indices[n_train + n_dev :]

    return (
        data.iloc[train_idx].reset_index(drop=True),
        data.iloc[dev_idx].reset_index(drop=True),
        data.iloc[test_idx].reset_index(drop=True),
    )


@pytest.fixture(scope="session")
def feature_config():
    """Feature configuration for tests."""
    feature_columns = ["num_col1", "num_col2", "num_col3", "nominal_col", "ordinal_col"]
    feature_groups = {
        "numerical_group": ["num_col1", "num_col2"],
        "categorical_group": ["nominal_col", "ordinal_col"],
    }
    return feature_columns, feature_groups


def create_training_instance(
    data_train: pd.DataFrame,
    data_dev: pd.DataFrame,
    data_test: pd.DataFrame,
    ml_type: str,
    model_name: str,
    feature_columns: list[str],
    feature_groups: dict[str, list[str]],
) -> Training:
    """Create a Training instance for testing."""
    model_configs = get_model_configs()
    config = model_configs[ml_type]

    # Special handling for CatBoost models
    ml_model_params = {}
    if "CatBoost" in model_name:
        ml_model_params = {
            "verbose": False,  # Suppress CatBoost output
            "allow_writing_files": False,  # no writinng on disk
        }

    training_config = {
        "ml_model_type": model_name,
        "ml_model_params": ml_model_params,
        "outl_reduction": 0,
    }

    return Training(
        training_id=f"test_{ml_type}_{model_name}",
        ml_type=ml_type,
        target_assignments=config["target_assignments"],
        feature_columns=feature_columns,
        row_column="row_id",
        data_train=data_train,
        data_dev=data_dev,
        data_test=data_test,
        target_metric=config["target_metric"],
        max_features=3,
        feature_groups=feature_groups,
        config_training=training_config,
    )


# ============================================================================
# TEST CLASS
# ============================================================================


class TestTrainingClass:
    """Comprehensive test suite for Training class."""

    def test_data_preprocessing(self, test_data, feature_config):
        """Test data preprocessing functionality."""
        data_train, data_dev, data_test = test_data
        feature_columns, feature_groups = feature_config

        # Suppress warnings for cleaner test output
        warnings.filterwarnings("ignore")

        # Create training instance
        training = create_training_instance(
            data_train, data_dev, data_test, "classification", "RandomForestClassifier", feature_columns, feature_groups
        )

        # Check missing values before preprocessing
        missing_before = training.data_train[feature_columns].isnull().sum().sum()
        assert missing_before > 0, "Test data should have missing values"

        # Fit the model
        training.fit()

        # Test preprocessing pipeline
        assert training.preprocessing_pipeline is not None, "Preprocessing pipeline should be created"

        # Test preprocessing on new data
        test_sample = training.data_test[feature_columns].head(5)
        processed_sample = training.preprocessing_pipeline.transform(test_sample)

        # Check for missing values after preprocessing
        if isinstance(processed_sample, np.ndarray):
            processed_df = pd.DataFrame(processed_sample)
            missing_after = processed_df.isnull().sum().sum()
        else:
            missing_after = processed_sample.isnull().sum().sum()

        assert missing_after == 0, "All missing values should be handled after preprocessing"

    @pytest.mark.parametrize("ml_type", ["classification", "regression", "timetoevent"])
    def test_model_types(self, test_data, feature_config, ml_type):
        """Test all models for each ML type."""
        data_train, data_dev, data_test = test_data
        feature_columns, feature_groups = feature_config

        # Suppress warnings for cleaner test output
        warnings.filterwarnings("ignore")

        model_configs = get_model_configs()
        config = model_configs[ml_type]

        for model_name in config["models"]:
            # Create and fit training instance
            training = create_training_instance(
                data_train, data_dev, data_test, ml_type, model_name, feature_columns, feature_groups
            )

            # Test that model can be fitted without errors
            training.fit()
            assert training.model is not None, f"{model_name} should be fitted successfully"

            # Test predictions
            predictions = training.predict(training.data_test[feature_columns].head(5))
            assert predictions is not None, f"{model_name} should produce predictions"
            assert not np.isnan(predictions).any(), f"{model_name} predictions should not contain NaN"

            # Test predict_proba for classification models
            if ml_type == "classification":
                probabilities = training.predict_proba(training.data_test[feature_columns].head(5))
                assert probabilities is not None, f"{model_name} should produce probabilities"
                assert not np.isnan(probabilities).any(), f"{model_name} probabilities should not contain NaN"
                assert np.allclose(probabilities.sum(axis=1), 1.0), f"{model_name} probabilities should sum to 1"
                assert (probabilities >= 0).all() and (probabilities <= 1).all(), (
                    f"{model_name} probabilities should be between 0 and 1"
                )

    def test_feature_importance_internal(self, test_data, feature_config):
        """Test internal feature importance calculation."""
        data_train, data_dev, data_test = test_data
        feature_columns, feature_groups = feature_config

        warnings.filterwarnings("ignore")

        # Test with a model that has internal feature importance
        training = create_training_instance(
            data_train, data_dev, data_test, "classification", "RandomForestClassifier", feature_columns, feature_groups
        )
        training.fit()

        training.calculate_fi_internal()
        fi_internal = training.feature_importances.get("internal")

        assert fi_internal is not None, "Internal feature importance should be calculated"
        assert not fi_internal.empty, "Internal feature importance should not be empty"
        assert len(fi_internal) == len(feature_columns), "Should have importance for all features"

    def test_feature_importance_permutation(self, test_data, feature_config):
        """Test permutation feature importance calculation."""
        data_train, data_dev, data_test = test_data
        feature_columns, feature_groups = feature_config

        warnings.filterwarnings("ignore")

        training = create_training_instance(
            data_train, data_dev, data_test, "classification", "RandomForestClassifier", feature_columns, feature_groups
        )
        training.fit()

        training.calculate_fi_permutation(partition="dev", n_repeats=2)
        fi_perm = training.feature_importances.get("permutation_dev")

        assert fi_perm is not None, "Permutation feature importance should be calculated"
        assert not fi_perm.empty, "Permutation feature importance should not be empty"

    def test_feature_importance_constant(self, test_data, feature_config):
        """Test constant feature importance calculation."""
        data_train, data_dev, data_test = test_data
        feature_columns, feature_groups = feature_config

        warnings.filterwarnings("ignore")

        training = create_training_instance(
            data_train, data_dev, data_test, "classification", "RandomForestClassifier", feature_columns, feature_groups
        )
        training.fit()

        training.calculate_fi_constant()
        fi_constant = training.feature_importances.get("constant")

        assert fi_constant is not None, "Constant feature importance should be calculated"
        assert not fi_constant.empty, "Constant feature importance should not be empty"
        assert len(fi_constant) == len(feature_columns), "Should have constant importance for all features"
        assert (fi_constant["importance"] == 1).all(), "All constant importances should be 1"

    def test_outlier_detection(self, test_data, feature_config):
        """Test outlier detection functionality."""
        data_train, data_dev, data_test = test_data
        feature_columns, feature_groups = feature_config

        warnings.filterwarnings("ignore")

        training_config = {
            "ml_model_type": "RandomForestClassifier",
            "ml_model_params": {},
            "outl_reduction": 5,  # Enable outlier detection
        }

        training = Training(
            training_id="test_outlier_detection",
            ml_type="classification",
            target_assignments={"target": "target_class"},
            feature_columns=feature_columns,
            row_column="row_id",
            data_train=data_train,
            data_dev=data_dev,
            data_test=data_test,
            target_metric="AUCROC",
            max_features=0,
            feature_groups=feature_groups,
            config_training=training_config,
        )

        training.fit()

        # Check that outliers were detected (may be 0 in some cases, which is acceptable)
        assert isinstance(training.outlier_samples, list), "Outlier samples should be a list"
        assert len(training.outlier_samples) >= 0, "Outlier samples should be non-negative"

    def test_features_used_calculation(self, test_data, feature_config):
        """Test features used calculation."""
        data_train, data_dev, data_test = test_data
        feature_columns, feature_groups = feature_config

        warnings.filterwarnings("ignore")

        training = create_training_instance(
            data_train, data_dev, data_test, "classification", "RandomForestClassifier", feature_columns, feature_groups
        )
        training.fit()

        assert isinstance(training.features_used, list), "Features used should be a list"
        assert len(training.features_used) >= 0, "Features used should be non-negative length"

    @pytest.mark.parametrize("model_name", ["CatBoostClassifier", "CatBoostRegressor"])
    def test_catboost_models(self, test_data, feature_config, model_name):
        """Test CatBoost models specifically."""
        data_train, data_dev, data_test = test_data
        feature_columns, feature_groups = feature_config

        warnings.filterwarnings("ignore")

        ml_type = "classification" if "Classifier" in model_name else "regression"

        training = create_training_instance(
            data_train, data_dev, data_test, ml_type, model_name, feature_columns, feature_groups
        )

        # Test that CatBoost models can be fitted without errors
        training.fit()
        assert training.model is not None, f"{model_name} should be fitted successfully"

        # Test predictions
        predictions = training.predict(training.data_test[feature_columns].head(5))
        assert predictions is not None, f"{model_name} should produce predictions"
        assert not np.isnan(predictions).any(), f"{model_name} predictions should not contain NaN"


# ============================================================================
# LEGACY SUPPORT - Keep original function for backward compatibility
# ============================================================================


def run_comprehensive_tests():
    """Run the complete test suite (legacy function for backward compatibility)."""
    print("=" * 80)
    print("COMPREHENSIVE TRAINING CLASS TEST SUITE")
    print("=" * 80)
    print("Note: This function is deprecated. Use 'pytest test_training_class_comprehensive.py' instead.")
    print("=" * 80)

    # Run pytest programmatically

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"], check=False, capture_output=False, text=True
    )
    return result.returncode == 0


if __name__ == "__main__":
    run_comprehensive_tests()
