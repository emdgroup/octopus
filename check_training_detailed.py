#!/usr/bin/env python3
"""Enhanced Training Class Test Suite with Detailed Visualization.

This enhanced test suite provides comprehensive testing of the Training class with:
- Rich console output with colors and formatting
- Detailed model-specific information and metrics
- Performance benchmarking and timing
- Feature importance visualization
- Model comparison tables
- Error analysis and debugging information
- Progress bars and status indicators

Usage:
    python test_training_detailed.py
    python test_training_detailed.py --model-type classification
    python test_training_detailed.py --quick  # Run subset of tests
"""

import argparse
import time
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from octopus.models.inventory import ModelInventory
from octopus.modules.octo.training import Training

# ============================================================================
# CONFIGURATION
# ============================================================================

TEST_CONFIG = {
    "n_samples": 300,
    "test_split": 0.3,
    "dev_split": 0.2,
    "random_seed": 42,
    "quick_mode_samples": 100,
}


# ANSI color codes for rich console output
class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# ============================================================================
# UTILITY CLASSES
# ============================================================================


class TestReporter:
    """Enhanced test reporter with rich formatting and detailed metrics."""

    def __init__(self):
        self.results = defaultdict(list)
        self.timings = {}
        self.model_details = {}
        self.start_time = time.time()

    def print_header(self, title: str, level: int = 1):
        """Print formatted header."""
        if level == 1:
            print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}")
            print(f"🚀 {title.upper()}")
            print(f"{'=' * 80}{Colors.ENDC}")
        elif level == 2:
            print(f"\n{Colors.OKBLUE}{Colors.BOLD}{'-' * 60}")
            print(f"📊 {title}")
            print(f"{'-' * 60}{Colors.ENDC}")
        else:
            print(f"\n{Colors.OKCYAN}▶ {title}{Colors.ENDC}")

    def print_success(self, message: str, details: str = ""):
        """Print success message."""
        print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")
        if details:
            print(f"   {Colors.OKCYAN}{details}{Colors.ENDC}")

    def print_warning(self, message: str, details: str = ""):
        """Print warning message."""
        print(f"{Colors.WARNING}⚠️  {message}{Colors.ENDC}")
        if details:
            print(f"   {Colors.WARNING}{details}{Colors.ENDC}")

    def print_error(self, message: str, details: str = ""):
        """Print error message."""
        print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")
        if details:
            print(f"   {Colors.FAIL}{details}{Colors.ENDC}")

    def print_info(self, message: str, indent: int = 0):
        """Print info message."""
        prefix = "   " * indent
        print(f"{prefix}{Colors.OKCYAN}ℹ️  {message}{Colors.ENDC}")

    def record_result(self, category: str, test_name: str, status: str, duration: float = 0, details: Dict = None):
        """Record test result."""
        self.results[category].append(
            {"test": test_name, "status": status, "duration": duration, "details": details or {}}
        )

    def record_model_details(self, model_name: str, ml_type: str, details: Dict):
        """Record detailed model information."""
        self.model_details[f"{ml_type}_{model_name}"] = details

    def print_model_summary(
        self, model_name: str, ml_type: str, training: Training, duration: float, predictions: np.ndarray = None
    ):
        """Print detailed model summary."""
        print(f"\n{Colors.BOLD}📋 Model Summary: {model_name} ({ml_type}){Colors.ENDC}")
        print(f"   Training Time: {duration:.2f}s")
        print(f"   Training ID: {training.training_id}")
        print(f"   Target Metric: {training.target_metric}")
        print(f"   Features Used: {len(training.features_used)} / {len(training.feature_columns)}")

        if hasattr(training, "model") and training.model:
            model_info = self._get_model_info(training.model)
            for key, value in model_info.items():
                print(f"   {key}: {value}")

        if predictions is not None:
            print("   Prediction Stats:")
            print(f"     Shape: {predictions.shape}")
            print(f"     Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            print(f"     Mean: {predictions.mean():.4f}")
            print(f"     Std: {predictions.std():.4f}")

    def _get_model_info(self, model) -> Dict[str, str]:
        """Extract model-specific information."""
        info = {}
        try:
            model_type = type(model).__name__
            info["Model Type"] = model_type

            # Get model-specific parameters
            if hasattr(model, "get_params"):
                try:
                    params = model.get_params()
                    # Show only important parameters
                    important_params = [
                        "n_estimators",
                        "max_depth",
                        "learning_rate",
                        "alpha",
                        "C",
                        "gamma",
                        "kernel",
                        "n_neighbors",
                    ]
                    for param in important_params:
                        if param in params:
                            info[param.replace("_", " ").title()] = str(params[param])
                except (NotImplementedError, AttributeError):
                    pass

            # Model-specific information
            try:
                if hasattr(model, "feature_importances_"):
                    info["Has Feature Importances"] = "Yes"
            except (NotImplementedError, AttributeError):
                pass

            try:
                if hasattr(model, "coef_"):
                    info["Has Coefficients"] = "Yes"
            except (NotImplementedError, AttributeError):
                pass

            try:
                if hasattr(model, "n_features_in_"):
                    info["Features In"] = str(model.n_features_in_)
            except (NotImplementedError, AttributeError):
                pass

        except Exception:
            # If any error occurs, return basic info
            info["Model Type"] = "Unknown"

        return info

    def print_feature_importance_summary(self, training: Training):
        """Print feature importance summary."""
        if not training.feature_importances:
            return

        print(f"\n{Colors.BOLD}🎯 Feature Importance Summary{Colors.ENDC}")
        for method, fi_df in training.feature_importances.items():
            if fi_df is not None and not fi_df.empty:
                print(f"   {method.title()}:")
                top_features = fi_df.nlargest(3, "importance")
                for _, row in top_features.iterrows():
                    print(f"     {row['feature']}: {row['importance']:.4f}")

    def print_final_summary(self):
        """Print comprehensive final summary."""
        total_time = time.time() - self.start_time

        self.print_header("🎉 TEST SUITE COMPLETED", level=1)

        # Overall statistics
        total_tests = sum(len(tests) for tests in self.results.values())
        passed_tests = sum(1 for tests in self.results.values() for test in tests if test["status"] == "PASS")
        failed_tests = sum(1 for tests in self.results.values() for test in tests if test["status"] == "FAIL")
        warning_tests = sum(1 for tests in self.results.values() for test in tests if test["status"] == "WARNING")

        print(f"\n{Colors.BOLD}📊 Overall Statistics:{Colors.ENDC}")
        print(f"   Total Tests: {total_tests}")
        print(f"   {Colors.OKGREEN}Passed: {passed_tests}{Colors.ENDC}")
        print(f"   {Colors.FAIL}Failed: {failed_tests}{Colors.ENDC}")
        print(f"   {Colors.WARNING}Warnings: {warning_tests}{Colors.ENDC}")
        print(f"   Success Rate: {passed_tests / total_tests * 100:.1f}%")
        print(f"   Total Time: {total_time:.2f}s")

        # Category breakdown
        print(f"\n{Colors.BOLD}📋 Results by Category:{Colors.ENDC}")
        for category, tests in self.results.items():
            category_passed = sum(1 for test in tests if test["status"] == "PASS")
            category_total = len(tests)
            print(f"   {category}: {category_passed}/{category_total} ({category_passed / category_total * 100:.1f}%)")

        # Model performance comparison
        if self.model_details:
            self.print_model_comparison()

    def print_model_comparison(self):
        """Print model performance comparison table."""
        print(f"\n{Colors.BOLD}🏆 Model Performance Comparison:{Colors.ENDC}")

        # Group by ML type
        by_type = defaultdict(list)
        for key, details in self.model_details.items():
            ml_type, model_name = key.split("_", 1)
            by_type[ml_type].append((model_name, details))

        for ml_type, models in by_type.items():
            print(f"\n   {Colors.UNDERLINE}{ml_type.title()} Models:{Colors.ENDC}")
            print(f"   {'Model':<25} {'Time (s)':<10} {'Features':<10} {'Status':<10}")
            print(f"   {'-' * 60}")

            for model_name, details in sorted(models, key=lambda x: x[1].get("duration", 0)):
                duration = details.get("duration", 0)
                features = details.get("features_used", "N/A")
                status = details.get("status", "Unknown")
                status_color = Colors.OKGREEN if status == "PASS" else Colors.FAIL
                print(f"   {model_name:<25} {duration:<10.2f} {features:<10} {status_color}{status}{Colors.ENDC}")


class ModelCache:
    """Enhanced model cache with filtering capabilities."""

    def __init__(self):
        self._cached_models_by_type = None
        self._tabpfn_skip_logged = False

    def get_available_models_by_type(self, filter_models: Optional[List[str]] = None):
        """Get available models with optional filtering."""
        if self._cached_models_by_type is None:
            self._discover_models()

        if filter_models:
            filtered = {}
            for ml_type, models in self._cached_models_by_type.items():
                filtered[ml_type] = [m for m in models if m in filter_models]
            return filtered

        return self._cached_models_by_type

    def _discover_models(self):
        """Discover available models."""
        inventory = ModelInventory()
        all_models = inventory.models

        models_by_type = {"classification": [], "regression": [], "timetoevent": []}
        skipped_tabpfn_models = []

        for model_name in all_models.keys():
            if "TabPFN" in model_name:
                skipped_tabpfn_models.append(model_name)
                continue

            try:
                model_config = inventory.get_model_config(model_name)
                ml_type = model_config.ml_type
                if ml_type in models_by_type:
                    models_by_type[ml_type].append(model_name)
            except Exception:
                continue

        if skipped_tabpfn_models and not self._tabpfn_skip_logged:
            print(f"{Colors.WARNING}⚠️  Skipping TabPFN models: {', '.join(skipped_tabpfn_models)}{Colors.ENDC}")
            self._tabpfn_skip_logged = True

        self._cached_models_by_type = models_by_type


# ============================================================================
# DATA GENERATION
# ============================================================================


def create_enhanced_test_data(quick_mode: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create enhanced test dataset with comprehensive features."""
    n_samples = TEST_CONFIG["quick_mode_samples"] if quick_mode else TEST_CONFIG["n_samples"]
    np.random.seed(TEST_CONFIG["random_seed"])

    print(f"{Colors.OKCYAN}🔧 Generating test data ({n_samples} samples)...{Colors.ENDC}")

    # Generate diverse numerical features
    data = pd.DataFrame(
        {
            "num_normal": np.random.normal(10, 2, n_samples),
            "num_uniform": np.random.uniform(0, 100, n_samples),
            "num_exponential": np.random.exponential(2, n_samples),
            "num_correlated": np.random.normal(50, 10, n_samples),
        }
    )

    # Create correlated feature
    data["num_correlated_2"] = data["num_correlated"] * 0.8 + np.random.normal(0, 5, n_samples)

    # Add missing values strategically
    missing_patterns = [15, 20, 25, 30, 35]
    for i, col in enumerate(["num_normal", "num_uniform", "num_exponential", "num_correlated", "num_correlated_2"]):
        data.loc[:: missing_patterns[i], col] = np.nan

    # Generate categorical features with different cardinalities
    data["cat_low"] = np.random.choice([1, 2, 3], n_samples).astype(float)
    data["cat_medium"] = np.random.choice([1, 2, 3, 4, 5], n_samples).astype(float)
    data["cat_high"] = np.random.choice(range(1, 11), n_samples).astype(float)

    # Add missing values to categorical features
    data.loc[::18, "cat_low"] = np.nan
    data.loc[::22, "cat_medium"] = np.nan
    data.loc[::28, "cat_high"] = np.nan

    # Add row identifier
    data["row_id"] = range(n_samples)

    # Generate realistic targets with different complexities
    # Classification target (binary)
    linear_combination = (
        0.3 * data["num_normal"].fillna(data["num_normal"].mean())
        + 0.2 * data["num_uniform"].fillna(data["num_uniform"].mean())
        + 0.1 * data["cat_medium"].fillna(data["cat_medium"].mean())
    )

    probabilities = 1 / (1 + np.exp(-0.1 * (linear_combination - linear_combination.mean())))
    data["target_class"] = np.random.binomial(1, probabilities)

    # Regression target
    data["target_reg"] = (
        0.5 * data["num_normal"].fillna(data["num_normal"].mean())
        + 0.3 * data["num_correlated"].fillna(data["num_correlated"].mean())
        + 0.2 * data["cat_medium"].fillna(data["cat_medium"].mean())
        + np.random.normal(0, 2, n_samples)
    )

    # Time-to-event targets
    data["duration"] = np.random.exponential(10, n_samples)
    event_prob = 0.7 + 0.2 * (data["num_normal"].fillna(data["num_normal"].mean()) - 10) / 4
    event_prob = np.clip(event_prob, 0.1, 0.9)
    data["event"] = np.random.binomial(1, event_prob).astype(bool)

    # Split data
    n_train = int(n_samples * (1 - TEST_CONFIG["test_split"] - TEST_CONFIG["dev_split"]))
    n_dev = int(n_samples * TEST_CONFIG["dev_split"])

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    dev_idx = indices[n_train : n_train + n_dev]
    test_idx = indices[n_train + n_dev :]

    train_data = data.iloc[train_idx].reset_index(drop=True)
    dev_data = data.iloc[dev_idx].reset_index(drop=True)
    test_data = data.iloc[test_idx].reset_index(drop=True)

    print(f"   Train: {len(train_data)} samples")
    print(f"   Dev: {len(dev_data)} samples")
    print(f"   Test: {len(test_data)} samples")

    return train_data, dev_data, test_data


def get_feature_config():
    """Get enhanced feature configuration."""
    feature_columns = [
        "num_normal",
        "num_uniform",
        "num_exponential",
        "num_correlated",
        "num_correlated_2",
        "cat_low",
        "cat_medium",
        "cat_high",
    ]

    feature_groups = {
        "numerical_basic": ["num_normal", "num_uniform", "num_exponential"],
        "numerical_correlated": ["num_correlated", "num_correlated_2"],
        "categorical_low": ["cat_low"],
        "categorical_medium": ["cat_medium"],
        "categorical_high": ["cat_high"],
    }

    return feature_columns, feature_groups


# ============================================================================
# ENHANCED TESTING FUNCTIONS
# ============================================================================


def create_training_instance(
    data_train: pd.DataFrame,
    data_dev: pd.DataFrame,
    data_test: pd.DataFrame,
    ml_type: str,
    model_name: str,
    feature_columns: List[str],
    feature_groups: Dict[str, List[str]],
) -> Training:
    """Create enhanced training instance with model-specific configurations."""
    model_configs = {
        "classification": {
            "target_assignments": {"target": "target_class"},
            "target_metric": "AUCROC",
        },
        "regression": {
            "target_assignments": {"target": "target_reg"},
            "target_metric": "R2",
        },
        "timetoevent": {
            "target_assignments": {"duration": "duration", "event": "event"},
            "target_metric": "CI",
        },
    }

    config = model_configs[ml_type]

    # Model-specific parameter optimization
    ml_model_params = {}
    if "CatBoost" in model_name:
        ml_model_params = {
            "verbose": False,
            "allow_writing_files": False,  # no writinng on disk
            "iterations": 100,
        }
    elif "XGB" in model_name:
        ml_model_params = {"verbosity": 0, "n_estimators": 100}
    elif "RandomForest" in model_name:
        ml_model_params = {"n_estimators": 50, "random_state": 42}
    elif "Ridge" in model_name or "Lasso" in model_name:
        ml_model_params = {"alpha": 1.0, "random_state": 42}

    training_config = {
        "ml_model_type": model_name,
        "ml_model_params": ml_model_params,
        "outl_reduction": 0,
    }

    return Training(
        training_id=f"enhanced_test_{ml_type}_{model_name}_{int(time.time())}",
        ml_type=ml_type,
        target_assignments=config["target_assignments"],
        feature_columns=feature_columns,
        row_column="row_id",
        data_train=data_train,
        data_dev=data_dev,
        data_test=data_test,
        target_metric=config["target_metric"],
        max_features=5,
        feature_groups=feature_groups,
        config_training=training_config,
    )


def test_model_comprehensive(
    model_name: str,
    ml_type: str,
    data_train: pd.DataFrame,
    data_dev: pd.DataFrame,
    data_test: pd.DataFrame,
    feature_columns: List[str],
    feature_groups: Dict[str, List[str]],
    reporter: TestReporter,
) -> bool:
    """Comprehensive model testing with detailed reporting."""
    reporter.print_header(f"Testing {model_name} ({ml_type})", level=3)

    start_time = time.time()
    success = True

    try:
        # Create and fit training instance
        training = create_training_instance(
            data_train, data_dev, data_test, ml_type, model_name, feature_columns, feature_groups
        )

        # Fit model with timing
        fit_start = time.time()
        training.fit()
        fit_duration = time.time() - fit_start

        reporter.print_success("Model fitted successfully", f"Time: {fit_duration:.2f}s")

        # Test predictions
        test_sample = training.data_test[feature_columns].head(10)
        predictions = training.predict(test_sample)

        if predictions is not None and not np.isnan(predictions).any():
            reporter.print_success(
                "Predictions generated",
                f"Shape: {predictions.shape}, Range: [{predictions.min():.3f}, {predictions.max():.3f}]",
            )
        else:
            reporter.print_error("Invalid predictions generated")
            success = False

        # Test predict_proba for classification
        if ml_type == "classification":
            try:
                probabilities = training.predict_proba(test_sample)
                if (
                    probabilities is not None
                    and not np.isnan(probabilities).any()
                    and np.allclose(probabilities.sum(axis=1), 1.0)
                    and (probabilities >= 0).all()
                    and (probabilities <= 1).all()
                ):
                    reporter.print_success("Probabilities generated", f"Shape: {probabilities.shape}")
                else:
                    reporter.print_error("Invalid probabilities generated")
                    success = False
            except NotImplementedError:
                reporter.print_warning("Probabilities not supported", "predict_proba not implemented for this model")

        # Test feature importance methods
        fi_results = test_feature_importance_methods(training, reporter)

        # Print detailed model summary
        total_duration = time.time() - start_time
        reporter.print_model_summary(model_name, ml_type, training, total_duration, predictions)
        reporter.print_feature_importance_summary(training)

        # Record detailed results
        model_details = {
            "status": "PASS" if success else "FAIL",
            "duration": total_duration,
            "fit_duration": fit_duration,
            "features_used": len(training.features_used),
            "total_features": len(feature_columns),
            "feature_importance_methods": len([k for k, v in fi_results.items() if v]),
            "predictions_valid": predictions is not None and not np.isnan(predictions).any(),
        }

        if ml_type == "classification":
            model_details["probabilities_valid"] = probabilities is not None and not np.isnan(probabilities).any()

        reporter.record_model_details(model_name, ml_type, model_details)
        reporter.record_result(
            "models", f"{ml_type}_{model_name}", "PASS" if success else "FAIL", total_duration, model_details
        )

    except Exception as e:
        total_duration = time.time() - start_time
        error_details = f"{type(e).__name__}: {str(e)}"
        reporter.print_error("Model test failed", error_details)
        reporter.record_result("models", f"{ml_type}_{model_name}", "FAIL", total_duration)
        success = False

    return success


def test_feature_importance_methods(training: Training, reporter: TestReporter) -> Dict[str, bool]:
    """Test feature importance methods with detailed reporting."""
    methods = {
        "internal": lambda: training.calculate_fi_internal(),
        "permutation": lambda: training.calculate_fi_permutation(partition="dev", n_repeats=2),
        "constant": lambda: training.calculate_fi_constant(),
    }

    results = {}

    for method_name, method_func in methods.items():
        try:
            method_func()
            fi_result = training.feature_importances.get(method_name) or training.feature_importances.get(
                f"{method_name}_dev"
            )

            if fi_result is not None and not fi_result.empty:
                reporter.print_success(f"Feature importance ({method_name})", f"{len(fi_result)} features")
                results[method_name] = True
            else:
                reporter.print_warning(f"Feature importance ({method_name})", "No results")
                results[method_name] = False

        except Exception as e:
            reporter.print_warning(f"Feature importance ({method_name})", f"Failed: {str(e)}")
            results[method_name] = False

    return results


def test_data_preprocessing(
    data_train: pd.DataFrame,
    data_dev: pd.DataFrame,
    data_test: pd.DataFrame,
    feature_columns: List[str],
    feature_groups: Dict[str, List[str]],
    reporter: TestReporter,
):
    """Test data preprocessing with detailed analysis."""
    reporter.print_header("Data Preprocessing Analysis", level=2)

    # Create training instance for preprocessing test
    training = create_training_instance(
        data_train, data_dev, data_test, "classification", "RandomForestClassifier", feature_columns, feature_groups
    )

    # Analyze data before preprocessing
    missing_before = training.data_train[feature_columns].isnull().sum()
    total_missing = missing_before.sum()

    reporter.print_info(f"Missing values before preprocessing: {total_missing}")
    for col, missing in missing_before.items():
        if missing > 0:
            reporter.print_info(f"  {col}: {missing} ({missing / len(training.data_train) * 100:.1f}%)", 1)

    # Fit and test preprocessing
    start_time = time.time()
    training.fit()
    preprocessing_time = time.time() - start_time

    if training.preprocessing_pipeline is not None:
        reporter.print_success("Preprocessing pipeline created", f"Time: {preprocessing_time:.2f}s")

        # Test preprocessing on new data
        test_sample = training.data_test[feature_columns].head(20)
        processed_sample = training.preprocessing_pipeline.transform(test_sample)

        # Analyze processed data
        if isinstance(processed_sample, np.ndarray):
            processed_df = pd.DataFrame(processed_sample)
            missing_after = processed_df.isnull().sum().sum()
        else:
            missing_after = processed_sample.isnull().sum().sum()

        reporter.print_info(f"Original shape: {test_sample.shape}")
        reporter.print_info(f"Processed shape: {processed_sample.shape}")
        reporter.print_info(f"Missing values after: {missing_after}")

        if missing_after == 0:
            reporter.print_success("All missing values handled")
        else:
            reporter.print_warning(f"{missing_after} missing values remain")

        reporter.record_result("preprocessing", "pipeline_creation", "PASS", preprocessing_time)
    else:
        reporter.print_error("Preprocessing pipeline not created")
        reporter.record_result("preprocessing", "pipeline_creation", "FAIL", preprocessing_time)


def run_enhanced_test_suite(
    ml_types: Optional[List[str]] = None, model_filter: Optional[List[str]] = None, quick_mode: bool = False
):
    """Run the enhanced test suite with comprehensive reporting."""
    # Initialize components
    reporter = TestReporter()
    model_cache = ModelCache()

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    reporter.print_header("🧪 ENHANCED TRAINING CLASS TEST SUITE")

    print(f"{Colors.BOLD}Configuration:{Colors.ENDC}")
    print(f"   Quick Mode: {'Yes' if quick_mode else 'No'}")
    print(f"   ML Types: {ml_types or 'All'}")
    print(f"   Model Filter: {model_filter or 'All'}")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Generate test data
    data_train, data_dev, data_test = create_enhanced_test_data(quick_mode)
    feature_columns, feature_groups = get_feature_config()

    # Test data preprocessing
    test_data_preprocessing(data_train, data_dev, data_test, feature_columns, feature_groups, reporter)

    # Get available models
    available_models = model_cache.get_available_models_by_type(model_filter)

    # Filter ML types if specified
    if ml_types:
        available_models = {k: v for k, v in available_models.items() if k in ml_types}

    # Test models by type
    for ml_type, models in available_models.items():
        if not models:
            continue

        reporter.print_header(f"{ml_type.title()} Models", level=2)
        reporter.print_info(f"Testing {len(models)} models: {', '.join(models)}")

        successful_models = 0
        for model_name in models:
            success = test_model_comprehensive(
                model_name, ml_type, data_train, data_dev, data_test, feature_columns, feature_groups, reporter
            )
            if success:
                successful_models += 1

        reporter.print_info(f"Completed {ml_type}: {successful_models}/{len(models)} successful")

    # Print final comprehensive summary
    reporter.print_final_summary()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def main():
    """Run the enhanced test suite with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Enhanced Training Class Test Suite")
    parser.add_argument(
        "--model-type", choices=["classification", "regression", "timetoevent"], help="Test only specific ML type"
    )
    parser.add_argument("--models", nargs="+", help="Test only specific models")
    parser.add_argument("--quick", action="store_true", help="Run in quick mode with fewer samples")

    args = parser.parse_args()

    ml_types = [args.model_type] if args.model_type else None
    model_filter = args.models
    quick_mode = args.quick

    # Run the enhanced test suite
    run_enhanced_test_suite(ml_types=ml_types, model_filter=model_filter, quick_mode=quick_mode)


if __name__ == "__main__":
    main()
