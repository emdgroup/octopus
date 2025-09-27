"""Test AutoGluon workflows."""

import os
import re
import shutil
import tempfile
from pathlib import Path

import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.experiment import OctoExperiment
from octopus.modules import AutoGluon


class TestAutogluonWorkflows:
    """Test the AutoGluon classification workflow."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for studies
        self.temp_dir = tempfile.mkdtemp()
        self.studies_path = os.path.join(self.temp_dir, "studies")
        os.makedirs(self.studies_path, exist_ok=True)

        # Load the breast cancer dataset and reduce to 100 samples
        breast_cancer = load_breast_cancer(as_frame=True)
        df_full = breast_cancer["frame"].reset_index()
        self.df = df_full.iloc[:100].copy()  # Reduce to 100 samples
        self.df.columns = self.df.columns.str.replace(" ", "_")
        self.features = list(breast_cancer["feature_names"])
        self.features = [feature.replace(" ", "_") for feature in self.features]

    def teardown_method(self):
        """Clean up after each test method."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_full_classification_workflow(self):
        """Test the complete classification workflow execution."""
        # Create OctoData object
        octo_data = OctoData(
            data=self.df,
            target_columns=["target"],
            feature_columns=self.features,
            sample_id="index",
            datasplit_type="sample",
            stratification_column="target",
        )

        # Create configurations
        config_study = ConfigStudy(
            name="test_classification_workflow",
            ml_type="classification",
            target_metric="ACCBAL",
            metrics=["AUCROC", "ACCBAL", "ACC", "LOGLOSS"],
            datasplit_seed_outer=1234,
            n_folds_outer=5,  # Increased to 5 outer folds
            start_with_empty_study=True,
            path=self.studies_path,
            silently_overwrite_study=True,
            ignore_data_health_warning=True,
        )

        config_manager = ConfigManager(
            outer_parallelization=True,  # Enable parallelization
            run_single_experiment_num=-1,  # Run all experiments
        )

        config_sequence = ConfigSequence(
            [
                AutoGluon(
                    description="ag_test",
                    sequence_id=0,
                    input_sequence_id=-1,
                    presets=["medium_quality"],
                    time_limit=30,  # Time limit for testing
                    verbosity=0,  # Minimize AutoGluon output
                ),
            ]
        )

        # Execute the workflow
        octo_ml = OctoML(
            octo_data,
            config_study=config_study,
            config_manager=config_manager,
            config_sequence=config_sequence,
        )

        # This should complete without errors
        octo_ml.run_study()

        # Verify that study files were created
        study_path = Path(self.studies_path) / "test_classification_workflow"
        assert study_path.exists(), "Study directory should be created"

        # Test specific keys exist
        self._test_specific_keys(study_path)

        success = True
        assert success is True

    def test_full_regression_workflow(self):
        """Test the complete regression workflow execution."""
        # Load diabetes dataset and reduce to 100 samples
        diabetes = load_diabetes(as_frame=True)
        df_full = diabetes["frame"].reset_index()
        df_regression = df_full.iloc[:100].copy()  # Reduce to 100 samples

        # Create OctoData object for regression
        octo_data = OctoData(
            data=df_regression,
            target_columns=["target"],
            feature_columns=diabetes["feature_names"],
            sample_id="index",
            datasplit_type="sample",
        )

        # Create configurations for regression
        config_study = ConfigStudy(
            name="test_regression_workflow",
            ml_type="regression",
            target_metric="MAE",
            metrics=["MAE", "MSE", "R2"],  # Use only available metrics
            datasplit_seed_outer=1234,
            n_folds_outer=2,  # Reduced for faster testing
            start_with_empty_study=True,
            path=self.studies_path,
            silently_overwrite_study=True,
            ignore_data_health_warning=True,
        )

        config_manager = ConfigManager(
            outer_parallelization=False,  # Disabled for regression test
            run_single_experiment_num=0,  # Run single experiment
        )

        config_sequence = ConfigSequence(
            [
                AutoGluon(
                    description="ag_regression_test",
                    sequence_id=0,
                    input_sequence_id=-1,
                    presets=["medium_quality"],
                    time_limit=30,  # Time limit for testing
                    verbosity=0,  # Minimize AutoGluon output
                ),
            ]
        )

        # Execute the workflow
        octo_ml = OctoML(
            octo_data,
            config_study=config_study,
            config_manager=config_manager,
            config_sequence=config_sequence,
        )

        # This should complete without errors
        octo_ml.run_study()

        # Verify that study files were created
        study_path = Path(self.studies_path) / "test_regression_workflow"
        assert study_path.exists(), "Study directory should be created"

        # Test specific keys exist
        self._test_specific_keys(study_path)

        success = True
        assert success is True

    def _test_specific_keys(self, study_path):
        """Test that specific keys exist in the experiment results."""
        print("\n=== Testing Specific Keys ===")

        # Find experiment directories
        path_experiments = [f for f in study_path.glob("experiment*") if f.is_dir()]

        assert len(path_experiments) > 0, "No experiment directories found"

        # Track if we found the required keys
        found_autogluon_result = False
        found_autogluon_permutation_test = False

        # Iterate through experiments
        for path_exp in path_experiments:
            exp_name = str(path_exp.name)
            match = re.search(r"\d+", exp_name)
            exp_num = int(match.group()) if match else None

            # Find sequence directories
            path_sequences = [f for f in path_exp.glob("sequence*") if f.is_dir()]

            # Iterate through sequences
            for path_seq in path_sequences:
                seq_name = str(path_seq.name)
                match = re.search(r"\d+", seq_name)
                seq_num = int(match.group()) if match else None

                # Look for experiment pickle file
                path_exp_pkl = path_seq.joinpath(f"exp{exp_num}_{seq_num}.pkl")

                if path_exp_pkl.exists():
                    try:
                        # Load experiment
                        exp = OctoExperiment.from_pickle(path_exp_pkl)

                        # Test for 'autogluon' results key
                        if "autogluon" in exp.results.keys():
                            found_autogluon_result = True
                            print(f"✓ Found 'autogluon' results key in {path_exp_pkl}")

                            # Test for 'autogluon_permutation_test' feature importance key
                            result = exp.results["autogluon"]
                            if hasattr(result, "feature_importances") and result.feature_importances:
                                if "autogluon_permutation_test" in result.feature_importances.keys():
                                    found_autogluon_permutation_test = True
                                    print(
                                        f"✓ Found 'autogluon_permutation_test' feature importance key in {path_exp_pkl}"
                                    )

                    except Exception as e:
                        print(f"Error loading experiment {path_exp_pkl}: {e}")

        # Assert that we found the required keys
        assert found_autogluon_result, "Expected 'autogluon' key not found in experiment results"
        assert found_autogluon_permutation_test, (
            "Expected 'autogluon_permutation_test' key not found in feature importances"
        )

        print("✓ All required keys found successfully")
        print("=== Key Testing Complete ===\n")


if __name__ == "__main__":
    # Allow running the test directly
    pytest.main([__file__, "-v"])
