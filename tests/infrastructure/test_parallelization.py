import os
import subprocess
import sys
import tempfile

import pandas as pd
import threadpoolctl
from sklearn.datasets import make_classification

import octopus  # noqa: F401
from octopus import OctoStudy
from octopus.modules import _PARALLELIZATION_ENV_VARS, Octo


def _run_octo_with_ray_parallelization():
    """Helper function that runs Octo module with Ray parallelization enabled.

    This function is designed to be run in a subprocess with OMP_NUM_THREADS=42
    to verify that Ray workers detect active parallelization and raise an error.

    Key requirements:
    - Uses outer_parallelization=True to enable Ray workers
    - Does NOT use run_single_experiment_num (defaults to -1)
    - Uses n_folds_outer=2 to create multiple experiments that run in parallel
    - Uses minimal Octo configuration with mostly defaults
    """
    # Create synthetic binary classification dataset
    X, y = make_classification(
        n_samples=30,
        n_features=5,
        n_informative=3,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )

    feature_names = [f"feature_{i}" for i in range(5)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    df = df.reset_index()

    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="test_ray_parallelization",
            ml_type="classification",
            target_metric="ACCBAL",
            feature_columns=feature_names,
            target_columns=["target"],
            sample_id="index",
            stratification_column="target",
            metrics=["AUCROC", "ACCBAL"],
            datasplit_seed_outer=1234,
            n_folds_outer=2,  # Creates 2 experiments to run in parallel
            path=temp_dir,
            ignore_data_health_warning=True,
            outer_parallelization=True,  # Enable Ray parallelization
            # DO NOT set run_single_experiment_num - let it default to -1
            workflow=[
                Octo(
                    description="octo_test",
                    task_id=0,
                    depends_on_task=-1,
                    n_trials=5,  # As requested
                    # All other parameters use defaults
                )
            ],
        )

        # This will trigger Ray worker execution and the parallelization check
        study.fit(data=df)


def test_parallelization_inactive_in_threadpoolctl():
    threadpool_info = threadpoolctl.threadpool_info()

    # we expect the following openmp libraries to be loaded: shipped with torch, shipped with sklearn, system openmp
    assert len(threadpool_info) >= 3

    for lib in threadpool_info:
        assert lib["num_threads"] == 1


def test_parallelization_limited_by_env():
    # these vars are being set in octopus/modules/__init__.py
    for env_var in _PARALLELIZATION_ENV_VARS:
        assert os.environ.get(env_var, None) == "1"


def test_ray_workers_detect_active_parallelization():
    """Test that ray workers abort if they detect active parallelization.

    This test spawns a subprocess that runs a dedicated test workflow with OMP_NUM_THREADS
    set to 42 to make sure the modification of the ENV var is not accidentally carried over
    to other tests. The dedicated test uses the Octo module with Ray parallelization enabled.
    """
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "42"  # Activate thread-level parallelization

    res = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.test_infrastructure import _run_octo_with_ray_parallelization; "
            "_run_octo_with_ray_parallelization()",
        ],
        check=False,
        env=env,
        capture_output=True,
    )

    assert res.returncode == 1  # Expecting failure due to active thread-level parallelization
    # Check that the failure is related to OMP_NUM_THREADS being set to 42
    output = res.stdout + res.stderr
    assert b"OMP_NUM_THREADS is set to 42" in output or b"OMP_NUM_THREADS=42" in output
