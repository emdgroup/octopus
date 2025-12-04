"""Test octo manager."""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from octopus.experiment import OctoExperiment
from octopus.manager import OctoManager


@pytest.fixture
def mock_tasks():
    """Create mock tasks."""
    return [
        Mock(
            task_id=1,
            depends_on_task=0,
            module="test_module",
            description="Test",
            load_task=False,
        ),
        Mock(
            task_id=2,
            depends_on_task=1,
            module="test_module",
            description="Test",
            load_task=False,
        ),
    ]


@pytest.fixture
def mock_experiment():
    """Create mock experiment."""
    experiment = Mock(spec=OctoExperiment)
    experiment.experiment_id = "test_exp"
    experiment.path_study = Path("/tmp/test_study")
    experiment.feature_columns = ["feature1", "feature2"]
    experiment.calculate_feature_groups = Mock(return_value=["group1"])
    return experiment


@pytest.fixture
def octo_manager(mock_tasks, mock_experiment):
    """Create octo manager."""
    return OctoManager(
        base_experiments=[mock_experiment],
        tasks=mock_tasks,
        outer_parallelization=False,
        run_single_experiment_num=-1,
    )


def test_run_outer_experiments_sequential(octo_manager):
    """Test run outer experiments sequential."""
    with patch.object(OctoManager, "create_execute_mlmodules") as mock_create_execute:
        octo_manager.run_outer_experiments()
        assert mock_create_execute.call_count == 1


def test_run_outer_experiments_parallel_A(octo_manager):
    # Arrange
    octo_manager.outer_parallelization = True

    with (
        patch.object(type(octo_manager), "_run_parallel_ray") as mock_run,
        patch("octopus.manager.core.init_ray"),
        patch("octopus.manager.core.shutdown_ray"),
    ):
        # Act
        octo_manager.run_outer_experiments()
        # Assert
        mock_run.assert_called_once()


def test_run_outer_experiments_parallel_B(octo_manager):
    octo_manager.outer_parallelization = True

    with (
        patch("octopus.manager.core.init_ray"),
        patch("octopus.manager.core.shutdown_ray"),
        patch("octopus.manager.core.run_parallel_outer_ray", return_value=[True]) as mock_run,
    ):
        octo_manager.run_outer_experiments()

        mock_run.assert_called_once()
        # Inspect args/kwargs
        _, kwargs = mock_run.call_args
        assert kwargs["base_experiments"] is octo_manager.base_experiments
        assert callable(kwargs["create_execute_mlmodules"])
        assert kwargs["num_workers"] == min(len(octo_manager.base_experiments), os.cpu_count() or 1)


def test_run_single_experiment(octo_manager):
    """Test run single experiment."""
    octo_manager.run_single_experiment_num = 0
    with patch.object(OctoManager, "create_execute_mlmodules") as mock_create_execute:
        octo_manager.run_outer_experiments()
        mock_create_execute.assert_called_once_with(octo_manager.base_experiments[0])


def test_create_new_experiment(octo_manager, mock_experiment):
    """Test create new experiment."""
    element = octo_manager.tasks[0]
    new_experiment = octo_manager._create_new_experiment(mock_experiment, element)
    assert new_experiment.ml_module == element.module
    assert new_experiment.task_id == element.task_id


def test_update_from_input_item(octo_manager, mock_experiment):
    """Test update from input item."""
    input_experiment = Mock(spec=OctoExperiment)
    input_experiment.selected_features = ["new_feature"]

    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(OctoExperiment, "from_pickle", return_value=input_experiment),
    ):
        exp_path_dict = {1: Path("/tmp/input_exp.pkl")}
        mock_experiment.depends_on_task = 1
        octo_manager._update_from_input_item(mock_experiment, exp_path_dict)
        assert mock_experiment.feature_columns == ["new_feature"]


def test_load_existing_experiment(octo_manager, mock_experiment):
    """Test load existing experiment."""
    element = Mock(task_id=3)
    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(OctoExperiment, "from_pickle", return_value=mock_experiment),
    ):
        loaded_experiment = octo_manager._load_existing_experiment(mock_experiment, element)
        assert loaded_experiment == mock_experiment


def test_load_existing_experiment_not_found(octo_manager, mock_experiment):
    """Test experiment not found."""
    element = Mock(task_id=3)
    with patch.object(Path, "exists", return_value=False), pytest.raises(FileNotFoundError):
        octo_manager._load_existing_experiment(mock_experiment, element)
