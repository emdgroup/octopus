"""Test octo manager and related components."""

from unittest.mock import Mock, patch

import attrs
import pytest
from upath import UPath
from upath.implementations.local import PosixUPath

from octopus.experiment import OctoExperiment
from octopus.manager import OctoManager
from octopus.manager.core import ResourceConfig
from octopus.manager.workflow_runner import WorkflowTaskRunner

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_workflow():
    """Create mock workflow."""
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
    experiment.path_study = UPath("/tmp/test_study")
    experiment.feature_columns = ["feature1", "feature2"]
    experiment.calculate_feature_groups = Mock(return_value=["group1"])
    return experiment


@pytest.fixture
def octo_manager(mock_workflow, mock_experiment):
    """Create octo manager."""
    return OctoManager(
        base_experiments=[mock_experiment],
        workflow=mock_workflow,
        outer_parallelization=False,
        run_single_experiment_num=-1,
        log_dir=mock_experiment.path_study,
    )


@pytest.fixture
def resources():
    """Create test resource config."""
    return ResourceConfig(num_cpus=4, num_workers=2, cpus_per_experiment=2)


# =============================================================================
# ResourceConfig Tests
# =============================================================================


class TestResourceConfig:
    """Tests for ResourceConfig."""

    def test_create_with_parallelization(self):
        """Test resource creation with outer parallelization."""
        config = ResourceConfig.create(
            num_experiments=4,
            outer_parallelization=True,
            num_cpus=8,
        )
        assert config.num_cpus == 8
        assert config.num_workers == 4  # min(4 experiments, 8 cpus)
        assert config.cpus_per_experiment == 2  # 8 / 4

    def test_create_without_parallelization(self):
        """Test resource creation without outer parallelization."""
        config = ResourceConfig.create(
            num_experiments=4,
            outer_parallelization=False,
            num_cpus=8,
        )
        assert config.num_cpus == 8
        assert config.num_workers == 4
        assert config.cpus_per_experiment == 8  # All CPUs for sequential

    def test_create_more_experiments_than_cpus(self):
        """Test when experiments exceed available CPUs."""
        config = ResourceConfig.create(
            num_experiments=16,
            outer_parallelization=True,
            num_cpus=4,
        )
        assert config.num_workers == 4  # Limited by CPUs
        assert config.cpus_per_experiment == 1

    def test_frozen(self):
        """Test that ResourceConfig is immutable (attrs frozen)."""
        config = ResourceConfig(num_cpus=4, num_workers=2, cpus_per_experiment=2)
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            config.num_cpus = 8


# =============================================================================
# OctoManager Tests
# =============================================================================


class TestOctoManager:
    """Tests for OctoManager orchestration."""

    def test_run_outer_experiments_sequential(self, octo_manager):
        """Test run outer experiments sequential."""
        with (
            patch("octopus.manager.core.shutdown_ray"),
            patch.object(WorkflowTaskRunner, "run") as mock_run,
        ):
            octo_manager.run_outer_experiments()
            assert mock_run.call_count == 1

    def test_run_outer_experiments_parallel(self, octo_manager):
        """Test run outer experiments with parallelization."""
        octo_manager.outer_parallelization = True

        with (
            patch("octopus.manager.core.shutdown_ray"),
            patch("octopus.manager.execution.run_parallel_outer_ray", return_value=[True]) as mock_ray,
        ):
            octo_manager.run_outer_experiments()
            mock_ray.assert_called_once()

    def test_run_single_experiment(self, octo_manager):
        """Test run single experiment."""
        octo_manager.run_single_experiment_num = 0

        with (
            patch("octopus.manager.core.shutdown_ray"),
            patch.object(WorkflowTaskRunner, "run") as mock_run,
        ):
            octo_manager.run_outer_experiments()
            mock_run.assert_called_once_with(octo_manager.base_experiments[0])

    def test_no_experiments_raises_error(self, mock_workflow):
        """Test that empty experiments raises ValueError."""
        manager = OctoManager(
            base_experiments=[],
            workflow=mock_workflow,
            log_dir=UPath("/tmp/test"),
        )
        with pytest.raises(ValueError, match="No experiments defined"):
            manager.run_outer_experiments()

    def test_ray_shutdown_on_error(self, octo_manager):
        """Test that Ray is shut down even if execution fails."""
        with (
            patch("octopus.manager.core.shutdown_ray") as mock_shutdown,
            patch.object(WorkflowTaskRunner, "run", side_effect=RuntimeError("Test error")),
        ):
            with pytest.raises(RuntimeError):
                octo_manager.run_outer_experiments()
            mock_shutdown.assert_called_once()


# =============================================================================
# WorkflowTaskRunner Tests
# =============================================================================


class TestWorkflowTaskRunner:
    """Tests for WorkflowTaskRunner."""

    def test_create_experiment(self, mock_workflow, mock_experiment, resources):
        """Test experiment creation from base experiment."""
        runner = WorkflowTaskRunner(mock_workflow, resources, UPath("/tmp/test"))
        task = mock_workflow[0]

        with patch("octopus.manager.workflow_runner.copy.deepcopy", return_value=mock_experiment):
            experiment = runner._create_experiment(mock_experiment, task)

        assert experiment.ml_module == task.module
        assert experiment.task_id == task.task_id
        assert experiment.num_assigned_cpus == resources.cpus_per_experiment

    def test_load_experiment(self, mock_workflow, mock_experiment, resources):
        """Test loading existing experiment."""
        runner = WorkflowTaskRunner(mock_workflow, resources, UPath("/tmp/test"))
        task = Mock(task_id=3)

        with (
            patch.object(PosixUPath, "exists", return_value=True),
            patch.object(OctoExperiment, "from_pickle", return_value=mock_experiment),
        ):
            loaded = runner._load_experiment(mock_experiment, task)
            assert loaded == mock_experiment

    def test_load_experiment_not_found(self, mock_workflow, mock_experiment, resources):
        """Test loading non-existent experiment raises error."""
        runner = WorkflowTaskRunner(mock_workflow, resources, UPath("/tmp/test"))
        task = Mock(task_id=3)

        with (
            patch.object(UPath, "exists", return_value=False),
            pytest.raises(FileNotFoundError),
        ):
            runner._load_experiment(mock_experiment, task)

    def test_apply_dependencies(self, mock_workflow, mock_experiment, resources):
        """Test applying dependencies from previous task."""
        runner = WorkflowTaskRunner(mock_workflow, resources, UPath("/tmp/test"))
        input_experiment = Mock(spec=OctoExperiment)
        input_experiment.selected_features = ["new_feature"]
        input_experiment.results = {"key": "value"}

        mock_experiment.depends_on_task = 1
        exp_path_dict = {1: UPath("/tmp/input_exp.pkl")}

        with (
            patch.object(PosixUPath, "exists", return_value=True),
            patch.object(OctoExperiment, "from_pickle", return_value=input_experiment),
        ):
            runner._apply_dependencies(mock_experiment, exp_path_dict)

        assert mock_experiment.feature_columns == ["new_feature"]
        assert mock_experiment.prior_results == {"key": "value"}
