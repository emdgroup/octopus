"""Test config sequence."""

import pytest

from octopus.config import ConfigWorkflow
from octopus.modules import Mrmr, Octo


@pytest.fixture
def octo_sequence():
    """Create fixture for octo sequence."""
    return Octo(
        task_id=0,
        depends_on_task=-1,
        description="step_1",
        models=["RandomForestRegressor", "XGBRegressor"],
    )


@pytest.fixture
def mrmr_sequence():
    """Create fixture for mrmr sequence."""
    return Mrmr(task_id=1, depends_on_task=0, description="step2_mrmr")


# Test cases for ConfigWorkflow
class TestConfigWorkflow:
    """Test config sequence."""

    @pytest.mark.parametrize(
        "tasks, expected_exception",
        [
            (["octo_sequence"], None),
            (["octo_sequence", "mrmr_sequence"], None),
            ([], ValueError),
            (["octo_sequence", "invalid_item"], TypeError),
            (["octo_sequence", "mrmr_sequence", "string"], TypeError),
            (None, TypeError),
        ],
    )
    def test_initialization(self, request, tasks, expected_exception):
        """Test class initialization."""
        # Handle the case where fixture_names is None
        if tasks is None:
            test_tasks = None
        else:
            test_tasks = []
            for name in tasks:
                # check if item is a fixture
                if name in request._fixturemanager._arg2fixturedefs:
                    test_tasks.append(request.getfixturevalue(name))
                else:
                    test_tasks.append(name)

        if expected_exception is None:
            config = ConfigWorkflow(tasks=test_tasks)
            assert config.tasks == test_tasks
        else:
            with pytest.raises(expected_exception):
                ConfigWorkflow(tasks=test_tasks)
