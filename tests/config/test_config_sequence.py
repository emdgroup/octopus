"""Test config sequence."""

import pytest

from octopus.config import ConfigSequence
from octopus.modules import Mrmr, Octo


@pytest.fixture
def octo_sequence():
    """Create fixture for octo sequence."""
    return Octo(
        description="step_1",
        models=["RandomForestRegressor", "XGBRegressor"],
    )


@pytest.fixture
def mrmr_sequence():
    """Create fixture for mrmr sequence."""
    return Mrmr(description="step2_mrmr")


# Test cases for ConfigSequence
class TestConfigSequence:
    """Test config sequence."""

    @pytest.mark.parametrize(
        "sequence_items, expected_exception",
        [
            (["octo_sequence"], None),
            (["octo_sequence", "mrmr_sequence"], None),
            ([], ValueError),
            (["octo_sequence", "invalid_item"], TypeError),
            (["octo_sequence", "mrmr_sequence", "string"], TypeError),
            (None, TypeError),
        ],
    )
    def test_initialization(self, request, sequence_items, expected_exception):
        """Test class initialization."""
        # Handle the case where fixture_names is None
        if sequence_items is None:
            test_sequence_items = None
        else:
            test_sequence_items = []
            for name in sequence_items:
                # check if item is a fixture
                if name in request._fixturemanager._arg2fixturedefs.keys():
                    test_sequence_items.append(request.getfixturevalue(name))
                else:
                    test_sequence_items.append(name)

        if expected_exception is None:
            config = ConfigSequence(sequence_items=test_sequence_items)
            assert config.sequence_items == test_sequence_items
        else:
            with pytest.raises(expected_exception):
                ConfigSequence(sequence_items=test_sequence_items)
