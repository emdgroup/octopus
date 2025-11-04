"""Test config manager."""

from octopus.config import ConfigManager


def test_config_manager_defaults():
    """Test default values for config manager."""
    config = ConfigManager()
    assert config.outer_parallelization is True
    assert config.run_single_experiment_num == -1
    assert config.reserve_cpus == 1
