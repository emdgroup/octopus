"""Test that octopus core package imports work correctly.

These tests run early in the test suite to catch import issues quickly.
They verify that all core modules can be imported without errors.
"""


def test_core_imports() -> None:
    """Test all core package imports work without optional dependencies."""
    import octopus  # noqa: F401, PLC0415
    from octopus.experiment import OctoExperiment  # noqa: F401, PLC0415
    from octopus.manager import OctoManager  # noqa: F401, PLC0415
    from octopus.metrics import MetricsInventory  # noqa: F401, PLC0415
    from octopus.models import Models  # noqa: F401, PLC0415
    from octopus.predict import OctoPredict  # noqa: F401, PLC0415
    from octopus.study import OctoStudy  # noqa: F401, PLC0415
    from octopus.task import Task  # noqa: F401, PLC0415


def test_core_functionality() -> None:
    """Test that core functionality can be instantiated."""
    from octopus.metrics import MetricsInventory  # noqa: PLC0415
    from octopus.models import Models  # noqa: PLC0415

    # Verify model registry is accessible
    num_models = len(Models._config_factories)
    assert num_models > 0, "Models registry should have at least one model"

    # Verify metrics inventory can be instantiated
    metrics = MetricsInventory()
    assert len(metrics.metrics) > 0, "MetricsInventory should have at least one metric"
