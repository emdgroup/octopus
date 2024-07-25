"""Test config study."""

import pytest

from octopus.config import ConfigStudy


def test_config_study_defaults():
    """Test default values."""
    config = ConfigStudy(
        name="Test_study", ml_type="classification", target_metric="AUCROC"
    )
    assert config.name == "Test_study"
    assert config.ml_type == "classification"
    assert config.target_metric == "AUCROC"
    assert config.path == "./studies/"
    assert config.start_with_empty_study is True
    assert config.n_folds_outer == 5
    assert config.datasplit_seed_outer == 0
    assert config.silently_overwrite_study is False
    assert config.metrics == ["AUCROC"]


def test_config_study_custom_values():
    """Test custom values."""
    config = ConfigStudy(
        name="Test_study_custom",
        ml_type="regression",
        target_metric="MSE",
        path="/custom/path/",
        start_with_empty_study=False,
        n_folds_outer=10,
        datasplit_seed_outer=42,
        silently_overwrite_study=True,
        metrics=["MAE", "MSE"],
    )
    assert config.name == "Test_study_custom"
    assert config.ml_type == "regression"
    assert config.target_metric == "MSE"
    assert config.path == "/custom/path/"
    assert config.start_with_empty_study is False
    assert config.n_folds_outer == 10
    assert config.datasplit_seed_outer == 42
    assert config.silently_overwrite_study is True
    assert config.metrics == ["MAE", "MSE"]


@pytest.mark.parametrize(
    "ml_type, target_metric, metrics, allowed",
    [
        ("classification", "AUCROC", ["AUCROC", "ACCBAL", "MAE"], False),
        ("classification", "ACC", ["ACC", "ACCBAL", "LOGLOSS"], True),
        ("regression", "R2", ["R2", "MAE", "MSE"], True),
        ("classification", "ACC", ["AUCROC", "ACC", "LOGLOSS"], True),
        ("regression", "MAE", ["MAE", "MSE"], True),
        ("classification", "R2", ["ACC", "ACCBAL"], False),
        ("regression", "ACC", ["MAE", "MSE"], False),
        ("timetoevent", "AUCROC", ["CI"], False),
        ("regression", "LOGLOSS", ["MAE", "R2"], False),
        ("classification", "MSE", ["AUCROC", "ACC"], False),
        ("timetoevent", "CI", ["CI"], True),
    ],
)
def test_metric_validation(ml_type, target_metric, metrics, allowed):
    """Test if ml_type does not match target_metric."""
    if allowed:
        ConfigStudy(
            name="Test_study",
            ml_type=ml_type,
            target_metric=target_metric,
            metrics=metrics,
        )
    else:
        with pytest.raises(ValueError):
            ConfigStudy(
                name="Test_study",
                ml_type=ml_type,
                target_metric=target_metric,
                metrics=metrics,
            )
