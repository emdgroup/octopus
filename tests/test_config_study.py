"""Test config study."""

import pytest

from octopus.config import ConfigStudy


def test_config_study_defaults():
    """Test default values."""
    "Test creation of a default study."
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
    assert config.metrics == [
        "AUCROC",
        "ACCBAL",
        "ACC",
        "LOGLOSS",
        "MAE",
        "MSE",
        "R2",
        "CI",
    ]


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
    "ml_type, target_metric",
    [
        ("classification", "MAE"),
        ("classification", "MSE"),
        ("classification", "R2"),
        ("classification", "CI"),
        ("regression", "AUCROC"),
        ("regression", "ACCBAL"),
        ("regression", "ACC"),
        ("regression", "LOGLOSS"),
        ("regression", "CI"),
        ("timetoevent", "MAE"),
        ("timetoevent", "MSE"),
        ("timetoevent", "R2"),
        ("timetoevent", "AUCROC"),
        ("timetoevent", "ACCBAL"),
        ("timetoevent", "ACC"),
        ("timetoevent", "LOGLOSS"),
    ],
)
def test_target_metric_validation(ml_type, target_metric):
    """Test if ml_type does not match target_metric."""
    with pytest.raises(ValueError):
        ConfigStudy(name="Test_study", ml_type=ml_type, target_metric=target_metric)
