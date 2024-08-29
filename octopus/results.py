"""Results."""

from attrs import Factory, define, field, validators


@define
class ModuleResults:
    """Results."""

    id: str = field(validator=[validators.instance_of(str)])
    """Results str id."""

    model = field(default="")
    """Saved Model."""

    scores: dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Scores, dictionary."""

    predictions: dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Predictions, dictionary."""

    feature_importances: dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Feature importances, dictionary."""

    selected_features: list = field(
        default=Factory(list), validator=[validators.instance_of(list)]
    )
    """Feature importances, dictionary."""

    results: dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Other results, dictionary."""
