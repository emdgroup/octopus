"""Machine learning models config."""

from typing import List, Type

from attrs import define, field, validators

from octopus.models.hyperparameter import Hyperparameter


def validate_hyperparameters(
    instance: "ModelConfig", attribute: str, value: List[Hyperparameter]
) -> None:
    """Validate hyperparameters.

    Make sure that the hyperparameters do not contain names
    that match the instance's n_jobs or model_seed.

    Args:
        instance: The instance of ModelConfig being validated.
        attribute: The name of the attribute being validated.
        value: The list of hyperparameters to validate.

    Raises:
        ValueError: If any hyperparameter's name matches n_jobs or model_seed.
    """
    forbidden_names = {instance.n_jobs, instance.model_seed}

    for hyperparameter in value:
        if hyperparameter.name in forbidden_names:
            raise ValueError(
                f"""Hyperparameter '{hyperparameter.name}' is not allowed in 'hyperparameters'."""  # noqa: E501
            )


@define
class ModelConfig:
    """Create model config."""

    name: str
    model_class: Type
    feature_method: str
    ml_type: str = field(
        validator=validators.in_(["regression", "classification", "timetoevent"])
    )
    hyperparameters: List[Hyperparameter] = field(validator=validate_hyperparameters)
    n_repeats: None | int = field(factory=lambda: None)
    n_jobs: None | str = field(factory=lambda: "n_jobs")
    model_seed: None | str = field(factory=lambda: "model_seed")
