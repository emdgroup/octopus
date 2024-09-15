"""Machine learning models config."""

from typing import Dict, List, Type

from attrs import Attribute, define, field, validators

from octopus.models.machine_learning.hyperparameter import Hyperparameter


def validate_hyperparameters(
    instance: "ModelConfig",
    attribute: "Attribute",
    value: List[Hyperparameter],
) -> None:
    """Validate that hyperparameter names do not conflict with translated names.

    Args:
        instance: The instance of BaseModelConfig.
        attribute: The attribute being validated.
        value: The list of Hyperparameter instances.

    Raises:
        ValueError: If a hyperparameter name conflicts with a translated name.
    """
    translated_names = set(instance.translate.values())
    for hp in value:
        if hp.name in translated_names:
            raise ValueError(
                f"Hyperparameter name '{hp.name}' conflicts with a translated name."
            )


@define
class ModelConfig:
    """Create model config."""

    name: str
    model_class: Type
    feature_method: str
    ml_type: str = field(validator=validators.in_(["regression", "classification"]))
    hyperparameters: List[Hyperparameter] = field(validator=validate_hyperparameters)
    translate: Dict[str, str] = field(factory=dict)
    n_repeats: None | int = field(factory=lambda: None)
