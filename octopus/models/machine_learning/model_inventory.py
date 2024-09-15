"""Model inventory."""

from typing import Any, Dict, List

import optuna
from attrs import define, field

from octopus.models.machine_learning.config import ModelConfig


@define
class ModelInventory:
    """Model inventory."""

    models: List[ModelConfig] = field(factory=list)

    def add_model(self, model_config: ModelConfig) -> None:
        """Add model configuration to the inventory.

        Args:
            model_config: The BaseModelConfig instance to add.
        """
        self.models.append(model_config)

    def get_model_by_name(self, name: str) -> ModelConfig | None:
        """Get model configuration by name.

        Args:
            name: The name of the model to retrieve.

        Returns:
            The ModelConfig instance with the specified name, or None if not found.
        """
        return next((model for model in self.models if model.name == name), None)

    def get_model_instance(self, name: str, params: dict) -> type:
        """Get model class by name and initializes it with the provided parameters.

        If the model is a Gaussian Process, the kernel parameter (if provided)
        is converted from a string name to the corresponding kernel object.

        Args:
            name: The name of the model to retrieve.
            params: The parameters of the model. For Gaussian Process models,
                the 'kernel' parametershould be provided as a string and will
                be converted to the corresponding kernel object.

        Returns:
            The initialized model instance with the specified name and parameters.

        Raises:
            ValueError: If no model with the specified name is found.
        """
        model_config = next(
            (model for model in self.models if model.name == name), None
        )
        if model_config is None:
            raise ValueError(f"Model with name '{name}' not found")

        return model_config.model_class(**params)

    def get_models_by_type(self, ml_type: str) -> List[ModelConfig]:
        """Get list of model configurations by machine learning type.

        Args:
            ml_type: The type of machine learning model to filter
                by (e.g., "regression").

        Returns:
            A list of BaseModelConfig instances that match the specified
            machine learning type.
        """
        return [config for config in self.models if config.ml_type == ml_type]

    def create_optuna_parameters(
        self,
        trial: optuna.trial.Trial,
        model_name: str,
        fixed_global_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create optuna parameters."""
        model_config = self.get_model_by_name(model_name)
        if not model_config:
            raise ValueError(f"Model '{model_name}' not found in the inventory.")

        params = {}
        for hp in model_config.hyperparameters:
            parameter_name = hp.name
            unique_name = f"{hp.name}_{model_name}"

            if hp.type == "int":
                params[parameter_name] = trial.suggest_int(
                    name=unique_name, low=hp.low, high=hp.high, log=hp.log
                )
            elif hp.type == "float":
                params[parameter_name] = trial.suggest_float(
                    name=unique_name, low=hp.low, high=hp.high, log=hp.log
                )
            elif hp.type == "categorical":
                params[parameter_name] = trial.suggest_categorical(
                    name=unique_name, choices=hp.choices
                )
            elif hp.type == "fixed":
                params[parameter_name] = hp.value
            else:
                raise ValueError(f"HP type '{hp.type}' not supported")

        # Overwrite model parameters specified by global settings
        translate = model_config.translate
        for key, value in fixed_global_parameters.items():
            if translate.get(key) != "NA":  # NA=ignore
                params[translate[key]] = value

        return params
