"""Model inventory."""

from typing import Any, Dict, List

import optuna
from attrs import define, field

from octopus.exceptions import UnknownModelError

from .config import ModelConfig
from .hyperparameter import Hyperparameter
from .registry import ModelRegistry


@define
class ModelInventory:
    """Model inventory."""

    models: Dict[str, Any] = field(factory=dict)
    _model_configs: Dict[str, ModelConfig] = field(factory=dict)

    def __attrs_post_init__(self):
        self.models = ModelRegistry.get_all_models()

    def get_model_config(self, name: str) -> ModelConfig | None:
        """Get model config."""
        if name not in self._model_configs:
            model_class = self.models.get(name)
            if model_class:
                config = model_class.get_model_config()
                config.name = name
                self._model_configs[name] = config
            else:
                raise UnknownModelError(
                    f"Unknown model '{name}'. "
                    f"Available models are: {', '.join(list(self.models.keys()))}. "
                    "Please check the model name and try again."
                )
        return self._model_configs[name]

    def get_model_by_name(self, name: str) -> ModelConfig | None:
        """Get model configuration by name.

        Args:
            name: The name of the model to retrieve.

        Returns:
            The ModelConfig instance with the specified name, or None if not found.
        """
        return self.get_model_config(name)

    def get_model_instance(self, name: str, params: dict) -> type:
        """Get model class by name and initializes it with the provided parameters.

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
        model_config = self.get_model_config(name)
        if model_config is None:
            raise ValueError(f"Model with name '{name}' not found")

        return model_config.model_class(**params)

    def get_feature_method(self, name: str) -> type:
        """Get feature method by name.

        If the model is a Gaussian Process, the kernel parameter (if provided)
        is converted from a string name to the corresponding kernel object.

        Args:
            name: The name of the model to retrieve.

        Returns:
            The feature method of a model.

        Raises:
            ValueError: If no model with the specified name is found.
        """
        model_config = self.get_model_config(name)
        if model_config is None:
            raise ValueError(f"Model with name '{name}' not found")

        return model_config.feature_method

    def get_models_by_type(self, ml_type: str) -> List[ModelConfig]:
        """Get list of model configurations by machine learning type.

        Args:
            ml_type: The type of machine learning model to filter
                by (e.g., "regression").

        Returns:
            A list of BaseModelConfig instances that match the specified
            machine learning type.
        """
        return [
            self.get_model_config(name)
            for name in self.models.keys()
            if self.get_model_config(name).ml_type == ml_type
        ]

    def create_trial_parameters(
        self,
        trial: optuna.trial.Trial,
        model_item: ModelConfig,
        hyperparameters: List[Hyperparameter],
        n_jobs: int,
        model_seed: int,
    ) -> Dict[str, Any]:
        """Create optuna parameters."""
        params = {}
        for hp in hyperparameters:
            parameter_name = hp.name
            unique_name = f"{hp.name}_{model_item.name}"

            if hp.type == "int":
                if hp.step is not None:
                    params[parameter_name] = trial.suggest_int(
                        name=unique_name, low=hp.low, high=hp.high, step=hp.step
                    )
                elif hp.log is not None:
                    params[parameter_name] = trial.suggest_int(
                        name=unique_name, low=hp.low, high=hp.high, log=hp.log
                    )
                else:
                    params[parameter_name] = trial.suggest_int(
                        name=unique_name, low=hp.low, high=hp.high
                    )

            elif hp.type == "float":
                if hp.step is not None:
                    params[parameter_name] = trial.suggest_float(
                        name=unique_name, low=hp.low, high=hp.high, step=hp.step
                    )
                elif hp.log is not None:
                    params[parameter_name] = trial.suggest_float(
                        name=unique_name, low=hp.low, high=hp.high, log=hp.log
                    )
                else:
                    params[parameter_name] = trial.suggest_float(
                        name=unique_name, low=hp.low, high=hp.high
                    )

            elif hp.type == "categorical":
                params[parameter_name] = trial.suggest_categorical(
                    name=unique_name, choices=hp.choices
                )
            elif hp.type == "fixed":
                params[parameter_name] = hp.value
            else:
                raise ValueError(f"HP type '{hp.type}' not supported")

        if model_item.n_jobs is not None:
            params[model_item.n_jobs] = n_jobs
        if model_item.model_seed is not None:
            params[model_item.model_seed] = model_seed

        return params
