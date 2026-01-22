"""Core models registry and inventory functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from octopus.exceptions import UnknownModelError

if TYPE_CHECKING:
    from collections.abc import Callable

    import optuna

    from .config import ModelConfig
    from .hyperparameter import Hyperparameter


class Models:
    """Central registry and inventory for models.

    Usage:
        # Get config
        cfg = Models.get_model_config("ExtraTreesClassifier")

        # Create Optuna params
        params = Models.create_trial_parameters(
            trial=trial,
            model_item=cfg,
            hyperparameters=cfg.hyperparameters,
            n_jobs=4,
            model_seed=42,
        )

        # Instantiate estimator
        model = Models.get_model_instance("ExtraTreesClassifier", params)
    """

    # Internal registry: model name -> function returning ModelConfig
    _config_factories: ClassVar[dict[str, Callable[[], ModelConfig]]] = {}

    # Internal cache: model name -> ModelConfig
    _model_configs: ClassVar[dict[str, ModelConfig]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[[], ModelConfig]], Callable[[], ModelConfig]]:
        """Register a model configuration factory function under a given name.

        Args:
            name: The name to register the model under.

        Returns:
            Decorator function.
        """

        def decorator(factory: Callable[[], ModelConfig]) -> Callable[[], ModelConfig]:
            if name in cls._config_factories:
                raise ValueError(f"Model '{name}' is already registered.")
            cls._config_factories[name] = factory
            return factory

        return decorator

    @classmethod
    def get_model_config(cls, name: str) -> ModelConfig:
        """Get model configuration by name.

        Args:
            name: The name of the model to retrieve.

        Returns:
            The ModelConfig instance for the specified model.

        Raises:
            UnknownModelError: If no model with the specified name is found.
        """
        # Return cached config if available
        if name in cls._model_configs:
            return cls._model_configs[name]

        # Lookup factory
        factory = cls._config_factories.get(name)
        if factory is None:
            available = ", ".join(sorted(cls._config_factories.keys()))
            raise UnknownModelError(
                f"Unknown model '{name}'. Available models are: {available}. Please check the model name and try again."
            )

        # Build config via factory and enforce name consistency
        config = factory()
        config.name = name
        cls._model_configs[name] = config
        return config

    @classmethod
    def get_model_instance(cls, name: str, params: dict[str, Any]):
        """Get model class by name and initialize it with the provided parameters.

        Args:
            name: The name of the model to retrieve.
            params: The parameters for model initialization.

        Returns:
            The initialized model instance.
        """
        model_config = cls.get_model_config(name)
        return model_config.model_class(**params)

    @classmethod
    def create_trial_parameters(
        cls,
        trial: optuna.trial.Trial,
        model_item: ModelConfig,
        hyperparameters: list[Hyperparameter],
        n_jobs: int,
        model_seed: int,
    ) -> dict[str, Any]:
        """Create Optuna parameters for a specific model.

        Args:
            trial: The Optuna trial object.
            model_item: The model configuration.
            hyperparameters: List of hyperparameters to suggest.
            n_jobs: Number of jobs for parallel execution.
            model_seed: Random seed for the model.

        Returns:
            Dictionary of parameter names to values.
        """
        params: dict[str, Any] = {}

        for hp in hyperparameters:
            unique_name = f"{hp.name}_{model_item.name}"
            params[hp.name] = hp.suggest(trial, unique_name)

        if model_item.n_jobs is not None:
            params[model_item.n_jobs] = n_jobs
        if model_item.model_seed is not None:
            params[model_item.model_seed] = model_seed

        return params
