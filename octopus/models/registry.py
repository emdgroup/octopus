"""Model Registry."""

from typing import ClassVar


class ModelRegistry:
    """Model Registry."""

    _models: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, name):
        """Register model."""

        def decorator(model_class):
            cls._models[name] = model_class
            return model_class

        return decorator

    @classmethod
    def get_all_models(cls):
        """Get all models."""
        return cls._models
