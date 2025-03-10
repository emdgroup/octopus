"""Model Registry."""


class ModelRegistry:
    _models = {}

    @classmethod
    def register(cls, name):
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class

        return decorator

    @classmethod
    def get_model(cls, name):
        return cls._models.get(name)

    @classmethod
    def get_all_models(cls):
        return cls._models
