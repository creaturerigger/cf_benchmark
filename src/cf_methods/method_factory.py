from .base_cf_method import BaseCounterfactualGenerationMethod
from typing import Callable


class MethodFactory:
    _registry: dict[str, type[BaseCounterfactualGenerationMethod]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def decorator(method_cls):
            cls._registry[name] = method_cls
            return method_cls
        return decorator

    @classmethod
    def create(cls, cfg, *args, **kwargs) -> BaseCounterfactualGenerationMethod:
        name = cfg["method"]["name"]
        if name not in cls._registry:
            raise ValueError(f"Method `{name}` is not registered")
        return cls._registry[name](cfg, *args, **kwargs)
