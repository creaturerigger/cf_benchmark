from typing import Callable, TypeVar
from .base_cf_method import BaseCounterfactualGenerationMethod


TMethod = TypeVar("TMethod", bound=BaseCounterfactualGenerationMethod)
METHOD_REGISTRY: dict[str, type[BaseCounterfactualGenerationMethod]] = {}


def register_method(name: str) -> Callable[[type[TMethod]], type[TMethod]]:
    def decorator(cls: type[TMethod]) -> type[TMethod]:
        METHOD_REGISTRY[name] = cls
        return cls
    return decorator


def get_method_class(name: str) -> type[BaseCounterfactualGenerationMethod]:
    if name not in METHOD_REGISTRY:
        raise ValueError(f"The method {name} is not found.")
    return METHOD_REGISTRY[name]
