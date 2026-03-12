from typing import Callable

DATASET_REGISTRY: dict[str, Callable] = {}


def register_dataset(name: str):
    def decorator(fn):
        DATASET_REGISTRY[name] = fn
        return fn
    return decorator


def load_dataset(cfg):
    name = cfg["dataset"]["name"]
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset `{name}` is not registered.")
    return DATASET_REGISTRY[name](cfg)
