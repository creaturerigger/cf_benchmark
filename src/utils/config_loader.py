from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml


CONFIGS_DIR = Path("configs")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates *base*)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    dataset: str,
    model: Optional[str] = None,
    cf_method: Optional[str] = None,
    experiment: Optional[str] = None,
) -> dict[str, Any]:
    """Load and merge configs by component name.

    Args:
        dataset:    name of the dataset config (e.g. "adult").
        model:      name of the model config (e.g. "pytorch_classifier").
        cf_method:  name of the CF method config (e.g. "dice").
        experiment: name of the experiment config
                    (e.g. "robustness_experiment").

    Returns:
        Merged dict with top-level keys per component.
    """
    cfg: dict[str, Any] = {}

    _deep_merge(cfg, load_yaml(CONFIGS_DIR / "dataset" / f"{dataset}.yaml"))

    if model is not None:
        _deep_merge(cfg, load_yaml(CONFIGS_DIR / "model" / f"{model}.yaml"))

    if cf_method is not None:
        _deep_merge(
            cfg, load_yaml(CONFIGS_DIR / "cf_method" / f"{cf_method}.yaml"),
        )

    if experiment is not None:
        exp_cfg = load_yaml(
            CONFIGS_DIR / "experiment" / f"{experiment}.yaml"
        )
        # Remove reference-only keys that would collide with the
        # actual component configs already loaded above.
        for ref_key in ("dataset", "model", "cf_method"):
            exp_cfg.pop(ref_key, None)
        _deep_merge(cfg, exp_cfg)

    return cfg


def load_all_dataset_configs() -> dict[str, dict[str, Any]]:
    """Load all dataset configs from configs/dataset/.

    Returns:
        Mapping of dataset name -> config dict.
    """
    dataset_dir = CONFIGS_DIR / "dataset"
    configs = {}
    for path in sorted(dataset_dir.glob("*.yaml")):
        cfg = load_yaml(path)
        name = cfg.get("dataset", {}).get("name", path.stem)
        configs[name] = cfg
    return configs
