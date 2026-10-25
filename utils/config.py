"""Configuration loading shared by training and experiment utilities.

SAFIR-Fuse ablations inherit from a common YAML file.  Keeping the loader in
this lightweight module lets protocol checks resolve configurations without
importing CUDA models or optional training dependencies.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def deep_update(target: dict, update: dict) -> dict:
    """Recursively merge ``update`` into ``target`` and return ``target``."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def load_config(path: str | Path) -> dict:
    """Load a YAML config, resolving an optional relative ``base_config``."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    base = config.pop("base_config", None)
    if not base:
        return config
    base_path = Path(base)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return deep_update(load_config(base_path), config)
