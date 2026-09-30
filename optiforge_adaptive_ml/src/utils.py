"""
General utilities: deterministic seeding, memory auditing, and configuration loading.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any, Dict, Set
import numpy as np
import yaml


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across standard library, numpy, and external environments
    for strict deterministic reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_deep_memory_bytes(obj: Any, seen: Set[int] | None = None) -> int:
    """
    Recursively estimate deep memory usage of an object in bytes,
    including numpy arrays, pandas dataframes/series, dicts, lists, and primitives.
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    # Fast path for numpy arrays
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)

    # Fast path for pandas DataFrame / Series
    if hasattr(obj, "memory_usage"):
        try:
            mem = obj.memory_usage(deep=True)
            if hasattr(mem, "sum"):
                return int(mem.sum())
            return int(mem)
        except Exception:
            pass

    # Recursive traversal for containers
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += get_deep_memory_bytes(k, seen)
            size += get_deep_memory_bytes(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += get_deep_memory_bytes(item, seen)
    elif hasattr(obj, "__dict__"):
        size += get_deep_memory_bytes(vars(obj), seen)

    return size


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
