from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = "config/config.yaml"


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must parse to a dictionary at the top level.")

    return cfg


def require(cfg: Dict[str, Any], key_path: str) -> Any:
    """
    Strict getter: require(cfg, "llm.generation.alias") -> value or hard-fail.
    """
    cur: Any = cfg
    for k in key_path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing required config key: {key_path}")
        cur = cur[k]
    return cur
