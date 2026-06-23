"""Configuration management for the HCAI project.

Loads default.yaml and optionally merges local.yaml if present.
"""
from __future__ import annotations

import yaml
from pathlib import Path

from src.utils import paths

_CONFIG_DIR = paths.ROOT / "config"
_DEFAULT = _CONFIG_DIR / "default.yaml"
_LOCAL = _CONFIG_DIR / "local.yaml"


class Config(dict):
    """Dict-like configuration with attribute-based access."""

    def __getattr__(self, key: str) -> any:
        try:
            val = self[key]
            if isinstance(val, dict):
                return Config(val)
            return val
        except KeyError:
            raise AttributeError(f"No such configuration key: {key}")

    def __setattr__(self, key: str, value: any) -> None:
        self[key] = value


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override dictionary into base dictionary."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def load_config() -> Config:
    """Load default.yaml and merge with local.yaml if it exists."""
    cfg = {}
    if _DEFAULT.exists():
        with open(_DEFAULT, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    if _LOCAL.exists():
        with open(_LOCAL, "r", encoding="utf-8") as f:
            local = yaml.safe_load(f) or {}
            _deep_merge(cfg, local)
    return Config(cfg)


CFG = load_config()
