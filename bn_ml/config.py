from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/bot.yaml")
PACKAGED_CONFIG_RESOURCE = "assets/bot.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    # When installed as a package, fall back to the bundled default config.
    if config_path is None or Path(config_path) == DEFAULT_CONFIG_PATH:
        bundled = resources.files("bn_ml").joinpath(PACKAGED_CONFIG_RESOURCE)
        if bundled.is_file():
            with bundled.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}

    raise FileNotFoundError(f"Config file not found: {path}")


def deep_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    node: Any = config
    for key in dotted_key.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
