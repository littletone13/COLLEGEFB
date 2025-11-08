"""Configuration helpers for the modelling toolkit."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:  # Optional dependency: PyYAML is not always available in minimal CI images
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised indirectly via import failure
    yaml = None  # type: ignore[assignment]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "defaults.yaml"


def load_config(path: Optional[str | Path] = None, *, env_var: str = "CFB_CONFIG") -> Dict[str, Any]:
    """Load configuration from YAML.

    Parameters
    ----------
    path :
        Optional explicit configuration file path. When omitted, the
        function looks for ``env_var`` (default ``CFB_CONFIG``) and
        finally falls back to ``config/defaults.yaml`` bundled with the
        repository.
    env_var :
        Environment variable that can override the configuration path.

    Returns
    -------
    dict
        Parsed configuration dictionary (empty when the file is blank).
    """

    candidate = path or os.environ.get(env_var)
    if candidate:
        config_path = Path(candidate).expanduser()
    else:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if yaml is None:  # pragma: no cover - defensive guard for optional dependency
        raise ModuleNotFoundError(
            "PyYAML is required to load configuration files. Install 'pyyaml' to use this helper."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping; got {type(data).__name__}")
    return data
