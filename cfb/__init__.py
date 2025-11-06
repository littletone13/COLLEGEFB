"""Shared utilities for College Football modelling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def load_config(path: Optional[str | Path] = None, *, env_var: str = "CFB_CONFIG") -> Dict[str, Any]:
    """Proxy to :func:`cfb.config.load_config` with a lazy import.

    The package previously imported ``cfb.config`` eagerly, which requires
    the optional ``pyyaml`` dependency. Some lightweight environments (like
    the execution sandbox used in CI for these exercises) do not provide
    that package, making ``import cfb`` fail even when the caller only needs
    modules that do not depend on YAML parsing. By postponing the import
    until the function is invoked we allow the rest of the package—
    including :mod:`cfb.player_prop_sim`—to be imported without PyYAML
    installed. Callers that do rely on configuration loading continue to
    receive the original behaviour (an informative ``ModuleNotFoundError``)
    when the dependency is missing.
    """

    from .config import load_config as _load_config

    return _load_config(path, env_var=env_var)


__all__ = ["load_config"]
