"""Helpers for loading PFF CSV exports."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


def load_latest_csv(data_dirs: Iterable[Path], patterns: Iterable[str]) -> pd.DataFrame:
    """Load the most recently modified CSV matching the provided patterns."""

    candidates: List[Path] = []
    for directory in data_dirs:
        directory = directory.expanduser()
        if not directory.exists():
            continue
        for pattern in patterns:
            paths = list(directory.glob(pattern)) if any(ch in pattern for ch in "*?[]") else [directory / pattern]
            for path in paths:
                if path.exists():
                    candidates.append(path)
    if not candidates:
        joined = ", ".join(patterns)
        raise FileNotFoundError(f"No CSV found matching patterns: {joined}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pd.read_csv(candidates[0])
