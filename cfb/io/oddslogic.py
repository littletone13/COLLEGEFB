"""OddsLogic archive helpers shared across models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

import oddslogic_loader


def load_archive(archive_dir: Path | str) -> pd.DataFrame:
    """Load the flattened OddsLogic archive DataFrame."""

    return oddslogic_loader.load_archive_dataframe(Path(archive_dir))


def build_closing_lookup(
    df: pd.DataFrame,
    classification: Optional[Sequence[str] | str] = None,
    providers: Optional[Sequence[str]] = None,
) -> Dict[Tuple[pd.Timestamp.date, str, str], Dict[str, object]]:
    """Build a closing-line lookup keyed by (date, home_key, away_key)."""

    return oddslogic_loader.build_closing_lookup(df, classification, providers=providers)


def summarize_coverage(
    df: pd.DataFrame,
    classification: Optional[Sequence[str] | str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aggregate coverage statistics for the archive."""

    return oddslogic_loader.summarize_coverage(df, classification=classification)
