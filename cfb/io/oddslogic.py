"""OddsLogic archive helpers shared across models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
import requests

import oddslogic_loader


ODDSLOGIC_PHP_BASE = "https://odds.oddslogic.com/OddsLogic/sources/php/"
INJURY_LEAGUE_IDS = {
    "ncaaf": 1,
    "nfl": 2,
}

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


def fetch_injuries(
    *,
    league: str = "ncaaf",
    team_id: int = 0,
    player_name: str = "",
    time_zone: str = "UTC",
    timeout: int = 30,
) -> Dict[str, dict]:
    """Fetch the latest injury payload from OddsLogic."""

    league_key = str(league).lower()
    if league_key in INJURY_LEAGUE_IDS:
        league_id = INJURY_LEAGUE_IDS[league_key]
    else:
        try:
            league_id = int(league)
        except ValueError as exc:  # pragma: no cover - defensive fallback
            raise ValueError(f"Unknown OddsLogic league identifier: {league}") from exc

    payload = {
        "team_id": team_id,
        "player_name": player_name,
        "time_zone": time_zone,
        "league_id": league_id,
        "method": "get_injuries",
    }

    response = requests.post(ODDSLOGIC_PHP_BASE + "get_injuries.php", data=payload, timeout=timeout)
    response.raise_for_status()
    text = response.text.strip()
    if not text:
        return {}
    try:
        return response.json()
    except ValueError:
        return {}
