"""OddsLogic archive helpers shared across models."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

import oddslogic_loader
from scripts import oddslogic_scraper


ODDSLOGIC_PHP_BASE = "https://odds.oddslogic.com/OddsLogic/sources/php/"
INJURY_LEAGUE_IDS = {
    "ncaaf": 1,
    "ncaaf_fbs": 1,
    "ncaaf-fbs": 1,
    "ncaaf_fcs": 1,
    "ncaaf-fcs": 1,
    "college_football": 1,
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


_LIVE_SPORTSBOOK_CACHE: Dict[int, oddslogic_scraper.Sportsbook] = {}


def _timestamp_to_iso(value: Optional[int]) -> Optional[str]:
    if value in (None, 0):
        return None
    try:
        ts = dt.datetime.fromtimestamp(int(value), tz=dt.timezone.utc).replace(microsecond=0)
        iso = ts.isoformat()
        return iso.replace("+00:00", "Z")
    except (ValueError, OSError, OverflowError):
        return None


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (ValueError, TypeError, OverflowError):
        return None


def _provider_label(name: Optional[str], sportsbook_id: Optional[int]) -> str:
    if isinstance(name, str):
        candidate = name.strip()
        if candidate:
            return candidate
    if sportsbook_id is not None:
        return f"id_{sportsbook_id}"
    return "unknown"


def _get_live_sportsbooks(refresh: bool = False) -> Dict[int, oddslogic_scraper.Sportsbook]:
    global _LIVE_SPORTSBOOK_CACHE
    if not _LIVE_SPORTSBOOK_CACHE or refresh:
        raw = oddslogic_scraper.http_get(oddslogic_scraper.SPORTSBOOKS_ENDPOINT)
        if not raw:
            raise RuntimeError("OddsLogic sportsbooks feed returned no data.")
        _LIVE_SPORTSBOOK_CACHE = oddslogic_scraper.parse_sportsbooks(raw)
    return _LIVE_SPORTSBOOK_CACHE


def _fetch_live_lines_dataframe(dates: Iterable[dt.date]) -> pd.DataFrame:
    sportsbook_map = _get_live_sportsbooks()
    frames: list[pd.DataFrame] = []
    seen: set[dt.date] = set()
    for day in dates:
        if not isinstance(day, dt.date) or day in seen:
            continue
        seen.add(day)
        date_str = day.isoformat()
        schedule_raw = oddslogic_scraper.http_get(oddslogic_scraper.SCHEDULE_ENDPOINT.format(date=date_str))
        if not schedule_raw:
            continue
        schedule = oddslogic_scraper.parse_schedule(schedule_raw)
        lines_raw = oddslogic_scraper.http_get(oddslogic_scraper.LINES_ENDPOINT.format(date=date_str))
        if not lines_raw:
            continue
        records = oddslogic_scraper.parse_lines_payload(lines_raw, sportsbook_map, schedule)
        if not records:
            continue
        frame = pd.DataFrame([asdict(record) for record in records])
        if frame.empty:
            continue
        frame["source_date"] = day
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["classification"] = df["classification"].str.lower()
    df["kickoff_date"] = pd.to_datetime(df.get("start_datetime"), errors="coerce").dt.date
    df["timestamp"] = pd.to_numeric(df.get("timestamp"), errors="coerce").fillna(0).astype("int64")
    df["sportsbook_id"] = pd.to_numeric(df.get("sportsbook_id"), errors="coerce").astype("Int64")
    df["sportsbook_name"] = df.get("sportsbook_name").fillna("")
    df["provider_label"] = [
        _provider_label(name, _safe_int(sid)) for name, sid in zip(df["sportsbook_name"], df["sportsbook_id"])
    ]
    return df


def _extract_latest_and_openers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        empty = pd.DataFrame()
        return empty, empty
    df = df.sort_values("timestamp")
    key_cols = ["kickoff_date", "home_key", "away_key", "provider_label"]
    latest = df.drop_duplicates(key_cols, keep="last")
    open_candidates = df[df["is_opener"]]
    if open_candidates.empty:
        openers = df.drop_duplicates(key_cols, keep="first")
    else:
        openers = open_candidates.sort_values("timestamp").drop_duplicates(key_cols, keep="first")
    return latest, openers


def _build_provider_index(df: pd.DataFrame) -> Dict[Tuple[dt.date, str, str], Dict[str, pd.Series]]:
    mapping: Dict[Tuple[dt.date, str, str], Dict[str, pd.Series]] = {}
    if df.empty:
        return mapping
    for row in df.itertuples():
        key = (row.kickoff_date, row.home_key, row.away_key)
        provider_map = mapping.setdefault(key, {})
        provider_map[row.provider_label] = row
    return mapping


def summarize_live_lines(
    df: pd.DataFrame,
    classification: str,
    *,
    providers: Optional[Sequence[str]] = None,
) -> Dict[Tuple[dt.date, str, str], Dict[str, object]]:
    """Summarize live OddsLogic lines into a provider lookup keyed by teams."""

    if df.empty:
        return {}

    classification_lower = classification.lower()
    subset = df[df["classification"] == classification_lower].copy()
    if subset.empty:
        return {}

    provider_filter = None
    if providers:
        provider_filter = {name.strip().lower() for name in providers if name and name.strip()}
        if provider_filter:
            subset = subset[subset["provider_label"].str.lower().isin(provider_filter)]
            if subset.empty:
                return {}

    subset = subset.dropna(subset=["kickoff_date", "home_key", "away_key"])
    if subset.empty:
        return {}

    spread_latest, spread_openers = _extract_latest_and_openers(subset[subset["row_type"] == "spread"])
    total_latest, total_openers = _extract_latest_and_openers(subset[subset["row_type"] == "total"])

    spread_open_map = _build_provider_index(spread_openers)
    total_open_map = _build_provider_index(total_openers)

    result: Dict[Tuple[dt.date, str, str], Dict[str, object]] = {}

    for row in spread_latest.itertuples():
        key = (row.kickoff_date, row.home_key, row.away_key)
        entry = result.setdefault(
            key,
            {
                "kickoff_date": row.kickoff_date,
                "start_datetime": row.start_datetime,
                "home_key": row.home_key,
                "away_key": row.away_key,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "providers": {},
            },
        )
        provider = entry["providers"].setdefault(
            row.provider_label,
            {
                "sportsbook_id": _safe_int(row.sportsbook_id),
                "sportsbook_name": row.sportsbook_name or row.provider_label,
                "spread_value": None,
                "spread_price": None,
                "total_value": None,
                "total_price": None,
                "spread_updated": None,
                "total_updated": None,
                "open_spread_value": None,
                "open_spread_price": None,
                "open_total_value": None,
                "open_total_price": None,
            },
        )
        provider["spread_value"] = row.line_value
        provider["spread_price"] = row.line_price
        provider["spread_updated"] = _timestamp_to_iso(row.timestamp)
        opener_row = spread_open_map.get(key, {}).get(row.provider_label)
        if opener_row is not None:
            provider["open_spread_value"] = opener_row.line_value
            provider["open_spread_price"] = opener_row.line_price
        elif provider["open_spread_value"] is None:
            provider["open_spread_value"] = row.line_value
            provider["open_spread_price"] = row.line_price

    for row in total_latest.itertuples():
        key = (row.kickoff_date, row.home_key, row.away_key)
        entry = result.setdefault(
            key,
            {
                "kickoff_date": row.kickoff_date,
                "start_datetime": row.start_datetime,
                "home_key": row.home_key,
                "away_key": row.away_key,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "providers": {},
            },
        )
        provider = entry["providers"].setdefault(
            row.provider_label,
            {
                "sportsbook_id": _safe_int(row.sportsbook_id),
                "sportsbook_name": row.sportsbook_name or row.provider_label,
                "spread_value": None,
                "spread_price": None,
                "total_value": None,
                "total_price": None,
                "spread_updated": None,
                "total_updated": None,
                "open_spread_value": None,
                "open_spread_price": None,
                "open_total_value": None,
                "open_total_price": None,
            },
        )
        provider["total_value"] = row.line_value
        provider["total_price"] = row.line_price
        provider["total_updated"] = _timestamp_to_iso(row.timestamp)
        opener_row = total_open_map.get(key, {}).get(row.provider_label)
        if opener_row is not None:
            provider["open_total_value"] = opener_row.line_value
            provider["open_total_price"] = opener_row.line_price
        elif provider["open_total_value"] is None:
            provider["open_total_value"] = row.line_value
            provider["open_total_price"] = row.line_price

    for entry in result.values():
        provider_values = entry["providers"].values()
        spreads = [prov["spread_value"] for prov in provider_values if prov["spread_value"] is not None]
        totals = [prov["total_value"] for prov in provider_values if prov["total_value"] is not None]
        open_spreads = [
            prov["open_spread_value"] for prov in provider_values if prov["open_spread_value"] is not None
        ]
        open_totals = [
            prov["open_total_value"] for prov in provider_values if prov["open_total_value"] is not None
        ]
        entry["spread"] = float(np.mean(spreads)) if spreads else None
        entry["total"] = float(np.mean(totals)) if totals else None
        entry["open_spread"] = float(np.mean(open_spreads)) if open_spreads else None
        entry["open_total"] = float(np.mean(open_totals)) if open_totals else None

    return result


def fetch_live_market_lookup(
    dates: Iterable[dt.date],
    classification: str,
    *,
    providers: Optional[Sequence[str]] = None,
) -> Dict[Tuple[dt.date, str, str], Dict[str, object]]:
    """Fetch current OddsLogic lines for the requested dates and classification."""

    date_list = [day for day in dates if isinstance(day, dt.date)]
    if not date_list:
        return {}
    df = _fetch_live_lines_dataframe(date_list)
    if df.empty:
        return {}
    return summarize_live_lines(df, classification, providers=providers)


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
