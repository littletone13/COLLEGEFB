"""FBS matchup projection tool using CFBD advanced metrics.

This script fetches play-by-play efficiency (PPA), SP+/Elo ratings, and
other team-level metrics from the CollegeFootballData API to build
offense/defense/power ratings for current FBS programs. It mirrors the
interface of ``fcs.py`` but is focused on the FBS universe and relies on
publicly available analytics that Rufus Peabody/Andrew Mack-style
handicappers typically reference.

Prerequisites
-------------
* Install dependencies: ``pip install requests pandas numpy`` (and
  ``cfbd`` if you want to explore additional endpoints).
* Set the environment variable ``CFBD_API_KEY`` to your personal API key
  (the raw string; the script prepends ``Bearer`` automatically).

Usage examples
--------------
* List current FBS ratings:
    ``python3 fbs.py --year 2024 --list``
* Project a matchup:
    ``python3 fbs.py Georgia Alabama``

Note: The CFBD API currently exposes advanced metrics only for FBS teams.
Attempting to query FCS opponents will raise a lookup error.
"""
from __future__ import annotations

import argparse
import math
import warnings
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests

import market_anchor
import oddslogic_loader
from cfb import market as market_utils
from cfb import weather as weather_utils
from cfb.config import load_config
from cfb.injuries import penalties_for_player
from cfb.io import oddslogic as oddslogic_io
from cfb.io.cfbd import CFBDClient, BASE_URL as CFBD_BASE_URL
from cfb.model import RatingBook, RatingConstants, fit_linear_calibrations, fit_probability_sigma

PFF_DATA_DIR = Path("~/Desktop/PFFMODEL_FBS").expanduser()
PFF_COMBINED_DATA_DIR = Path("~/Desktop/POST WEEK 9 FBS & FCS DATA").expanduser()

BASE_URL = CFBD_BASE_URL

CONFIG = load_config()
FBS_CONFIG = CONFIG.get("fbs", {})
_MARKET_CONFIG = FBS_CONFIG.get("market", {})
_WEATHER_CONFIG = FBS_CONFIG.get("weather", {})

MARKET_SPREAD_WEIGHT = float(_MARKET_CONFIG.get("spread_weight", 0.4))
MARKET_TOTAL_WEIGHT = float(_MARKET_CONFIG.get("total_weight", 0.4))

_DEFAULT_WEATHER_COEFFS = {
    "wind_high": -0.3462,
    "cold": -0.0546,
    "heat": -0.1730,
    "rain": 0.0,  # Clamp to zero to prevent weather nudges from inflating totals
    "snow": -0.0500,
    "humid": -0.1960,
    "dewpos": -0.0554,
}

_configured_coeffs = _WEATHER_CONFIG.get("coeffs", {}) if isinstance(_WEATHER_CONFIG.get("coeffs", {}), dict) else {}
WEATHER_COEFFS = {key: float(_configured_coeffs.get(key, value)) for key, value in _DEFAULT_WEATHER_COEFFS.items()}
for key, value in _configured_coeffs.items():
    if key not in WEATHER_COEFFS:
        WEATHER_COEFFS[key] = float(value)

_configured_bounds = _WEATHER_CONFIG.get("clamp_bounds", (-12.0, 6.0))
if isinstance(_configured_bounds, (list, tuple)) and len(_configured_bounds) == 2:
    WEATHER_CLAMP_BOUNDS = (float(_configured_bounds[0]), float(_configured_bounds[1]))
else:
    WEATHER_CLAMP_BOUNDS = (-12.0, 6.0)
WEATHER_MAX_TOTAL_ADJ = float(_WEATHER_CONFIG.get("max_total_adjustment", 10.0))

_RATING_CONFIG = FBS_CONFIG.get("ratings", {}) if isinstance(FBS_CONFIG.get("ratings"), dict) else {}


def _rating_constants_from_config() -> RatingConstants:
    return RatingConstants(
        avg_total=float(_RATING_CONFIG.get("avg_total", 53.5109)),
        home_field_advantage=float(_RATING_CONFIG.get("home_field_advantage", 3.4066)),
        offense_factor=float(_RATING_CONFIG.get("offense_factor", 6.2171)),
        defense_factor=float(_RATING_CONFIG.get("defense_factor", 5.0088)),
        power_factor=float(_RATING_CONFIG.get("power_factor", 2.1035)),
        spread_sigma=float(_RATING_CONFIG.get("spread_sigma", 15.0)),
    )

ODDSLOGIC_ARCHIVE_DIR = Path(os.environ.get("ODDSLOGIC_ARCHIVE_DIR", "oddslogic_ncaa_all"))

_INJURY_WARNING_EMITTED = False


def _flatten_ppa(records: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for entry in records:
        offense = entry.get("offense", {})
        defense = entry.get("defense", {})
        rows.append(
            {
                "team": entry["team"],
                "ppa_offense": offense.get("overall"),
                "ppa_passing": offense.get("passing"),
                "ppa_rushing": offense.get("rushing"),
                "ppa_defense": defense.get("overall"),
                "ppa_def_pass": defense.get("passing"),
                "ppa_def_rush": defense.get("rushing"),
            }
        )
    return pd.DataFrame(rows)


def _flatten_sp(records: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for entry in records:
        offense = entry.get("offense", {})
        defense = entry.get("defense", {})
        rows.append(
            {
                "team": entry["team"],
                "sp_rating": entry.get("rating"),
                "sp_offense": offense.get("rating"),
                "sp_defense": defense.get("rating"),
                "sp_special": entry.get("specialTeams"),
            }
        )
    return pd.DataFrame(rows)


def _flatten_elo(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team": entry["team"],
            "elo": entry.get("elo"),
            "elo_prob": entry.get("winProbability"),
        }
        for entry in records
    )


def _flatten_fpi(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team": entry["team"],
            "fpi": entry.get("fpi"),
            "fpi_offense": entry.get("offense"),
            "fpi_defense": entry.get("defense"),
        }
        for entry in records
    )


def _flatten_game_ppa(records: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for entry in records:
        offense = entry.get("offense") or {}
        defense = entry.get("defense") or {}
        rows.append(
            {
                "game_id": entry.get("gameId"),
                "team": entry.get("team"),
                "opponent": entry.get("opponent"),
                "season": entry.get("season"),
                "week": entry.get("week"),
                "season_type": entry.get("seasonType"),
                "off_ppa_overall": offense.get("overall"),
                "off_ppa_passing": offense.get("passing"),
                "off_ppa_rushing": offense.get("rushing"),
                "def_ppa_overall": defense.get("overall"),
                "def_ppa_passing": defense.get("passing"),
                "def_ppa_rushing": defense.get("rushing"),
            }
    )
    return pd.DataFrame(rows)


def _load_latest_csv(data_dirs: list[Path], patterns: Iterable[str]) -> pd.DataFrame:
    candidates: list[Path] = []
    for directory in data_dirs:
        if not directory.exists():
            continue
        for pattern in patterns:
            if any(ch in pattern for ch in "*?[]"):
                paths = list(directory.glob(pattern))
            else:
                paths = [directory / pattern]
            for path in paths:
                if path.exists():
                    candidates.append(path)
    if not candidates:
        joined = ", ".join(patterns)
        raise FileNotFoundError(f"No CSV found matching patterns: {joined}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pd.read_csv(candidates[0])


def _aggregate_team_metrics(
    df: pd.DataFrame,
    *,
    team_col: str,
    weight_col: Optional[str],
    weighted_metrics: Dict[str, str],
    sum_metrics: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    sum_metrics = sum_metrics or {}
    records: Dict[str, Dict[str, float]] = {}
    for team, group in df.groupby(team_col):
        record: Dict[str, float] = {}
        weights = None
        if weight_col and weight_col in group:
            weights = group[weight_col].fillna(0)
        for out_col, src_col in weighted_metrics.items():
            series = group[src_col]
            if weights is not None and weights.sum() > 0:
                value = float((series * weights).sum() / weights.sum())
            else:
                value = float(series.mean()) if not series.dropna().empty else float("nan")
            record[out_col] = value
        for out_col, src_col in sum_metrics.items():
            record[out_col] = float(group[src_col].fillna(0).sum())
        records[team] = record
    return pd.DataFrame.from_dict(records, orient="index").reset_index().rename(columns={"index": team_col})


def load_pff_team_metrics(data_dir: Path = PFF_DATA_DIR) -> pd.DataFrame:
    data_dir = data_dir.expanduser()
    source_dirs: list[Path] = []
    if PFF_COMBINED_DATA_DIR.exists():
        source_dirs.append(PFF_COMBINED_DATA_DIR.expanduser())
    source_dirs.append(data_dir)

    try:
        receiving = _load_latest_csv(
            source_dirs,
            [
                "receiving_summary*FBS*FCS*.csv",
                "receiving_summary_FBS.csv",
                "receiving_summaryFBS.csv",
                "receiving_summary.csv",
            ],
        )
        blocking = _load_latest_csv(
            source_dirs,
            [
                "offense_blocking*FBS*FCS*.csv",
                "offense_blocking FBS.csv",
                "offense_blocking_FBS.csv",
                "offense_blocking.csv",
            ],
        )
        defense = _load_latest_csv(
            source_dirs,
            [
                "defense_summary*FBS*FCS*.csv",
                "defense_summary FBS.csv",
                "defense_summary_FBS.csv",
                "defense_summary.csv",
            ],
        )
        special = _load_latest_csv(
            source_dirs,
            [
                "special_teams_summary*FBS*FCS*.csv",
                "special_teams_summary_FBS.csv",
                "special_teams_summary.csv",
            ],
        )
    except FileNotFoundError:
        return pd.DataFrame(columns=["team"])

    rec = _aggregate_team_metrics(
        receiving,
        team_col="team_name",
        weight_col="routes" if "routes" in receiving.columns else None,
        weighted_metrics={
            "receiving_grade_route": "grades_pass_route",
            "receiving_grade_offense": "grades_offense",
            "receiving_yprr": "yprr",
            "receiving_catch_rate": "caught_percent",
        },
        sum_metrics={
            "receiving_targets": "targets",
            "receiving_yards": "yards",
        },
    )

    blk = _aggregate_team_metrics(
        blocking,
        team_col="team_name",
        weight_col="snap_counts_offense" if "snap_counts_offense" in blocking.columns else None,
        weighted_metrics={
            "blocking_grade_pass": "grades_pass_block",
            "blocking_grade_run": "grades_run_block",
            "blocking_pbe": "pbe",
        },
        sum_metrics={"pressures_allowed": "pressures_allowed"},
    )

    dfn = _aggregate_team_metrics(
        defense,
        team_col="team_name",
        weight_col="snap_counts_defense" if "snap_counts_defense" in defense.columns else None,
        weighted_metrics={
            "defense_grade_overall": "grades_defense",
            "defense_grade_coverage": "grades_coverage_defense",
            "defense_grade_run": "grades_run_defense",
            "defense_grade_pass_rush": "grades_pass_rush_defense",
        },
        sum_metrics={"defense_sacks": "sacks"},
    )

    st = _aggregate_team_metrics(
        special,
        team_col="team_name",
        weight_col=None,
        weighted_metrics={"special_grade_misc": "grades_misc_st"},
    )

    pff = rec.merge(blk, on="team_name", how="outer")
    pff = pff.merge(dfn, on="team_name", how="outer")
    pff = pff.merge(st, on="team_name", how="outer")
    return pff.rename(columns={"team_name": "team"})


def fetch_team_metrics(
    year: int,
    api_key: Optional[str] = None,
    *,
    through_week: Optional[int] = None,
    season_type: str = "regular",
) -> pd.DataFrame:
    client = CFBDClient(api_key)

    ppa_raw = client.get("/ppa/teams", year=year, seasonType=season_type)
    ppa = _flatten_ppa(ppa_raw)
    sp = _flatten_sp(client.get("/ratings/sp", year=year))
    elo = _flatten_elo(client.get("/ratings/elo", year=year))
    fpi = _flatten_fpi(client.get("/ratings/fpi", year=year))

    teams = ppa.merge(sp, on="team", how="outer")
    teams = teams.merge(elo, on="team", how="outer")
    teams = teams.merge(fpi, on="team", how="outer")

    try:
        game_records = client.get("/ppa/games", year=year, seasonType=season_type)
        games = _flatten_game_ppa(game_records)
        if through_week is not None:
            games = games[games["week"].fillna(0) <= through_week]
        adj = compute_opponent_adjusted_ppa(ppa, games)
    except requests.HTTPError:
        adj = pd.DataFrame(columns=["team"])

    if not adj.empty:
        teams = teams.merge(adj, on="team", how="left")

    pff = load_pff_team_metrics()
    if not pff.empty:
        teams = teams.merge(pff, on="team", how="left")

    injuries = fetch_injury_impacts(
        year,
        api_key or os.environ.get("CFBD_API_KEY"),
        week=through_week,
        season_type=season_type,
    )
    if not injuries.empty:
        teams = teams.merge(injuries, on="team", how="left")

    for col in teams.columns:
        if col == "team":
            continue
        teams[col] = pd.to_numeric(teams[col], errors="coerce")

    if "injury_offense_penalty" not in teams.columns:
        teams["injury_offense_penalty"] = 0.0
    else:
        teams["injury_offense_penalty"] = teams["injury_offense_penalty"].fillna(0.0)
    if "injury_defense_penalty" not in teams.columns:
        teams["injury_defense_penalty"] = 0.0
    else:
        teams["injury_defense_penalty"] = teams["injury_defense_penalty"].fillna(0.0)

    return teams


def add_z_scores(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            continue
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        z = 0.0 if std == 0 or math.isnan(std) else (df[col] - mean) / std
        df[f"{col}_z"] = z
    return df


def fetch_games(year: int, api_key: str, *, season_type: str = "regular") -> list[dict]:
    client = CFBDClient(api_key)
    return client.get("/games", year=year, seasonType=season_type)


def fetch_game_weather(
    year: int,
    api_key: str,
    *,
    week: Optional[int] = None,
    season_type: str = "regular",
    team: Optional[str] = None,
    conference: Optional[str] = None,
    classification: Optional[str] = None,
) -> Dict[int, dict]:
    """Return a lookup of CFBD weather forecasts keyed by game id."""

    client = CFBDClient(api_key)
    params: Dict[str, object] = {"year": year}
    if season_type:
        params["seasonType"] = season_type
    if week is not None:
        params["week"] = week
    if team:
        params["team"] = team
    if conference:
        params["conference"] = conference
    if classification:
        params["classification"] = classification

    data = client.get("/games/weather", **params)
    return {entry.get("id"): entry for entry in data if entry.get("id") is not None}


def fetch_market_lines(
    year: int,
    api_key: str,
    *,
    week: Optional[int] = None,
    season_type: str = "regular",
    team: Optional[str] = None,
    conference: Optional[str] = None,
    classification: Optional[str] = None,
    providers: Optional[Iterable[str]] = None,
) -> Dict[int, dict]:
    """Return aggregated sportsbook lines keyed by game id.

    Parameters
    ----------
    year : int
        Season year to query.
    api_key : str
        CFBD API key (Bearer token without the prefix).
    week : Optional[int], optional
        Week filter. Defaults to None (all weeks).
    season_type : str, optional
        Season type (regular/postseason). Defaults to regular.
    team : Optional[str], optional
        Filter to a specific team.
    conference : Optional[str], optional
        Filter to a specific conference.
    classification : Optional[str], optional
        Team classification ("fbs" or "fcs").
    providers : Optional[Iterable[str]], optional
        Subset of sportsbook providers to retain. Case-insensitive. When
        omitted, all providers returned by CFBD are included.
    """

    params: Dict[str, object] = {"year": year}
    if season_type:
        params["seasonType"] = season_type
    if week is not None:
        params["week"] = week
    if team:
        params["team"] = team
    if conference:
        params["conference"] = conference
    if classification:
        params["classification"] = classification

    client = CFBDClient(api_key)
    provider_filter = None
    if providers:
        provider_filter = {str(name).lower() for name in providers if str(name).strip()}

    lookup: Dict[int, dict] = {}
    for entry in client.get("/lines", **params):
        game_id = entry.get("id")
        if not game_id:
            continue
        lines = entry.get("lines") or []
        provider_names: set[str] = set()
        provider_lines: Dict[str, dict] = {}
        for line in lines:
            provider = line.get("provider")
            provider_normalized = str(provider or "").strip()
            if not provider_normalized:
                continue
            if provider_filter and provider_normalized.lower() not in provider_filter:
                continue
            provider_names.add(provider_normalized)

            spread_raw = _coerce_float(line.get("spread"))
            total_raw = _coerce_float(line.get("overUnder"))
            home_ml = _coerce_float(line.get("homeMoneyline"))
            away_ml = _coerce_float(line.get("awayMoneyline"))
            last_updated = line.get("lastUpdated") or line.get("updated")

            provider_lines[provider_normalized] = {
                "spread_home": -spread_raw if spread_raw is not None else None,
                "spread_away": spread_raw,
                "total": total_raw,
                "home_moneyline": home_ml,
                "away_moneyline": away_ml,
                "last_updated": last_updated,
            }
        if provider_filter and not provider_names:
            continue

        spread_values = [
            info.get("spread_home") for info in provider_lines.values() if info.get("spread_home") is not None
        ]
        total_values = [
            info.get("total") for info in provider_lines.values() if info.get("total") is not None
        ]
        spread_value: Optional[float] = float(np.mean(spread_values)) if spread_values else None
        total_value: Optional[float] = float(np.mean(total_values)) if total_values else None

        lookup[game_id] = {
            "game_id": game_id,
            "home_team": entry.get("homeTeam"),
            "away_team": entry.get("awayTeam"),
            "start_date": entry.get("startDate"),
            "neutral_site": entry.get("neutralSite"),
            "spread": spread_value,
            "total": total_value,
            "providers": sorted(provider_names),
            "provider_lines": provider_lines,
        }
    return lookup


def compute_opponent_adjusted_ppa(
    season_ppa: pd.DataFrame,
    game_ppa: pd.DataFrame,
) -> pd.DataFrame:
    if season_ppa.empty or game_ppa.empty:
        return pd.DataFrame(columns=["team"])

    season_lookup = {}
    for team_name, row in season_ppa.set_index("team").iterrows():
        season_lookup[str(team_name).lower()] = row

    def _lookup(team: Optional[str], key: str) -> Optional[float]:
        if not team:
            return None
        record = season_lookup.get(str(team).lower())
        if record is None:
            return None
        value = record.get(key)
        return float(value) if pd.notna(value) else None

    df = game_ppa.copy()
    df.dropna(subset=["team", "opponent"], inplace=True)

    df["opp_def_overall"] = df["opponent"].apply(lambda t: _lookup(t, "ppa_defense"))
    df["opp_def_passing"] = df["opponent"].apply(lambda t: _lookup(t, "ppa_def_pass"))
    df["opp_def_rushing"] = df["opponent"].apply(lambda t: _lookup(t, "ppa_def_rush"))
    df["opp_off_overall"] = df["opponent"].apply(lambda t: _lookup(t, "ppa_offense"))
    df["opp_off_passing"] = df["opponent"].apply(lambda t: _lookup(t, "ppa_passing"))
    df["opp_off_rushing"] = df["opponent"].apply(lambda t: _lookup(t, "ppa_rushing"))

    for col in [
        "off_ppa_overall",
        "off_ppa_passing",
        "off_ppa_rushing",
        "def_ppa_overall",
        "def_ppa_passing",
        "def_ppa_rushing",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["adj_off_overall"] = df["off_ppa_overall"] - df["opp_def_overall"]
    df["adj_off_passing"] = df["off_ppa_passing"] - df["opp_def_passing"]
    df["adj_off_rushing"] = df["off_ppa_rushing"] - df["opp_def_rushing"]
    df["adj_def_overall"] = df["def_ppa_overall"] - df["opp_off_overall"]
    df["adj_def_passing"] = df["def_ppa_passing"] - df["opp_off_passing"]
    df["adj_def_rushing"] = df["def_ppa_rushing"] - df["opp_off_rushing"]

    agg = (
        df.groupby("team")[
            [
                "adj_off_overall",
                "adj_off_passing",
                "adj_off_rushing",
                "adj_def_overall",
                "adj_def_passing",
                "adj_def_rushing",
            ]
        ]
        .mean()
        .reset_index()
    )
    agg = agg.rename(
        columns={
            "adj_off_overall": "adj_ppa_offense",
            "adj_off_passing": "adj_ppa_offense_pass",
            "adj_off_rushing": "adj_ppa_offense_rush",
            "adj_def_overall": "adj_ppa_defense",
            "adj_def_passing": "adj_ppa_defense_pass",
            "adj_def_rushing": "adj_ppa_defense_rush",
        }
    )
    return agg


def _graphql_request(api_key: str, query: str, variables: dict) -> Optional[dict]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        resp = requests.post(
            BASE_URL + "/graphql",
            json={"query": query, "variables": variables},
            headers=headers,
            timeout=60,
        )
    except requests.RequestException:
        return None
    content_type = resp.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        return None
    try:
        payload = resp.json()
    except ValueError:
        return None
    if payload.get("errors"):
        first_error = payload["errors"][0]
        message = first_error.get("message", "unknown GraphQL error")
        warnings.warn(f"CFBD GraphQL error: {message}", RuntimeWarning)
        return None
    return payload.get("data")


def fetch_injury_impacts(
    year: int,
    api_key: Optional[str],
    *,
    week: Optional[int] = None,
    season_type: str = "regular",
) -> pd.DataFrame:
    """Aggregate injury penalties from CFBD and OddsLogic."""

    global _INJURY_WARNING_EMITTED

    # CFBD GraphQL feed
    cfbd_df = pd.DataFrame(columns=["team"])
    if api_key:
        variables = {
            "season": year,
            "seasonType": season_type.upper() if isinstance(season_type, str) else season_type,
            "week": week,
        }
        query = """
        query TeamInjuries($season: Int!, $seasonType: SeasonType, $week: Int) {
          teamInjuries(season: $season, seasonType: $seasonType, week: $week) {
            team
            status
            player {
              name
              position
            }
          }
        }
        """
        data = _graphql_request(api_key, query, variables)
        entries: list[dict] = []
        if not data:
            if not _INJURY_WARNING_EMITTED:
                warnings.warn(
                    "CFBD GraphQL injuries unavailable; skipping injury adjustments.",
                    RuntimeWarning,
                )
                _INJURY_WARNING_EMITTED = True
        else:
            raw = data.get("teamInjuries") or data.get("injuries") or []
            for entry in raw:
                team = entry.get("team")
                status_raw = (entry.get("status") or "").strip()
                player = entry.get("player") or entry.get("athlete") or {}
                position = (player.get("position") or entry.get("position") or "").strip().upper()
                offense_pen, defense_pen = penalties_for_player(status_raw, position)
                if not team or (offense_pen == 0.0 and defense_pen == 0.0):
                    continue
                entries.append(
                    {
                        "team": team,
                        "injury_offense_penalty": offense_pen,
                        "injury_defense_penalty": defense_pen,
                    }
                )
        if entries:
            cfbd_df = (
                pd.DataFrame(entries)
                .groupby("team")[["injury_offense_penalty", "injury_defense_penalty"]]
                .sum()
                .reset_index()
            )

    # OddsLogic feed (multi-book injury news)
    try:
        ol_payload = oddslogic_io.fetch_injuries(league="ncaaf_fbs")
    except requests.RequestException:
        ol_payload = {}

    ol_entries: list[dict] = []
    for info in ol_payload.values():
        team = info.get("player_team")
        status_raw = info.get("injury_status") or ""
        custom_text = info.get("custom_text") or ""
        if not team:
            continue
        position = (info.get("player_position") or "").upper()
        offense_pen, defense_pen = penalties_for_player(status_raw, position, custom_text=custom_text)
        if offense_pen == 0.0 and defense_pen == 0.0:
            continue
        ol_entries.append(
            {
                "team": team,
                "injury_offense_penalty": offense_pen,
                "injury_defense_penalty": defense_pen,
            }
        )

    if ol_entries:
        ol_df = (
            pd.DataFrame(ol_entries)
            .groupby("team")[["injury_offense_penalty", "injury_defense_penalty"]]
            .sum()
            .reset_index()
        )
        if cfbd_df.empty:
            cfbd_df = ol_df
        else:
            cfbd_df = (
                pd.concat([cfbd_df, ol_df])
                .groupby("team")[["injury_offense_penalty", "injury_defense_penalty"]]
                .sum()
                .reset_index()
            )

    return cfbd_df

def _coerce_float(value: Optional[float]) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    filtered = ''.join(ch for ch in str(value) if ch.isdigit() or ch in {'.', '-', '+'})
    if not filtered:
        return None
    try:
        return float(filtered)
    except ValueError:
        return None


def apply_market_prior(
    result: Dict[str, float],
    market: Optional[dict],
    *,
    prob_sigma: float,
    spread_weight: float = MARKET_SPREAD_WEIGHT,
    total_weight: float = MARKET_TOTAL_WEIGHT,
) -> Dict[str, float]:
    return market_utils.apply_market_prior(
        result,
        market,
        prob_sigma=prob_sigma,
        spread_weight=spread_weight,
        total_weight=total_weight,
    )


def apply_weather_adjustment(result: Dict[str, float], weather: Optional[dict]) -> Dict[str, float]:
    return weather_utils.apply_weather_adjustment(
        result,
        weather,
        coeffs=WEATHER_COEFFS,
        clamp_bounds=WEATHER_CLAMP_BOUNDS,
        max_total_adjustment=WEATHER_MAX_TOTAL_ADJ,
    )


def build_ratings(teams: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "ppa_offense",
        "ppa_passing",
        "ppa_rushing",
        "ppa_defense",
        "ppa_def_pass",
        "ppa_def_rush",
        "adj_ppa_offense",
        "adj_ppa_offense_pass",
        "adj_ppa_offense_rush",
        "adj_ppa_defense",
        "adj_ppa_defense_pass",
        "adj_ppa_defense_rush",
        "sp_rating",
        "sp_offense",
        "sp_defense",
        "sp_special",
        "fpi",
        "fpi_offense",
        "fpi_defense",
        "elo",
        "receiving_grade_route",
        "receiving_grade_offense",
        "receiving_yprr",
        "blocking_grade_pass",
        "blocking_grade_run",
        "blocking_pbe",
        "defense_grade_overall",
        "defense_grade_pass_rush",
        "defense_grade_coverage",
        "defense_grade_run",
        "special_grade_misc",
    ]
    add_z_scores(teams, candidates)

    for metric in candidates:
        z_col = f"{metric}_z"
        if z_col in teams.columns:
            teams[z_col] = teams[z_col].fillna(0.0)

    teams["offense_rating"] = (
        0.20 * teams.get("ppa_offense_z", 0.0)
        + 0.11 * teams.get("ppa_passing_z", 0.0)
        + 0.11 * teams.get("ppa_rushing_z", 0.0)
        + 0.18 * teams.get("sp_offense_z", 0.0)
        + 0.08 * teams.get("fpi_offense_z", 0.0)
        + 0.05 * teams.get("sp_special_z", 0.0)
        + 0.06 * teams.get("adj_ppa_offense_z", 0.0)
        + 0.04 * teams.get("adj_ppa_offense_pass_z", 0.0)
        + 0.04 * teams.get("adj_ppa_offense_rush_z", 0.0)
        + 0.10 * teams.get("receiving_grade_route_z", 0.0)
        + 0.05 * teams.get("receiving_yprr_z", 0.0)
        + 0.07 * teams.get("blocking_grade_pass_z", 0.0)
        + 0.04 * teams.get("blocking_grade_run_z", 0.0)
        + 0.02 * teams.get("blocking_pbe_z", 0.0)
    )

    teams["defense_rating"] = (
        -0.26 * teams.get("ppa_defense_z", 0.0)
        - 0.16 * teams.get("ppa_def_pass_z", 0.0)
        - 0.16 * teams.get("ppa_def_rush_z", 0.0)
        - 0.18 * teams.get("sp_defense_z", 0.0)
        - 0.09 * teams.get("fpi_defense_z", 0.0)
        - 0.07 * teams.get("adj_ppa_defense_z", 0.0)
        - 0.06 * teams.get("adj_ppa_defense_pass_z", 0.0)
        - 0.06 * teams.get("adj_ppa_defense_rush_z", 0.0)
        - 0.06 * teams.get("defense_grade_overall_z", 0.0)
        - 0.05 * teams.get("defense_grade_pass_rush_z", 0.0)
        - 0.05 * teams.get("defense_grade_coverage_z", 0.0)
        - 0.03 * teams.get("defense_grade_run_z", 0.0)
    )

    teams["offense_rating"] = teams["offense_rating"] - teams.get("injury_offense_penalty", 0.0)
    teams["defense_rating"] = teams["defense_rating"] + teams.get("injury_defense_penalty", 0.0)

    teams["power_rating"] = (
        teams["offense_rating"]
        + teams["defense_rating"]
        + 0.35 * teams.get("sp_rating_z", 0.0)
        + 0.25 * teams.get("fpi_z", 0.0)
        + 0.15 * teams.get("elo_z", 0.0)
    )

    return teams


def compute_power_adjustments(
    ratings: pd.DataFrame,
    games: Iterable[dict],
    *,
    constants: RatingConstants,
    up_to_week: int,
    reg: float = 4.0,
) -> Dict[str, float]:
    teams = ratings["team"].tolist()
    index = {team: i for i, team in enumerate(teams)}

    book = RatingBook(ratings, constants)
    rows = []
    residuals = []
    for game in games:
        if not game.get("completed"):
            continue
        if game.get("week", 0) >= up_to_week:
            continue
        if game.get("homeClassification") != "fbs" or game.get("awayClassification") != "fbs":
            continue
        home = game["homeTeam"]
        away = game["awayTeam"]
        if home not in index or away not in index:
            continue
        home_points = game.get("homePoints")
        away_points = game.get("awayPoints")
        if home_points is None or away_points is None:
            continue
        try:
            pred = book.predict(home, away, neutral_site=game.get("neutralSite", False))
        except KeyError:
            continue
        residual = (home_points - away_points) - pred["spread_home_minus_away"]
        row = np.zeros(len(teams))
        row[index[home]] = 1.0
        row[index[away]] = -1.0
        rows.append(row)
        residuals.append(residual)

    if not rows:
        return {}

    A = np.vstack(rows)
    b = np.array(residuals)
    reg_matrix = math.sqrt(reg) * np.eye(len(teams))
    A_reg = np.vstack([A, reg_matrix])
    b_reg = np.concatenate([b, np.zeros(len(teams))])
    adjustments = np.linalg.lstsq(A_reg, b_reg, rcond=None)[0]
    scale = 0.25
    clipped = {}
    for team, adj in zip(teams, adjustments):
        if abs(adj) <= 1e-6:
            continue
        value = float(max(min(adj * scale, 6.0), -6.0))
        if abs(value) <= 1e-3:
            continue
        clipped[team] = value
    return clipped


def compute_market_anchor_adjustments(
    book: RatingBook,
    games: Iterable[dict],
    *,
    config: market_anchor.MarketAnchorConfig,
    up_to_week: Optional[int] = None,
) -> Dict[str, float]:
    classification = (config.classification or "").lower()
    anchor_candidates: list[dict] = []
    for game in games:
        if game.get("completed") is not True:
            continue
        if up_to_week is not None and (game.get("week") or 0) >= up_to_week:
            continue
        home_cls = (game.get("homeClassification") or "").lower()
        away_cls = (game.get("awayClassification") or "").lower()
        if classification == "fbs":
            if home_cls != "fbs" or away_cls != "fbs":
                continue
        elif classification == "fcs":
            if home_cls != "fcs" or away_cls != "fcs":
                continue
        elif classification:
            if home_cls != classification or away_cls != classification:
                continue
        start_dt = game.get("startDate") or game.get("startTime")
        if not start_dt:
            continue
        try:
            pred = book.predict(game["homeTeam"], game["awayTeam"], neutral_site=game.get("neutralSite", False))
        except KeyError:
            continue
        anchor_candidates.append(
            {
                "id": game.get("id"),
                "startDate": start_dt,
                "homeTeam": game["homeTeam"],
                "awayTeam": game["awayTeam"],
                "model_spread": pred["spread_home_minus_away"],
            }
        )
    if not anchor_candidates:
        return {}
    return market_anchor.derive_power_adjustments(anchor_candidates, config=config)


def fit_probability_sigma(book: RatingBook, games: Iterable[dict]) -> Optional[float]:
    spreads = []
    outcomes = []
    for game in games:
        if game.get("completed") is not True:
            continue
        if game.get("homeClassification") != "fbs" or game.get("awayClassification") != "fbs":
            continue
        home_points = game.get("homePoints")
        away_points = game.get("awayPoints")
        if home_points is None or away_points is None:
            continue
        try:
            pred = book.predict(game["homeTeam"], game["awayTeam"], neutral_site=game.get("neutralSite", False))
        except KeyError:
            continue
        spreads.append(pred["spread_home_minus_away"])
        outcomes.append(1.0 if home_points > away_points else 0.0)
    if len(spreads) < 25:
        return None
    spreads = np.array(spreads)
    outcomes = np.array(outcomes)
    candidates = np.linspace(6.0, 25.0, 200)
    best_sigma = None
    best_loss = float("inf")
    for sigma in candidates:
        scaled = spreads / (sigma * math.sqrt(2))
        erf_vals = np.vectorize(math.erf)(scaled)
        probs = 0.5 * (1.0 + erf_vals)
        loss = np.mean((probs - outcomes) ** 2)
        if loss < best_loss:
            best_loss = loss
            best_sigma = sigma
    if best_sigma is None:
        return None
    return float(best_sigma)


def _lookup_team_row(ratings: pd.DataFrame, team: str) -> Optional[pd.Series]:
    if "team" not in ratings.columns:
        return None
    team_lower = team.lower()
    mask = ratings["team"].str.lower() == team_lower
    if mask.any():
        return ratings.loc[mask].iloc[0]
    mask = ratings["team"].str.contains(team, case=False, regex=False)
    if mask.any():
        return ratings.loc[mask].iloc[0]
    return None


def refit_scoring_constants(
    ratings: pd.DataFrame,
    games: Iterable[dict],
    power_adjustments: Dict[str, float],
    constants: RatingConstants,
) -> RatingConstants:
    margin_rows: list[tuple[float, float, float, float]] = []
    margin_targets: list[float] = []
    total_rows: list[tuple[float, float, float]] = []
    total_targets: list[float] = []

    for game in games:
        if game.get("completed") is not True:
            continue
        if game.get("homeClassification") != "fbs" or game.get("awayClassification") != "fbs":
            continue
        home_points = game.get("homePoints")
        away_points = game.get("awayPoints")
        if home_points is None or away_points is None:
            continue
        home_team = game["homeTeam"]
        away_team = game["awayTeam"]
        home_row = _lookup_team_row(ratings, home_team)
        away_row = _lookup_team_row(ratings, away_team)
        if home_row is None or away_row is None:
            continue

        off_diff = float(home_row.get("offense_rating", 0.0) - away_row.get("offense_rating", 0.0))
        def_diff = float(home_row.get("defense_rating", 0.0) - away_row.get("defense_rating", 0.0))
        power_diff = float(
            home_row.get("power_rating", 0.0)
            + power_adjustments.get(home_row.get("team", ""), 0.0)
            - away_row.get("power_rating", 0.0)
            - power_adjustments.get(away_row.get("team", ""), 0.0)
        )
        neutral = bool(game.get("neutralSite"))
        home_flag = 0.0 if neutral else 1.0

        margin_rows.append((home_flag, off_diff, def_diff, power_diff))
        margin_targets.append(float(home_points - away_points))

        off_sum = float(home_row.get("offense_rating", 0.0) + away_row.get("offense_rating", 0.0))
        neg_def_sum = float(-(home_row.get("defense_rating", 0.0) + away_row.get("defense_rating", 0.0)))
        total_rows.append((1.0, off_sum, neg_def_sum))
        total_targets.append(float(home_points + away_points))

    if len(margin_rows) < 25 or len(total_rows) < 25:
        return constants

    X_margin = np.array(margin_rows)
    y_margin = np.array(margin_targets)
    try:
        X_margin_base = X_margin[:, :3]
        coeff_margin_base, _, _, _ = np.linalg.lstsq(X_margin_base, y_margin, rcond=None)
    except np.linalg.LinAlgError:
        return constants
    residual_margin = y_margin - X_margin_base @ coeff_margin_base
    power_vec = X_margin[:, 3]
    denom = float(np.dot(power_vec, power_vec))
    if denom > 1e-6:
        power_factor = float(np.dot(power_vec, residual_margin) / denom)
    else:
        power_factor = constants.power_factor

    X_total = np.array(total_rows)
    y_total = np.array(total_targets)
    try:
        coeff_total, _, _, _ = np.linalg.lstsq(X_total, y_total, rcond=None)
    except np.linalg.LinAlgError:
        return constants

    home_field = float(coeff_margin_base[0])
    offense_factor_spread = float(coeff_margin_base[1])
    defense_factor_spread = float(coeff_margin_base[2])

    avg_total = float(coeff_total[0])
    offense_factor_total = float(coeff_total[1])
    defense_factor_total = float(coeff_total[2])

    def _fallback(value: float, default: float, low: float, high: float) -> float:
        if not math.isfinite(value):
            return default
        return float(np.clip(value, low, high))

    offense_factor = 0.5 * (offense_factor_spread + offense_factor_total)
    defense_factor = 0.5 * (defense_factor_spread + defense_factor_total)
    offense_factor = _fallback(offense_factor, constants.offense_factor, 0.0, 8.0)
    defense_factor = _fallback(defense_factor, constants.defense_factor, 0.0, 8.0)
    if offense_factor < 1e-6:
        offense_factor = constants.offense_factor
    if defense_factor < 1e-6:
        defense_factor = constants.defense_factor

    return RatingConstants(
        avg_total=_fallback(avg_total, constants.avg_total, 30.0, 90.0),
        home_field_advantage=_fallback(home_field, constants.home_field_advantage, 0.0, 6.0),
        offense_factor=offense_factor,
        defense_factor=defense_factor,
        power_factor=_fallback(power_factor, constants.power_factor, -2.0, 4.0),
        spread_sigma=constants.spread_sigma,
    )


def build_rating_book(
    year: int,
    *,
    api_key: Optional[str] = None,
    adjust_week: Optional[int] = None,
    calibration_games: Optional[Iterable[dict]] = None,
    constants: Optional[RatingConstants] = None,
    market_anchor_config: Optional[market_anchor.MarketAnchorConfig] = None,
) -> tuple[pd.DataFrame, RatingBook]:
    constants = constants or _rating_constants_from_config()
    calibration_list = list(calibration_games) if calibration_games else []
    teams_raw = fetch_team_metrics(
        year,
        api_key=api_key,
        through_week=adjust_week,
    )
    ratings = build_ratings(teams_raw)
    adjustments: Dict[str, float] = {}
    games_for_anchor: list[dict] = calibration_list.copy()
    cfbd_games: list[dict] = []
    if adjust_week is not None and adjust_week > 1:
        key = api_key or os.environ.get("CFBD_API_KEY")
        if not key:
            raise RuntimeError("CFBD API key required for opponent adjustments.")
        cfbd_games = fetch_games(year, key)
        adjustments = compute_power_adjustments(
            ratings,
            cfbd_games,
            constants=constants,
            up_to_week=adjust_week,
        )
        if not games_for_anchor:
            games_for_anchor = cfbd_games

    anchor_config = market_anchor_config
    if anchor_config is None:
        default_archive = ODDSLOGIC_ARCHIVE_DIR
        if default_archive.exists():
            anchor_config = market_anchor.MarketAnchorConfig(
                archive_path=default_archive,
                classification="fbs",
            )
    anchor_adjustments: Dict[str, float] = {}
    if anchor_config and anchor_config.archive_path.exists():
        anchor_source = games_for_anchor or cfbd_games
        if anchor_source:
            anchor_book = RatingBook(ratings, constants, power_adjustments=adjustments)
            anchor_adjustments = compute_market_anchor_adjustments(
                anchor_book,
                anchor_source,
                config=anchor_config,
                up_to_week=adjust_week,
            )

    combined_adjustments = adjustments.copy()
    for team, value in anchor_adjustments.items():
        combined_adjustments[team] = combined_adjustments.get(team, 0.0) + value

    if calibration_list:
        constants = refit_scoring_constants(ratings, calibration_list, combined_adjustments, constants)
    book = RatingBook(ratings, constants, power_adjustments=combined_adjustments)
    if calibration_list:
        spread_cal, total_cal = fit_linear_calibrations(book, calibration_list)
        book.spread_calibration = spread_cal
        book.total_calibration = total_cal
        sigma = fit_probability_sigma(book, calibration_list)
        if sigma:
            book.prob_sigma = sigma
    return ratings, book


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project FBS matchups using CFBD metrics.")
    parser.add_argument("home_team", nargs="?", help="Home team (if omitted, list ratings).")
    parser.add_argument("away_team", nargs="?", help="Away team.")
    parser.add_argument("--year", dest="year", type=int, default=2024, help="Season year (default 2024).")
    parser.add_argument("--api-key", dest="api_key", help="CFBD API key (optional; otherwise env used).")
    parser.add_argument("--neutral", dest="neutral", action="store_true", help="Treat matchup as neutral site.")
    parser.add_argument("--list", dest="list_only", action="store_true", help="List team ratings and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratings, book = build_rating_book(args.year, api_key=args.api_key, adjust_week=None)

    if args.list_only or not (args.home_team and args.away_team):
        cols = ["team", "offense_rating", "defense_rating", "power_rating", "sp_rating", "fpi", "elo"]
        display = ratings[cols].sort_values("power_rating", ascending=False)
        pd.set_option("display.float_format", lambda v: f"{v:0.2f}")
        print(display.to_string(index=False))
        return

    result = book.predict(args.home_team, args.away_team, neutral_site=args.neutral)
    print(f"Matchup: {result['home_team']} vs {result['away_team']}")
    print(f"Spread (home - away): {result['spread_home_minus_away']:.2f}")
    print(f"Total: {result['total_points']:.2f}")
    print(f"Projected score: {result['home_team']} {result['home_points']:.1f} | {result['away_team']} {result['away_points']:.1f}")
    print(f"Home win probability: {result['home_win_prob']*100:0.1f}%")
    print(f"Away win probability: {result['away_win_prob']*100:0.1f}%")
    if result['home_moneyline'] is not None:
        print(
            f"Moneyline: {result['home_team']} {result['home_moneyline']:+.0f} | {result['away_team']} {result['away_moneyline']:+.0f}"
        )
    else:
        print("Moneyline: probabilities at bounds; cannot compute.")


if __name__ == "__main__":
    main()
