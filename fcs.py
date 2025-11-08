"""FCS matchup projection tool built from PFF unit grades.

This script aggregates the player-level PFF exports located in
``~/Desktop/PFFMODEL_FBS/FCS_DATA`` (configurable) to produce team-level ratings and simple
spread/total/moneyline projections for any two FCS programs.

The modelling approach is heuristic: it builds weighted averages of the
available unit grades (receiving, blocking, defense, special teams),
normalises them via z-scores, and then maps those ratings into expected
points using tunable constants. Without play-by-play or historical game
results this is best used for directional comparisons rather than
hard-number betting edges.
"""
from __future__ import annotations

import argparse
import difflib
import logging
import math
import re
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import json
import pandas as pd
import numpy as np
import requests

import ncaa_stats
from cfb import market as market_utils
from cfb.config import load_config
from cfb.fcs_aliases import normalize_label as alias_normalize, map_team as alias_map
from cfb.injuries import penalties_for_player
from cfb.io import oddslogic as oddslogic_io
from cfb.io import the_odds_api
import oddslogic_loader
from cfb.model import BayesianConfig, FCSRatingBook, FCSRatingConstants

ADJUSTED_DATA_GLOB = "fcs_adjusted*/fcs_adjusted_{kind}_*.csv"
CFBD_BASE_URL = "https://api.collegefootballdata.com"

CONFIG = load_config()
FCS_CONFIG = CONFIG.get("fcs", {}) if isinstance(CONFIG.get("fcs"), dict) else {}
_MARKET_CONFIG = FCS_CONFIG.get("market", {}) if isinstance(FCS_CONFIG.get("market"), dict) else {}
_DATA_CONFIG = FCS_CONFIG.get("data", {}) if isinstance(FCS_CONFIG.get("data"), dict) else {}

logger = logging.getLogger(__name__)

def _resolve_path(value: Optional[str], default: Optional[Path]) -> Optional[Path]:
    """Coerce configuration path values into ``Path`` objects."""
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "null":
        return default
    return Path(text).expanduser()


DATA_DIR_DEFAULT = _resolve_path(_DATA_CONFIG.get("pff_dir"), Path("data/pff/fcs"))
PFF_COMBINED_DATA_DIR = _resolve_path(_DATA_CONFIG.get("pff_combined_dir"), None)

_RATING_CONFIG = FCS_CONFIG.get("ratings", {}) if isinstance(FCS_CONFIG.get("ratings"), dict) else {}
_BAYESIAN_CONFIG = FCS_CONFIG.get("bayesian", {}) if isinstance(FCS_CONFIG.get("bayesian"), dict) else {}

FCS_MARKET_SPREAD_WEIGHT = float(_MARKET_CONFIG.get("spread_weight", 0.7))
FCS_MARKET_TOTAL_WEIGHT = float(_MARKET_CONFIG.get("total_weight", 0.7))
_providers_config = _MARKET_CONFIG.get("providers")
if isinstance(_providers_config, (list, tuple)):
    FCS_MARKET_PROVIDERS = [str(name).strip() for name in _providers_config if str(name).strip()]
else:
    FCS_MARKET_PROVIDERS = []
FCS_TEAM_SET = set(ncaa_stats.SLUG_TO_PFF.values())

_THE_ODDS_API_CONFIG = _MARKET_CONFIG.get("the_odds_api", {}) if isinstance(_MARKET_CONFIG.get("the_odds_api"), dict) else {}
_THE_ODDS_API_ENABLED = bool(_THE_ODDS_API_CONFIG.get("enabled", True))
THE_ODDS_API_SPORT_KEY = str(_THE_ODDS_API_CONFIG.get("sport_key", "americanfootball_ncaaf")).strip()
if not THE_ODDS_API_SPORT_KEY:
    _THE_ODDS_API_ENABLED = False


def _coerce_sequence(value: object, fallback: Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return tuple(items) if items else tuple(fallback)
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return tuple(fallback)


THE_ODDS_API_REGIONS = _coerce_sequence(_THE_ODDS_API_CONFIG.get("regions"), ("us", "us2"))
THE_ODDS_API_MARKETS = _coerce_sequence(_THE_ODDS_API_CONFIG.get("markets"), ("spreads", "totals", "h2h"))
_BOOKMAKERS_RAW = _coerce_sequence(_THE_ODDS_API_CONFIG.get("bookmakers"), ())
THE_ODDS_API_BOOKMAKERS = _BOOKMAKERS_RAW if _BOOKMAKERS_RAW else None
THE_ODDS_API_FORMAT = str(_THE_ODDS_API_CONFIG.get("odds_format", "american")).strip() or "american"
THE_ODDS_API_DAYS_FROM_RAW = _THE_ODDS_API_CONFIG.get("days_from")
THE_ODDS_API_DAYS_FROM = None
if THE_ODDS_API_DAYS_FROM_RAW is not None:
    try:
        THE_ODDS_API_DAYS_FROM = max(0, int(THE_ODDS_API_DAYS_FROM_RAW))
    except (TypeError, ValueError):
        THE_ODDS_API_DAYS_FROM = None
if THE_ODDS_API_DAYS_FROM is None:
    THE_ODDS_API_DAYS_FROM = 7


def _normalize_provider_name(name: Optional[str]) -> str:
    return str(name or "").strip().lower()


_BOOKMAKER_ALIASES = {
    "betmgm": "BetMGM",
    "betrivers": "BetRivers",
    "betus": "BetUS",
    "bovada": "Bovada",
    "caesars": "Caesars",
    "circa": "Circa",
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "pointsbetus": "PointsBet US",
    "superbook": "SuperBook",
    "williamhill_us": "William Hill US",
    "wynnbet": "WynnBET",
}


def _format_bookmaker_name(key: Optional[str]) -> str:
    if not key:
        return "Unknown"
    normalized = str(key).strip()
    if not normalized:
        return "Unknown"
    alias = _BOOKMAKER_ALIASES.get(normalized.lower())
    if alias:
        return alias
    parts = normalized.replace("_", " ").replace("-", " ").split()
    return " ".join(part.capitalize() for part in parts)


def _datetime_to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_match_label(label: Optional[str]) -> str:
    if not label:
        return ""
    normalized = alias_normalize(label)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _tokenize_match_label(label: Optional[str]) -> tuple[str, ...]:
    if not label:
        return ()
    tokens = set()
    normalized = _clean_match_label(label)
    tokens.update(token for token in normalized.split(" ") if token)
    alias = alias_map(label)
    if alias:
        alias_normalized = _clean_match_label(alias)
        tokens.update(token for token in alias_normalized.split(" ") if token)
    return tuple(sorted(tokens))


def _tokens_match(candidate: tuple[str, ...], target: tuple[str, ...]) -> bool:
    if not candidate or not target:
        return False
    candidate_set = set(candidate)
    target_set = set(target)
    if candidate_set <= target_set or target_set <= candidate_set:
        return True
    candidate_str = " ".join(candidate)
    target_str = " ".join(target)
    if candidate_str and target_str:
        if candidate_str in target_str or target_str in candidate_str:
            return True
        ratio = difflib.SequenceMatcher(None, candidate_str, target_str).ratio()
        if ratio >= 0.82:
            return True
    return False


def _resolve_odds_api_match(
    cfbd_entries: Iterable[dict],
    home_tokens: tuple[str, ...],
    away_tokens: tuple[str, ...],
    commence_time: Optional[datetime],
) -> Optional[tuple[dict, bool]]:
    if not cfbd_entries:
        return None
    candidates: list[tuple[float, dict, bool]] = []
    for meta in cfbd_entries:
        record = meta.get("record")
        if not record:
            continue
        cfbd_home = meta.get("home_tokens") or ()
        cfbd_away = meta.get("away_tokens") or ()
        kickoff = meta.get("kickoff")
        if _tokens_match(cfbd_home, home_tokens) and _tokens_match(cfbd_away, away_tokens):
            delta = float("inf")
            if kickoff and commence_time:
                delta = abs((kickoff - commence_time).total_seconds())
            candidates.append((delta, meta, False))
        elif _tokens_match(cfbd_home, away_tokens) and _tokens_match(cfbd_away, home_tokens):
            delta = float("inf")
            if kickoff and commence_time:
                delta = abs((kickoff - commence_time).total_seconds())
            candidates.append((delta, meta, True))
    if not candidates:
        return None
    candidates.sort()
    _, meta, invert = candidates[0]
    return meta, invert


def _summarize_provider_lines(record: dict) -> None:
    provider_lines = record.get("provider_lines") or {}
    if not provider_lines:
        record["providers"] = []
        record["primary_provider"] = None
        return
    provider_names = sorted(provider_lines.keys())
    record["providers"] = provider_names
    primary_name = provider_names[0]
    record["primary_provider"] = primary_name
    spread_choice: Optional[float] = None
    total_choice: Optional[float] = None
    if primary_name in provider_lines:
        info = provider_lines[primary_name]
        spread_home = info.get("spread_home")
        if spread_home is not None:
            spread_choice = -float(spread_home)
        total_val = info.get("total")
        if total_val is not None:
            total_choice = float(total_val)
    if spread_choice is None:
        spreads = [
            -float(info.get("spread_home"))
            for info in provider_lines.values()
            if info.get("spread_home") is not None
        ]
        if spreads:
            spread_choice = float(np.mean(spreads))
    if total_choice is None:
        totals = [
            float(info.get("total"))
            for info in provider_lines.values()
            if info.get("total") is not None
        ]
        if totals:
            total_choice = float(np.mean(totals))
    if spread_choice is not None:
        record["spread"] = spread_choice
    if total_choice is not None:
        record["total"] = total_choice


def _merge_the_odds_api_events(
    cfbd_entries: list[dict],
    events: Iterable[dict],
    *,
    provider_filter: Optional[set[str]] = None,
) -> set[int]:
    matched: set[int] = set()
    if not cfbd_entries or not events:
        return matched
    for event in events:
        rows = the_odds_api.normalise_prices(event)
        if not rows:
            continue
        home_tokens = _tokenize_match_label(event.get("home_team"))
        away_tokens = _tokenize_match_label(event.get("away_team"))
        commence = rows[0].get("commence_time")
        match = _resolve_odds_api_match(cfbd_entries, home_tokens, away_tokens, commence)
        if not match:
            continue
        meta, invert = match
        record = meta.get("record")
        if not record:
            continue
        game_id = record.get("game_id")
        if game_id is not None:
            matched.add(int(game_id))
        provider_lines = record.setdefault("provider_lines", {})
        provider_weights = record.setdefault("provider_weights", {})
        for row in rows:
            provider_key = _format_bookmaker_name(row.get("bookmaker"))
            existing = provider_lines.get(provider_key, {}).copy()
            existing["source"] = "TheOddsAPI"
            last_update = _datetime_to_iso(row.get("last_update"))
            if last_update:
                current = existing.get("last_updated")
                if not current or last_update > current:
                    existing["last_updated"] = last_update
            market = row.get("market")
            if market == "spread":
                point = row.get("point")
                if point is not None:
                    if invert:
                        point = -point
                    existing["spread_home"] = point
                    existing["spread_away"] = -point
                price_home = row.get("price_home")
                price_away = row.get("price_away")
                if invert:
                    price_home, price_away = price_away, price_home
                if price_home is not None:
                    existing["spread_price_home"] = price_home
                if price_away is not None:
                    existing["spread_price_away"] = price_away
            elif market == "total":
                point = row.get("point")
                if point is not None:
                    existing["total"] = point
                price_over = row.get("price_over")
                price_under = row.get("price_under")
                if price_over is not None:
                    existing["total_price_over"] = price_over
                if price_under is not None:
                    existing["total_price_under"] = price_under
            elif market == "moneyline":
                price_home = row.get("price_home")
                price_away = row.get("price_away")
                if invert:
                    price_home, price_away = price_away, price_home
                if price_home is not None:
                    existing["team_one_moneyline"] = price_home
                if price_away is not None:
                    existing["team_two_moneyline"] = price_away
            provider_lines[provider_key] = existing
            if provider_key not in provider_weights:
                provider_weights[provider_key] = 1.0
        _summarize_provider_lines(record)
    return matched


def _log_missing_odds(
    records: list[dict],
    matched: Optional[set[int]],
    *,
    label: str = "FCS",
) -> None:
    if matched is None:
        return
    missing: list[dict] = []
    for entry in records:
        record = entry.get("record") if isinstance(entry, dict) else None
        if not isinstance(record, dict):
            continue
        game_id = record.get("game_id")
        if game_id is not None and matched and int(game_id) in matched:
            continue
        missing.append(record)
    if not missing:
        return
    preview = []
    for record in missing[:3]:
        home = record.get("home_team") or "Unknown home"
        away = record.get("away_team") or "Unknown away"
        preview.append(f"{home} vs {away}")
    sample = "; ".join(preview)
    logger.warning(
        "The Odds API returned no %s odds for %d games (sample: %s); falling back to secondary feeds.",
        label,
        len(missing),
        sample,
    )


def _rating_constants_from_config() -> FCSRatingConstants:
    return FCSRatingConstants(
        avg_total=float(_RATING_CONFIG.get("avg_total", 52.7442)),
        home_field_advantage=float(_RATING_CONFIG.get("home_field_advantage", 4.9105)),
        offense_factor=float(_RATING_CONFIG.get("offense_factor", 0.7333)),
        defense_factor=float(_RATING_CONFIG.get("defense_factor", 0.2318)),
        special_teams_factor=float(_RATING_CONFIG.get("special_teams_factor", 1.4226)),
        spread_sigma=float(_RATING_CONFIG.get("spread_sigma", 16.0)),
    )


def _bayesian_config_from_config() -> Optional[BayesianConfig]:
    if not _BAYESIAN_CONFIG or not bool(_BAYESIAN_CONFIG.get("enabled", False)):
        return None
    total_prior_mean_raw = _BAYESIAN_CONFIG.get("total_prior_mean", None)
    total_prior_mean = (
        float(total_prior_mean_raw)
        if total_prior_mean_raw is not None
        else None
    )
    return BayesianConfig(
        enabled=True,
        spread_prior_strength=float(_BAYESIAN_CONFIG.get("spread_prior_strength", 0.0)),
        total_prior_strength=float(_BAYESIAN_CONFIG.get("total_prior_strength", 0.0)),
        spread_prior_mean=float(_BAYESIAN_CONFIG.get("spread_prior_mean", 0.0)),
        total_prior_mean=total_prior_mean,
        min_games=float(_BAYESIAN_CONFIG.get("min_games", 0.0)),
        max_games=float(_BAYESIAN_CONFIG.get("max_games", 20.0)),
        default_games=float(_BAYESIAN_CONFIG.get("default_games", 6.0)),
        prob_sigma_scale=float(_BAYESIAN_CONFIG.get("prob_sigma_scale", 0.0)),
    )

LEGACY_SPREAD_CALIBRATION = (0.01194, 0.99456)
FCS_TOTAL_CALIBRATION = (0.0, 1.0)
FCS_PROB_SIGMA = 16.0
LEGACY_SPREAD_REG_INTERCEPT = 5.013813339075651
FCS_SPREAD_REG_WEIGHTS = {
    'receiving_grade_offense_z': -0.359949177,
    'receiving_grade_route_z': -0.772627951,
    'receiving_grade_pass_block_z': 0.992779226,
    'receiving_target_qb_rating_z': -1.971349339,
    'receiving_yprr_z': 0.786078117,
    'blocking_grade_offense_z': 5.992998767,
    'blocking_grade_pass_z': -2.918807137,
    'blocking_grade_run_z': -2.526163359,
    'blocking_pbe_z': -0.489362404,
    'defense_grade_overall_z': 1.820714587,
    'defense_grade_coverage_z': -1.297574619,
    'defense_grade_run_z': -0.12659368,
    'defense_grade_pass_rush_z': 2.267166957,
    'defense_missed_tackle_rate_z': -0.086953135,
    'defense_qb_rating_against_z': 1.854237316,
    'special_grade_misc_z': 0.910956481,
    'special_grade_return_z': -0.824076876,
    'special_grade_punt_return_z': -0.412718338,
    'special_grade_kickoff_z': 1.028323866,
    'special_grade_fg_offense_z': -0.648123833,
    'plays_per_game_z': -8.662630872,
    'offense_ypp_z': -20.242904659,
    'offense_ypg_z': 11.482006909,
    'third_down_pct_z': -0.654515071,
    'team_pass_eff_z': -1.453155962,
    'points_per_game_z': 2.228742158,
    'avg_time_of_possession_z': 0.722003921,
    'defense_ypp_allowed_z': -2.694353376,
    'defense_ypg_allowed_z': -1.34643046,
    'third_down_def_pct_z': -1.377282778,
    'opp_pass_eff_z': -1.872320426,
    'points_allowed_per_game_z': 2.474259229,
    'qb_pass_eff_z': 2.834557872,
    'rb_rush_ypg_z': 0.583739072,
    'wr_rec_yards_z': -0.484271947,
    'offense_ypp_adj_z': 0.0,
    'points_per_game_adj_z': 0.0,
    'team_pass_eff_adj_z': 0.0,
    'defense_ypp_allowed_adj_z': 0.0,
    'points_allowed_per_game_adj_z': 0.0,
    'red_zone_pct_z': -0.512890797,
    'red_zone_def_pct_z': -0.620689965,
    'red_zone_attempts_z': -1.679994143,
    'red_zone_def_attempts_z': 0.602810343,
    'turnover_gain_z': -4.363649358,
    'turnover_loss_z': 2.146492429,
    'turnover_margin_total_z': -4.254275126,
    'turnover_margin_avg_z': 8.487381147,
    'penalties_per_game_z': -2.045464015,
    'penalty_yards_per_game_z': 1.691846468,
    'rush_yards_per_game_z': 8.683764489,
    'rush_yards_allowed_per_game_z': 2.423571653,
    'pass_yards_per_game_z': 9.549474757,
    'pass_yards_allowed_per_game_z': 1.97290554,
    'sacks_per_game_z': -2.1758381,
    'tfl_per_game_z': -1.348546304,
    'sacks_allowed_per_game_z': 0.928272521,
    'tfl_allowed_per_game_z': -0.254408342,
    'kick_return_avg_z': 1.392441025,
    'kick_return_defense_avg_z': -1.236548609,
    'punt_return_avg_z': 0.253832037,
    'punt_return_defense_avg_z': -0.502303225,
    'net_punting_avg_z': -0.070792218,
}
LEGACY_SPREAD_LINEAR_COEFFS = {
    "intercept": 2.60184,
    "spread_rating_diff": 0.55249,
    "power_diff": 0.72979,
    "offense_diff": 1.17088,
    "defense_diff": 0.30222,
    "special_diff": -0.69429,
    "sr_x_power": 0.000103,
    "off_x_def": 0.24222,
    "power_x_special": -0.09227,
    "power_sq": 0.11817,
}

_SPREAD_MODEL_PATH_RAW = _RATING_CONFIG.get("spread_model_path", "calibration/fcs_spread_model.json")
SPREAD_MODEL_PATH = Path(str(_SPREAD_MODEL_PATH_RAW)).expanduser()


def _coerce_float(value: Optional[float]) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    filtered = "".join(ch for ch in str(value) if ch.isdigit() or ch in {".", "-", "+"})
    if not filtered:
        return None
    try:
        return float(filtered)
    except ValueError:
        return None


def _load_spread_model_defaults() -> tuple[dict[str, float], tuple[float, float]]:
    try:
        with SPREAD_MODEL_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return dict(LEGACY_SPREAD_LINEAR_COEFFS), LEGACY_SPREAD_CALIBRATION
    coefficients = payload.get("coefficients")
    if isinstance(coefficients, dict) and coefficients:
        coeffs = {str(key): float(value) for key, value in coefficients.items()}
    else:
        coeffs = dict(LEGACY_SPREAD_LINEAR_COEFFS)
    calibration = payload.get("calibration")
    if isinstance(calibration, dict):
        intercept_val = _coerce_float(calibration.get("intercept"))
        slope_val = _coerce_float(calibration.get("slope"))
        intercept = float(intercept_val) if intercept_val is not None else LEGACY_SPREAD_CALIBRATION[0]
        slope = float(slope_val) if slope_val is not None else LEGACY_SPREAD_CALIBRATION[1]
        calibration_tuple = (intercept, slope)
    else:
        calibration_tuple = LEGACY_SPREAD_CALIBRATION
    return coeffs, calibration_tuple


FCS_SPREAD_LINEAR_COEFFS, FCS_SPREAD_CALIBRATION = _load_spread_model_defaults()
FCS_SPREAD_REG_INTERCEPT = FCS_SPREAD_LINEAR_COEFFS.get("intercept", LEGACY_SPREAD_REG_INTERCEPT)


def _read_first_available(data_dir: Optional[Path], candidates: Iterable[str]) -> pd.DataFrame:
    directories: list[Path] = []
    if PFF_COMBINED_DATA_DIR:
        directories.append(PFF_COMBINED_DATA_DIR.expanduser())
    if data_dir:
        directories.append(data_dir.expanduser())
    if not directories:
        directories.append(Path("."))
    for directory in directories:
        for name in candidates:
            if any(ch in name for ch in "*?[]"):
                paths = sorted(
                    directory.glob(name),
                    key=lambda p: p.stat().st_mtime if p.exists() else 0,
                    reverse=True,
                )
            else:
                paths = [(directory / name)]
            for path in paths:
                if path.exists():
                    return pd.read_csv(path)
    joined = ", ".join(candidates)
    raise FileNotFoundError(f"None of the files were found in {data_dir} or supplemental sources: {joined}")


def _load_adjusted_metrics(season_year: Optional[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ridge-regression opponent-adjusted offense/defense metrics if available."""

    if not season_year:
        return pd.DataFrame(), pd.DataFrame()

    def _first_matching(kind: str) -> Optional[Path]:
        patterns = [
            Path(".") / f"fcs_adjusted_{season_year}/fcs_adjusted_{kind}_{season_year}.csv",
            Path(".") / f"fcs_adjusted_{season_year}_wk1_9/fcs_adjusted_{kind}_{season_year}.csv",
        ]
        matches = [p for p in patterns if p.exists()]
        if not matches:
            glob_pattern = ADJUSTED_DATA_GLOB.format(kind=kind)
            matches = sorted(
                Path(".").glob(glob_pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            matches = [p for p in matches if f"_{season_year}" in p.name]
        return matches[0] if matches else None

    offense_path = _first_matching("offense")
    defense_path = _first_matching("defense")
    offense_df = pd.read_csv(offense_path) if offense_path else pd.DataFrame()
    defense_df = pd.read_csv(defense_path) if defense_path else pd.DataFrame()
    return offense_df, defense_df


def fetch_injury_penalties(league: str = "ncaaf_fcs") -> pd.DataFrame:
    """Fetch aggregated injury penalties from OddsLogic for the requested league."""

    try:
        payload = oddslogic_io.fetch_injuries(league=league)
    except requests.RequestException:
        return pd.DataFrame(columns=["team_name", "injury_offense_penalty", "injury_defense_penalty"])

    entries: list[dict[str, float | str]] = []
    for info in (payload or {}).values():
        team = info.get("player_team")
        if not team:
            continue
        status = info.get("injury_status") or ""
        position = info.get("player_position") or ""
        custom_text = info.get("custom_text") or ""
        offense_pen, defense_pen = penalties_for_player(status, position, custom_text=custom_text)
        if offense_pen == 0.0 and defense_pen == 0.0:
            continue
        entries.append(
            {
                "team_name": str(team),
                "injury_offense_penalty": float(offense_pen),
                "injury_defense_penalty": float(defense_pen),
            }
        )

    if not entries:
        return pd.DataFrame(columns=["team_name", "injury_offense_penalty", "injury_defense_penalty"])

    df = pd.DataFrame(entries)
    return (
        df.groupby("team_name", as_index=False)[["injury_offense_penalty", "injury_defense_penalty"]]
        .sum()
    )


def apply_injury_penalties(teams: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Apply injury penalties to the aggregated team ratings."""

    if injuries.empty:
        return teams
    if "team_name" not in teams.columns:
        return teams

    teams = teams.copy()
    lookup = {alias_normalize(name): name for name in teams["team_name"]}
    team_names = teams["team_name"].tolist()
    for entry in injuries.itertuples():
        normalized = alias_normalize(entry.team_name)
        resolved = lookup.get(normalized)
        if not resolved:
            alias = alias_map(entry.team_name)
            if alias and alias in team_names:
                resolved = alias
        if not resolved:
            matches = difflib.get_close_matches(normalized, lookup.keys(), n=1, cutoff=0.92)
            if matches:
                resolved = lookup[matches[0]]
        if not resolved:
            continue
        mask = teams["team_name"] == resolved
        if "offense_rating" in teams.columns:
            teams.loc[mask, "offense_rating"] = teams.loc[mask, "offense_rating"].astype(float) - float(entry.injury_offense_penalty)
        if "defense_rating" in teams.columns:
            teams.loc[mask, "defense_rating"] = teams.loc[mask, "defense_rating"].astype(float) + float(entry.injury_defense_penalty)
    return teams


def fetch_market_lines(
    year: int,
    api_key: str,
    *,
    week: Optional[int] = None,
    season_type: str = "regular",
    classification: str = "fcs",
    provider: Optional[str] = None,
    providers: Optional[Iterable[str]] = None,
) -> list[dict]:
    headers = {"Authorization": f"Bearer {api_key}"}
    params: Dict[str, object] = {"year": year}
    if season_type:
        params["seasonType"] = season_type
    if week is not None:
        params["week"] = week
    if classification:
        params["classification"] = classification
    provider_names_input: list[str] = [name for name in FCS_MARKET_PROVIDERS]
    if providers:
        provider_names_input.extend(str(name).strip() for name in providers if str(name).strip())
    if provider:
        provider_names_input.append(str(provider).strip())
    seen_providers: set[str] = set()
    normalized_inputs: list[str] = []
    for name in provider_names_input:
        cleaned = name.strip()
        if not cleaned:
            continue
        lower = cleaned.lower()
        if lower in seen_providers:
            continue
        seen_providers.add(lower)
        normalized_inputs.append(cleaned)
    provider_names_input = normalized_inputs
    provider_filter = None

    resp = requests.get(
        CFBD_BASE_URL + "/lines",
        headers=headers,
        params=params,
        timeout=60,
    )
    if resp.status_code == 401:
        raise RuntimeError("CFBD API rejected the key when fetching FCS lines (401).")
    resp.raise_for_status()

    records: list[dict] = []
    cfbd_entries: list[dict] = []
    record_index: Dict[Tuple[dt.date, str, str], dict] = {}
    kickoff_dates: set[dt.date] = set()
    for entry in resp.json():
        if entry.get("homeClassification") != "fcs" or entry.get("awayClassification") != "fcs":
            continue
        start_raw = entry.get("startDate")
        kickoff_dt = None
        kickoff_py = None
        kickoff_date = None
        if start_raw:
            try:
                kickoff_dt = pd.to_datetime(start_raw, utc=True)
            except (TypeError, ValueError):
                kickoff_dt = pd.NaT
            if isinstance(kickoff_dt, pd.Timestamp) and not pd.isna(kickoff_dt):
                kickoff_py = kickoff_dt.to_pydatetime()
                kickoff_date = kickoff_dt.date()
                kickoff_dates.add(kickoff_date)
        lines = entry.get("lines") or []
        provider_names: set[str] = set()
        provider_lines: Dict[str, dict] = {}
        for line in lines:
            prov_raw = line.get("provider")
            provider_name = str(prov_raw or "").strip()
            if not provider_name:
                continue
            normalized_provider = _normalize_provider_name(provider_name)

            spread_raw = _coerce_float(line.get("spread"))
            total_raw = _coerce_float(line.get("overUnder"))
            home_ml = _coerce_float(line.get("homeMoneyline"))
            away_ml = _coerce_float(line.get("awayMoneyline"))
            last_updated = line.get("lastUpdated") or line.get("updated")

            provider_names.add(provider_name)
            if spread_raw is not None:
                spread_home = float(spread_raw)
            else:
                spread_home = None
            provider_lines[provider_name] = {
                "spread_home": spread_home,
                "spread_away": -spread_home if spread_home is not None else None,
                "total": total_raw,
                "home_moneyline": home_ml,
                "away_moneyline": away_ml,
                "last_updated": last_updated,
            }

        spread_values = [
            -float(info.get("spread_home"))
            for info in provider_lines.values()
            if info.get("spread_home") is not None
        ]
        total_values = [
            info.get("total") for info in provider_lines.values() if info.get("total") is not None
        ]
        provider_names_sorted = sorted(provider_names)
        primary_provider = provider_names_sorted[0] if provider_names_sorted else None
        spread_value = None
        total_value = None
        if primary_provider and primary_provider in provider_lines:
            primary_info = provider_lines[primary_provider]
            primary_spread = primary_info.get("spread_home")
            if primary_spread is not None:
                spread_value = -float(primary_spread)
            primary_total = primary_info.get("total")
            if primary_total is not None:
                total_value = float(primary_total)
        if spread_value is None and spread_values:
            spread_value = float(sum(spread_values) / len(spread_values))
        if total_value is None and total_values:
            total_value = float(sum(total_values) / len(total_values))

        record = {
            "game_id": entry.get("id"),
            "home_team": entry.get("homeTeam"),
            "away_team": entry.get("awayTeam"),
            "start_date": entry.get("startDate"),
            "spread": spread_value,
            "total": total_value,
            "providers": provider_names_sorted,
            "provider_lines": provider_lines,
            "primary_provider": primary_provider,
            "provider_weights": {name: 1.0 for name in provider_names_sorted},
        }
        records.append(record)
        if kickoff_date:
            key = (
                kickoff_date,
                oddslogic_loader.normalize_label(entry.get("homeTeam") or ""),
                oddslogic_loader.normalize_label(entry.get("awayTeam") or ""),
            )
            record_index[key] = record
        cfbd_entries.append(
            {
                "record": record,
                "home_tokens": _tokenize_match_label(entry.get("homeTeam")),
                "away_tokens": _tokenize_match_label(entry.get("awayTeam")),
                "kickoff": kickoff_py,
            }
        )

    schedule_index: Dict[Tuple[dt.date, str, str], dict] = {}
    if week is not None:
        schedule_params = {
            "year": year,
            "week": week,
            "seasonType": season_type,
            "division": "fcs",
        }
        try:
            games_resp = requests.get(
                CFBD_BASE_URL + "/games",
                headers=headers,
                params=schedule_params,
                timeout=60,
            )
            games_resp.raise_for_status()
            for game in games_resp.json():
                start_raw = game.get("startDate")
                if not start_raw:
                    continue
                try:
                    kickoff_dt = pd.to_datetime(start_raw, utc=True)
                except (TypeError, ValueError):
                    kickoff_dt = pd.NaT
                if pd.isna(kickoff_dt):
                    continue
                kickoff_date = kickoff_dt.date()
                kickoff_dates.add(kickoff_date)
                key = (
                    kickoff_date,
                    oddslogic_loader.normalize_label(game.get("homeTeam") or ""),
                    oddslogic_loader.normalize_label(game.get("awayTeam") or ""),
                )
                schedule_index[key] = game
        except requests.RequestException:
            pass

    odds_api_provider_filter = None
    matched_odds_api: Optional[set[int]] = None
    if _THE_ODDS_API_ENABLED and THE_ODDS_API_SPORT_KEY:
        try:
            odds_events = the_odds_api.fetch_current_odds(
                THE_ODDS_API_SPORT_KEY,
                regions=THE_ODDS_API_REGIONS,
                markets=THE_ODDS_API_MARKETS,
                bookmakers=THE_ODDS_API_BOOKMAKERS,
                odds_format=THE_ODDS_API_FORMAT,
                days_from=THE_ODDS_API_DAYS_FROM,
            )
        except the_odds_api.TheOddsAPIError as exc:
            logger.warning("The Odds API fetch failed for FCS: %s", exc)
            odds_events = []
        except Exception as exc:  # pragma: no cover
            logger.warning("Unexpected The Odds API error for FCS: %s", exc)
            odds_events = []
        else:
            matched_odds_api = _merge_the_odds_api_events(
                cfbd_entries,
                odds_events,
                provider_filter=odds_api_provider_filter,
            )
    _log_missing_odds(cfbd_entries, matched_odds_api, label="FCS")

    live_lookup: Dict[Tuple[dt.date, str, str], Dict[str, object]] = {}
    if kickoff_dates:
        live_lookup = oddslogic_io.fetch_live_market_lookup(
            kickoff_dates,
            classification="fcs",
            providers=provider_names_input or None,
        )

    def _merge_provider_lines(existing: dict, payload: dict, *, invert: bool = False) -> dict:
        spread_value = payload.get("spread_value")
        if spread_value is not None and invert:
            spread_value = -spread_value
        total_value = payload.get("total_value")
        provider_entry = {
            "spread_home": spread_value,
            "spread_away": -spread_value if spread_value is not None else None,
            "total": total_value,
            "home_moneyline": existing.get("home_moneyline"),
            "away_moneyline": existing.get("away_moneyline"),
            "last_updated": payload.get("spread_updated") or payload.get("total_updated"),
            "source": "OddsLogic",
            "open_spread_home": (
                -payload["open_spread_value"]
                if invert and payload.get("open_spread_value") is not None
                else payload.get("open_spread_value")
            ),
            "open_total": payload.get("open_total_value"),
        }
        return provider_entry

    for key, live_entry in live_lookup.items():
        kickoff_date, home_key, away_key = key
        record = record_index.get(key)
        invert = False
        if record is None:
            alt_key = (kickoff_date, away_key, home_key)
            record = record_index.get(alt_key)
            if record:
                invert = True
        if record is None:
            schedule_game = schedule_index.get(key)
            if not schedule_game:
                schedule_game = schedule_index.get((kickoff_date, away_key, home_key))
                if schedule_game:
                    invert = True
            if schedule_game:
                start_raw = schedule_game.get("startDate")
                try:
                    kickoff_dt = pd.to_datetime(start_raw, utc=True)
                except (TypeError, ValueError):
                    kickoff_dt = pd.NaT
                start_iso = kickoff_dt.isoformat() if not pd.isna(kickoff_dt) else f"{kickoff_date.isoformat()}T00:00:00"
                record = {
                    "game_id": schedule_game.get("id"),
                    "home_team": schedule_game.get("homeTeam"),
                    "away_team": schedule_game.get("awayTeam"),
                    "start_date": start_iso,
                    "spread": None,
                    "total": None,
                    "providers": [],
                    "provider_lines": {},
                    "primary_provider": None,
                    "provider_weights": {},
                }
                records.append(record)
                record_index[key] = record
                cfbd_entries.append(
                    {
                        "record": record,
                        "home_tokens": _tokenize_match_label(schedule_game.get("homeTeam")),
                        "away_tokens": _tokenize_match_label(schedule_game.get("awayTeam")),
                        "kickoff": kickoff_dt.to_pydatetime() if not pd.isna(kickoff_dt) else None,
                    }
                )
            else:
                start_raw = live_entry.get("start_datetime")
                try:
                    kickoff_dt = pd.to_datetime(start_raw, utc=True) if start_raw else pd.NaT
                except (TypeError, ValueError):
                    kickoff_dt = pd.NaT
                start_iso = kickoff_dt.isoformat() if not pd.isna(kickoff_dt) else f"{kickoff_date.isoformat()}T00:00:00"
                record = {
                    "game_id": None,
                    "home_team": live_entry.get("home_team"),
                    "away_team": live_entry.get("away_team"),
                    "start_date": start_iso,
                    "spread": None,
                    "total": None,
                    "providers": [],
                    "provider_lines": {},
                    "primary_provider": None,
                    "provider_weights": {},
                }
                records.append(record)
                record_index[key] = record
                cfbd_entries.append(
                    {
                        "record": record,
                        "home_tokens": _tokenize_match_label(record.get("home_team")),
                        "away_tokens": _tokenize_match_label(record.get("away_team")),
                        "kickoff": kickoff_dt.to_pydatetime() if not pd.isna(kickoff_dt) else None,
                    }
                )

        if not record:
            continue
        provider_lines = record.setdefault("provider_lines", {})
        for provider_name, payload in live_entry["providers"].items():
            existing = provider_lines.get(provider_name, {})
            provider_lines[provider_name] = _merge_provider_lines(existing, payload, invert=invert)
        record["providers"] = sorted(provider_lines.keys())
        primary_provider = record.get("primary_provider")
        if primary_provider not in provider_lines and record["providers"]:
            primary_provider = record["providers"][0]
        record["primary_provider"] = primary_provider
        spread_choice = None
        total_choice = None
        if primary_provider and primary_provider in provider_lines:
            primary_info = provider_lines[primary_provider]
            spread_home = primary_info.get("spread_home")
            if spread_home is not None:
                spread_choice = -float(spread_home)
            total_val = primary_info.get("total")
            if total_val is not None:
                total_choice = float(total_val)
        if spread_choice is None:
            spreads = [
                -float(info.get("spread_home"))
                for info in provider_lines.values()
                if info.get("spread_home") is not None
            ]
            if spreads:
                spread_choice = float(sum(spreads) / len(spreads))
        if total_choice is None:
            totals = [info.get("total") for info in provider_lines.values() if info.get("total") is not None]
            if totals:
                total_choice = float(sum(totals) / len(totals))
        if spread_choice is not None:
            record["spread"] = spread_choice
        if total_choice is not None:
            record["total"] = total_choice
        _summarize_provider_lines(record)

    for record in records:
        _summarize_provider_lines(record)

    return records


def apply_market_prior(
    result: Dict[str, float],
    market: Optional[dict],
    *,
    spread_weight: float = FCS_MARKET_SPREAD_WEIGHT,
    total_weight: float = FCS_MARKET_TOTAL_WEIGHT,
) -> Dict[str, float]:
    return market_utils.apply_market_prior(
        result,
        market,
        prob_sigma=FCS_PROB_SIGMA,
        spread_weight=spread_weight,
        total_weight=total_weight,
        spread_key="spread_team_one_minus_team_two",
        total_key="total_points",
        home_points_key="team_one_points",
        away_points_key="team_two_points",
        win_prob_key="team_one_win_prob",
        away_win_prob_key="team_two_win_prob",
        home_moneyline_key="team_one_moneyline",
        away_moneyline_key="team_two_moneyline",
    )

# --- Aggregation helpers --------------------------------------------------

def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = values.astype(float)
    weights = weights.astype(float)
    mask = values.notna() & weights.notna()
    values = values[mask]
    weights = weights[mask]
    total_weight = weights.sum()
    if total_weight <= 0 or values.empty:
        return float("nan")
    return float((values * weights).sum() / total_weight)


def aggregate_team_metrics(
    df: pd.DataFrame,
    *,
    team_col: str,
    weight_col: Optional[str],
    weighted_metrics: Mapping[str, str],
    sum_metrics: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate player rows to team-level metrics.

    Parameters
    ----------
    df:
        Source dataframe.
    team_col:
        Column containing the team identifier.
    weight_col:
        Column containing weights for weighted averages. If ``None`` the
        script falls back to simple means.
    weighted_metrics:
        Mapping from output column name to source column that should be
        averaged (weighted by ``weight_col`` when available).
    sum_metrics:
        Mapping from output column name to source column that should be
        summed (e.g., overall team totals).
    """

    sum_metrics = sum_metrics or {}
    records: Dict[str, Dict[str, float]] = {}

    for team, group in df.groupby(team_col):
        record: Dict[str, float] = {}
        weights = None
        if weight_col and weight_col in group:
            weights = group[weight_col].fillna(0)
        for out_col, src_col in weighted_metrics.items():
            series = group[src_col]
            if weights is not None:
                value = _weighted_mean(series, weights)
                if math.isnan(value):
                    value = float(series.mean()) if not series.dropna().empty else float("nan")
            else:
                value = float(series.mean()) if not series.dropna().empty else float("nan")
            record[out_col] = value
        for out_col, src_col in sum_metrics.items():
            record[out_col] = float(group[src_col].fillna(0).sum())
        records[team] = record
    result = pd.DataFrame.from_dict(records, orient="index")
    result.index.name = "team_name"
    return result


SPECIAL_METRIC_CANDIDATES: Dict[str, tuple[str, ...]] = {
    "special_grade_misc": ("grades_misc_st", "grades_misc_special", "grades_misc"),
    "special_grade_return": ("grades_return", "grades_kick_return", "grades_return_units"),
    "special_grade_punt_return": ("grades_punt_return", "grades_puntreturn"),
    "special_grade_kickoff": ("grades_kickoff", "grades_kickoff_units"),
    "special_grade_fg_offense": ("grades_field_goal", "grades_fg_offense"),
}


def load_pff_special_metrics(data_dir: Optional[Path]) -> pd.DataFrame:
    """Aggregate PFF special-teams grades to team-level metrics."""

    if data_dir is None:
        return pd.DataFrame(columns=["team_name"])
    try:
        special = _read_first_available(
            data_dir,
            [
                "special_teams_summary*FCS*.csv",
                "special_teams_summary_FCS.csv",
                "special_teams_summary.csv",
            ],
        )
    except FileNotFoundError:
        return pd.DataFrame(columns=["team_name"])

    if "team_name" not in special.columns:
        if "team" in special.columns:
            special = special.rename(columns={"team": "team_name"})
        else:
            return pd.DataFrame(columns=["team_name"])

    metric_map: Dict[str, str] = {}
    for out_col, candidates in SPECIAL_METRIC_CANDIDATES.items():
        src = next((name for name in candidates if name in special.columns), None)
        if src:
            special[src] = pd.to_numeric(special[src], errors="coerce")
            metric_map[out_col] = src
    if not metric_map:
        return pd.DataFrame(columns=["team_name"])

    aggregated = aggregate_team_metrics(
        special,
        team_col="team_name",
        weight_col=None,
        weighted_metrics=metric_map,
    )
    return aggregated.reset_index()


# --- Rating construction --------------------------------------------------


def load_team_ratings(
    data_dir: Optional[Path] = None,
    *,
    season_year: Optional[int] = None,
) -> pd.DataFrame:
    """Load team ratings derived from opponent-adjusted PFF metrics."""

    source_dir = data_dir or DATA_DIR_DEFAULT
    if source_dir is None:
        raise RuntimeError("PFF data directory not configured; set fcs.data.pff_dir or pass --data-dir.")

    if season_year is None:
        today = datetime.utcnow().date()
        season_year = today.year if today.month >= 7 else today.year - 1

    special_metrics = load_pff_special_metrics(source_dir)

    offense_df, defense_df = _load_adjusted_metrics(season_year)
    fallback_features: Optional[pd.DataFrame] = None
    if offense_df.empty or defense_df.empty:
        warnings.warn(
            f"Adjusted metrics for season {season_year} missing; falling back to unadjusted NCAA stats.",
            RuntimeWarning,
        )
        try:
            fallback_features = ncaa_stats.build_team_feature_frame(season_year)
        except Exception:
            fallback_features = pd.DataFrame()
        if fallback_features.empty:
            raise RuntimeError(
                f"Adjusted metrics for season {season_year} not found and NCAA stats fallback failed."
            )
        offense = fallback_features[["team_name", "points_per_game"]].rename(
            columns={"points_per_game": "points_offense"}
        )
        defense = fallback_features[["team_name", "points_allowed_per_game"]].rename(
            columns={"points_allowed_per_game": "points_defense"}
        )
    else:
        offense = offense_df.rename(columns={"team": "team_name"})[["team_name", "points_offense"]]
        defense = defense_df.rename(columns={"team": "team_name"})[["team_name", "points_defense"]]
    merged = offense.merge(defense, on="team_name", how="inner")

    try:
        features = fallback_features if fallback_features is not None else ncaa_stats.build_team_feature_frame(season_year)
    except Exception:  # pragma: no cover - NCAA site hiccups shouldn't halt build
        features = pd.DataFrame()
    if not features.empty and "games_played" in features.columns:
        merged = merged.merge(features[["team_name", "games_played"]], on="team_name", how="left")
    else:
        merged["games_played"] = float("nan")

    if not special_metrics.empty:
        merged = merged.merge(special_metrics, on="team_name", how="left")
        special_cols = [
            col
            for col in [
                "special_grade_misc",
                "special_grade_return",
                "special_grade_punt_return",
                "special_grade_kickoff",
                "special_grade_fg_offense",
            ]
            if col in merged.columns
        ]
        z_cols: list[str] = []
        for col in special_cols:
            values = pd.to_numeric(merged[col], errors="coerce")
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            if std and not math.isnan(std):
                merged[f"{col}_z"] = (values - mean) / std
            else:
                merged[f"{col}_z"] = 0.0
            z_cols.append(f"{col}_z")
        if z_cols:
            merged["special_rating"] = merged[z_cols].mean(axis=1)
        else:
            merged["special_rating"] = 0.0
    else:
        merged["special_rating"] = 0.0
    merged["special_rating"] = merged["special_rating"].fillna(0.0)

    records: list[dict] = []
    offense_mean = merged["points_offense"].mean()
    defense_mean = merged["points_defense"].mean()

    for row in merged.itertuples():
        raw_name = row.team_name
        alias = alias_map(raw_name)
        normalized = alias_normalize(raw_name)
        candidate = alias or normalized
        if candidate not in FCS_TEAM_SET:
            matches = difflib.get_close_matches(candidate, FCS_TEAM_SET, n=1, cutoff=0.7)
            if matches:
                candidate = matches[0]
        if candidate not in FCS_TEAM_SET:
            continue
        offense_rating = row.points_offense - offense_mean
        defense_rating = defense_mean - row.points_defense
        power_rating = offense_rating + defense_rating
        games_played = getattr(row, "games_played", float("nan"))
        special_rating = float(getattr(row, "special_rating", 0.0))
        records.append(
            {
                "team_name": candidate,
                "offense_rating": offense_rating,
                "defense_rating": defense_rating,
                "special_rating": special_rating,
                "power_rating": power_rating,
                "spread_rating": power_rating,
                "games_played": games_played,
            }
        )

    if not records:
        raise RuntimeError("Adjusted metrics could not be mapped to FCS team names.")

    merged = pd.DataFrame(records)

    cols = [
        "team_name",
        "offense_rating",
        "defense_rating",
        "special_rating",
        "power_rating",
        "spread_rating",
        "games_played",
    ]
    return merged[cols].sort_values("power_rating", ascending=False).reset_index(drop=True)


def build_rating_book(
    data_dir: Optional[Path] = None,
    *,
    season_year: Optional[int] = None,
    spread_calibration: tuple[float, float] = FCS_SPREAD_CALIBRATION,
    total_calibration: tuple[float, float] = FCS_TOTAL_CALIBRATION,
    prob_sigma: float = FCS_PROB_SIGMA,
) -> tuple[pd.DataFrame, FCSRatingBook]:
    teams = load_team_ratings(data_dir=data_dir, season_year=season_year)
    team_games_map: Dict[str, float] = {}
    if "games_played" in teams.columns:
        for row in teams[["team_name", "games_played"]].itertuples(index=False):
            team = str(row.team_name).lower()
            try:
                games_value = float(row.games_played)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(games_value):
                continue
            team_games_map[team] = max(0.0, games_value)
    try:
        injuries = fetch_injury_penalties()
        if not injuries.empty:
            teams = apply_injury_penalties(teams, injuries)
    except Exception:  # pragma: no cover - injury feed issues shouldn't halt rating build
        pass
    spread_model = dict(FCS_SPREAD_LINEAR_COEFFS)
    if "intercept" not in spread_model:
        spread_model["intercept"] = FCS_SPREAD_REG_INTERCEPT
    rating_constants = _rating_constants_from_config()
    bayesian_cfg = _bayesian_config_from_config()
    if bayesian_cfg and bayesian_cfg.total_prior_mean is None:
        bayesian_cfg.total_prior_mean = rating_constants.avg_total
    book = FCSRatingBook(
        teams,
        rating_constants,
        spread_calibration=spread_calibration,
        total_calibration=total_calibration,
        prob_sigma=prob_sigma,
        spread_model=spread_model,
        team_games=team_games_map,
        bayesian_config=bayesian_cfg,
    )
    return teams, book


# --- CLI ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project FBS matchups using PFF data.")
    parser.add_argument("team_one", nargs="?", help="First team (defaults to listing teams if omitted).")
    parser.add_argument("team_two", nargs="?", help="Second team.")
    parser.add_argument("--data-dir", dest="data_dir", type=Path, default=DATA_DIR_DEFAULT,
                        help="Directory holding the PFF CSV exports.")
    parser.add_argument("--neutral", dest="neutral", action="store_true",
                        help="Treat the game as a neutral-site matchup (no home edge).")
    parser.add_argument("--list", dest="list_only", action="store_true",
                        help="List team ratings and exit.")
    parser.add_argument("--season-year", dest="season_year", type=int, default=None,
                        help="Season year for NCAA statistics (defaults to current year).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teams, book = build_rating_book(args.data_dir, season_year=args.season_year)

    if args.list_only or not (args.team_one and args.team_two):
        cols = ["team_name", "offense_rating", "defense_rating", "special_rating", "power_rating"]
        display = teams[cols].sort_values("power_rating", ascending=False)
        pd.set_option("display.float_format", lambda v: f"{v:0.2f}")
        print(display.to_string(index=False))
        return

    result = book.predict(args.team_one, args.team_two, neutral_site=args.neutral)

    print(f"Matchup: {result['team_one']} vs {result['team_two']}")
    print(f"Spread (team_one - team_two): {result['spread_team_one_minus_team_two']:.2f} pts")
    print(f"Total: {result['total_points']:.2f} pts")
    print(f"Projected score: {result['team_one']}: {result['team_one_points']:.1f} | {result['team_two']}: {result['team_two_points']:.1f}")
    print(f"Win probability {result['team_one']}: {result['team_one_win_prob']*100:0.1f}%")
    print(f"Win probability {result['team_two']}: {result['team_two_win_prob']*100:0.1f}%")
    ml_one = result['team_one_moneyline']
    ml_two = result['team_two_moneyline']
    if ml_one is not None and ml_two is not None:
        print(f"Moneyline: {result['team_one']} {ml_one:+.0f} | {result['team_two']} {ml_two:+.0f}")
    else:
        print("Moneyline: not defined (probabilities at bounds).")


if __name__ == "__main__":
    main()
