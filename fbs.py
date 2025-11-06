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
import datetime as dt
import json
import logging
import math
import re
import warnings
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

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
from cfb.io import cached_cfbd
from cfb.io import the_odds_api
from cfb.model import BayesianConfig, RatingBook, RatingConstants, fit_linear_calibrations, fit_probability_sigma
from cfb.names import normalize_team

BASE_URL = CFBD_BASE_URL

logger = logging.getLogger(__name__)

CONFIG = load_config()
FBS_CONFIG = CONFIG.get("fbs", {})
_MARKET_CONFIG = FBS_CONFIG.get("market", {})
_WEATHER_CONFIG = FBS_CONFIG.get("weather", {})
_DATA_CONFIG = FBS_CONFIG.get("data", {}) if isinstance(FBS_CONFIG.get("data"), dict) else {}


def _resolve_path(value: Optional[str], default: Optional[Path]) -> Optional[Path]:
    """Expand user-relative configuration paths, keeping ``None`` intact."""
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "null":
        return default
    return Path(text).expanduser()


def _team_key(label: Optional[str]) -> str:
    normalized = normalize_team(label)
    normalized = normalized.replace("&", " AND ")
    return re.sub(r"[^A-Z0-9]", "", normalized)


PFF_DATA_DIR = _resolve_path(_DATA_CONFIG.get("pff_dir"), Path("data/pff/fbs"))
PFF_COMBINED_DATA_DIR = _resolve_path(_DATA_CONFIG.get("pff_combined_dir"), None)

MARKET_SPREAD_WEIGHT = float(_MARKET_CONFIG.get("spread_weight", 0.4))
MARKET_TOTAL_WEIGHT = float(_MARKET_CONFIG.get("total_weight", 0.4))
_MARKET_PROVIDERS = _MARKET_CONFIG.get("providers")
if isinstance(_MARKET_PROVIDERS, (list, tuple)):
    FBS_MARKET_PROVIDERS = [str(name).strip() for name in _MARKET_PROVIDERS if str(name).strip()]
else:
    FBS_MARKET_PROVIDERS = []

_THE_ODDS_API_CONFIG = _MARKET_CONFIG.get("the_odds_api", {}) if isinstance(_MARKET_CONFIG.get("the_odds_api"), dict) else {}
_THE_ODDS_API_ENABLED = bool(_THE_ODDS_API_CONFIG.get("enabled", True))
THE_ODDS_API_SPORT_KEY = str(_THE_ODDS_API_CONFIG.get("sport_key", "americanfootball_ncaaf")).strip() or ""
if not THE_ODDS_API_SPORT_KEY:
    _THE_ODDS_API_ENABLED = False
def _coerce_sequence(value: object, fallback: Sequence[str]) -> tuple[str, ...]:
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

DEFAULT_PROVIDER_BIAS = {"spread": 0.0, "total": 0.0}
PROVIDER_BIAS_ADJUSTMENTS = {
    "draftkings": {"spread": 0.43, "total": -0.93},
    "consensus": {"spread": 1.50, "total": -0.30},
    "bovada": {"spread": 1.50, "total": -0.30},
}
PROVIDER_WEIGHT_OVERRIDES = {}
DEFAULT_BIAS_PROVIDER = os.environ.get("FBS_DEFAULT_BIAS_PROVIDER", "draftkings").lower()
ERA_TOTAL_SCALE = float(os.environ.get("FBS_ERA_TOTAL_SCALE", "0.986"))
PROB_SIGMA_SCALE = float(os.environ.get("FBS_PROB_SIGMA_SCALE", "0.93"))

_RATING_CONFIG = FBS_CONFIG.get("ratings", {}) if isinstance(FBS_CONFIG.get("ratings"), dict) else {}
_BAYESIAN_CONFIG = FBS_CONFIG.get("bayesian", {}) if isinstance(FBS_CONFIG.get("bayesian"), dict) else {}
_RATING_WEIGHTS_PATH_RAW = _RATING_CONFIG.get("weights_path", "calibration/fbs_rating_weights.json")
RATING_WEIGHTS_PATH = Path(str(_RATING_WEIGHTS_PATH_RAW)).expanduser()


def _rating_constants_from_config() -> RatingConstants:
    constants = RatingConstants(
        avg_total=float(_RATING_CONFIG.get("avg_total", 53.5109)),
        home_field_advantage=float(_RATING_CONFIG.get("home_field_advantage", 3.4066)),
        offense_factor=float(_RATING_CONFIG.get("offense_factor", 6.2171)),
        defense_factor=float(_RATING_CONFIG.get("defense_factor", 5.0088)),
        power_factor=float(_RATING_CONFIG.get("power_factor", 2.1035)),
        spread_sigma=float(_RATING_CONFIG.get("spread_sigma", 15.0)),
    )
    constants.avg_total *= ERA_TOTAL_SCALE
    constants.spread_sigma *= PROB_SIGMA_SCALE
    return constants


def _bayesian_config_from_config() -> Optional[BayesianConfig]:
    if not _BAYESIAN_CONFIG or not bool(_BAYESIAN_CONFIG.get("enabled", False)):
        return None
    total_prior_mean_raw = _BAYESIAN_CONFIG.get("total_prior_mean", None)
    total_prior_mean = (
        float(total_prior_mean_raw)
        if total_prior_mean_raw is not None
        else None
    )
    if total_prior_mean is not None:
        total_prior_mean *= ERA_TOTAL_SCALE
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

ODDSLOGIC_ARCHIVE_DIR = Path(os.environ.get("ODDSLOGIC_ARCHIVE_DIR", "oddslogic_ncaa_all"))

_INJURY_WARNING_EMITTED = False


def _normalize_provider_name(name: Optional[str]) -> str:
    return str(name or "").strip().lower()


def _provider_bias(provider: Optional[str]) -> Dict[str, float]:
    key = _normalize_provider_name(provider) or DEFAULT_BIAS_PROVIDER
    return PROVIDER_BIAS_ADJUSTMENTS.get(key, DEFAULT_PROVIDER_BIAS)


def _provider_weight(provider: Optional[str]) -> float:
    return 1.0


def _select_primary_provider(provider_names: Iterable[str]) -> Optional[str]:
    normalized = [_normalize_provider_name(name) for name in provider_names if str(name or "").strip()]
    for candidate in ("draftkings", "consensus", "bovada"):
        if candidate in normalized:
            return candidate
    return normalized[0] if normalized else None


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


def _datetime_to_iso(value: Optional[dt.datetime]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    else:
        value = value.astimezone(dt.timezone.utc)
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_match_label(label: Optional[str]) -> str:
    if not label:
        return ""
    text = normalize_team(label)
    text = re.sub(r"\(.*?\)", " ", text)
    text = text.replace("-", " ")
    text = text.replace("/", " ")
    text = text.replace("'", "")
    text = text.replace(".", "")
    text = text.replace("&", " AND ")
    text = " ".join(text.split())
    return text


def _tokenize_match_label(label: Optional[str]) -> tuple[str, ...]:
    cleaned = _clean_match_label(label)
    if not cleaned:
        return ()
    tokens = tuple(token for token in cleaned.split(" ") if token)
    if not tokens and label:
        return (_clean_match_label(label).strip(),)
    return tokens


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
        if target_str.startswith(candidate_str + " ") or candidate_str.startswith(target_str + " "):
            return True
    return False


def _resolve_odds_api_match(
    cfbd_index: Dict[int, dict],
    home_tokens: tuple[str, ...],
    away_tokens: tuple[str, ...],
    commence_time: Optional[dt.datetime],
) -> Optional[tuple[int, bool]]:
    if not cfbd_index:
        return None
    candidates: list[tuple[float, int, bool]] = []
    for game_id, meta in cfbd_index.items():
        cfbd_home = meta.get("home_tokens") or ()
        cfbd_away = meta.get("away_tokens") or ()
        kickoff = meta.get("kickoff")
        if _tokens_match(cfbd_home, home_tokens) and _tokens_match(cfbd_away, away_tokens):
            delta = float("inf")
            if kickoff and commence_time:
                delta = abs((kickoff - commence_time).total_seconds())
            candidates.append((delta, game_id, False))
        elif _tokens_match(cfbd_home, away_tokens) and _tokens_match(cfbd_away, home_tokens):
            delta = float("inf")
            if kickoff and commence_time:
                delta = abs((kickoff - commence_time).total_seconds())
            candidates.append((delta, game_id, True))
    if not candidates:
        return None
    candidates.sort()
    _, game_id, invert = candidates[0]
    return game_id, invert


def _summarize_provider_lines(data: dict) -> None:
    provider_lines = data.get("provider_lines") or {}
    if not provider_lines:
        data["providers"] = []
        data["primary_provider"] = None
        return
    provider_names = sorted(provider_lines.keys())
    data["providers"] = provider_names
    primary_key = _select_primary_provider(provider_names)
    primary_name = None
    if primary_key:
        for name in provider_names:
            if _normalize_provider_name(name) == primary_key:
                primary_name = name
                break
        if primary_name is None:
            primary_name = primary_key
    data["primary_provider"] = primary_name
    spread_choice: Optional[float] = None
    total_choice: Optional[float] = None
    if primary_name and primary_name in provider_lines:
        primary_info = provider_lines.get(primary_name, {})
        primary_spread = primary_info.get("spread_home")
        if primary_spread is not None:
            spread_choice = -float(primary_spread)
        primary_total = primary_info.get("total")
        if primary_total is not None:
            total_choice = float(primary_total)
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
        data["spread"] = spread_choice
    if total_choice is not None:
        data["total"] = total_choice


def _merge_the_odds_api_events(
    lookup: Dict[int, dict],
    cfbd_index: Dict[int, dict],
    events: Iterable[dict],
    *,
    provider_filter: Optional[set[str]] = None,
) -> None:
    if not lookup or not events:
        return

    for event in events:
        rows = the_odds_api.normalise_prices(event)
        if not rows:
            continue
        home_tokens = _tokenize_match_label(event.get("home_team"))
        away_tokens = _tokenize_match_label(event.get("away_team"))
        commence = rows[0].get("commence_time")
        match = _resolve_odds_api_match(cfbd_index, home_tokens, away_tokens, commence)
        if not match:
            continue
        game_id, invert = match
        data = lookup.get(game_id)
        if not data:
            continue
        provider_lines = data.setdefault("provider_lines", {})
        provider_weights = data.setdefault("provider_weights", {})
        for row in rows:
            provider_key = _format_bookmaker_name(row.get("bookmaker"))
            normalized_provider = _normalize_provider_name(provider_key)
            if provider_filter and normalized_provider not in provider_filter:
                continue
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
                price_draw = row.get("price_draw")
                if invert:
                    price_home, price_away = price_away, price_home
                if price_home is not None:
                    existing["home_moneyline"] = price_home
                if price_away is not None:
                    existing["away_moneyline"] = price_away
                if price_draw is not None:
                    existing["draw_moneyline"] = price_draw
            provider_lines[provider_key] = existing
            if provider_key not in provider_weights:
                provider_weights[provider_key] = 1.0


def apply_bias_recenter(
    result: Dict[str, float],
    *,
    provider_hint: Optional[str],
    prob_sigma: Optional[float],
) -> Dict[str, float]:
    bias = _provider_bias(provider_hint)
    if bias == DEFAULT_PROVIDER_BIAS:
        return result
    updated = result.copy()
    spread = updated.get("spread_home_minus_away")
    total = updated.get("total_points")
    if spread is not None:
        spread = float(spread) + bias["spread"]
        updated["spread_home_minus_away"] = spread
    if total is not None:
        total = float(total) + bias["total"]
        updated["total_points"] = total
    if spread is not None and total is not None:
        home_points = (total + spread) / 2.0
        away_points = total - home_points
        updated["home_points"] = home_points
        updated["away_points"] = away_points
    sigma = prob_sigma
    if sigma and spread is not None:
        win_prob = 0.5 * (1.0 + math.erf(spread / (sigma * math.sqrt(2))))
        updated["home_win_prob"] = win_prob
        updated["away_win_prob"] = 1.0 - win_prob
        updated["home_moneyline"] = market_utils.moneyline_from_prob(win_prob)
        updated["away_moneyline"] = market_utils.moneyline_from_prob(1.0 - win_prob)
        updated["prob_sigma_used"] = sigma
    updated["bias_provider"] = _normalize_provider_name(provider_hint) or DEFAULT_BIAS_PROVIDER
    return updated


def market_weight_for_provider(provider_hint: Optional[str]) -> float:
    return _provider_weight(provider_hint)


def select_primary_provider(provider_names: Iterable[str]) -> Optional[str]:
    return _select_primary_provider(provider_names)


def primary_provider_from_market(market: Optional[dict]) -> Optional[str]:
    if not market:
        return None
    explicit = market.get("primary_provider")
    if explicit:
        return explicit
    provider_lines = market.get("providers") or list((market.get("provider_lines") or {}).keys())
    selected = _select_primary_provider(provider_lines)
    if not selected:
        return None
    for name in provider_lines:
        if _normalize_provider_name(name) == selected:
            return name
    return selected


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


def _flatten_season_advanced(records: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    top_level_metrics = {
        "ppa": "ppa",
        "successRate": "success_rate",
        "explosiveness": "explosiveness",
        "powerSuccess": "power_success",
        "stuffRate": "stuff_rate",
        "standardDownSuccessRate": "standard_down_success",
        "passingDownSuccessRate": "passing_down_success",
        "standardDownPPA": "standard_down_ppa",
        "passingDownPPA": "passing_down_ppa",
        "lineYards": "line_yards",
        "secondLevelYards": "second_level_yards",
        "openFieldYards": "open_field_yards",
    }
    sub_unit_metrics = {
        "rushing": {
            "ppa": "rush_ppa",
            "successRate": "rush_success_rate",
            "explosiveness": "rush_explosiveness",
        },
        "passing": {
            "ppa": "pass_ppa",
            "successRate": "pass_success_rate",
            "explosiveness": "pass_explosiveness",
        },
    }
    for entry in records:
        team = entry.get("team")
        if not team:
            continue
        offense = entry.get("offense") or {}
        defense = entry.get("defense") or {}
        row: dict[str, float | str] = {"team": team}

        def _assign(prefix: str, source: dict) -> None:
            for api_key, alias in top_level_metrics.items():
                key = f"{prefix}_{alias}"
                value = source.get(api_key) if isinstance(source, dict) else None
                row[key] = value
            for unit_key, unit_metrics in sub_unit_metrics.items():
                unit_source = source.get(unit_key) if isinstance(source, dict) else None
                for api_key, alias in unit_metrics.items():
                    key = f"{prefix}_{alias}"
                    value = unit_source.get(api_key) if isinstance(unit_source, dict) else None
                    row[key] = value

        _assign("adv_off", offense)
        _assign("adv_def", defense)
        rows.append(row)
    return pd.DataFrame(rows)


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


def _fit_feature_weights(
    teams: pd.DataFrame,
    games: Iterable[dict],
    offense_cols: Sequence[str],
    defense_cols: Sequence[str],
    *,
    l2: float = 12.0,
    min_rows: int = 40,
) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
    if not games:
        return None
    if not offense_cols or not defense_cols:
        return None
    feature_cols = set(offense_cols) | set(defense_cols)
    if "team" not in teams.columns:
        return None
    lookup = teams.set_index("team")
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for game in games:
        if game.get("completed") is not True:
            continue
        if game.get("homeClassification") != "fbs" or game.get("awayClassification") != "fbs":
            continue
        home = game.get("homeTeam")
        away = game.get("awayTeam")
        if home not in lookup.index or away not in lookup.index:
            continue
        home_pts = game.get("homePoints")
        away_pts = game.get("awayPoints")
        if home_pts is None or away_pts is None:
            continue
        home_row = lookup.loc[home, list(feature_cols)].astype(float)
        away_row = lookup.loc[away, list(feature_cols)].astype(float)
        home_off = home_row.reindex(offense_cols).fillna(0.0).to_numpy(dtype=float)
        away_off = away_row.reindex(offense_cols).fillna(0.0).to_numpy(dtype=float)
        home_def = home_row.reindex(defense_cols).fillna(0.0).to_numpy(dtype=float)
        away_def = away_row.reindex(defense_cols).fillna(0.0).to_numpy(dtype=float)
        off_diff = home_off - away_off
        def_diff = away_def - home_def
        row = np.concatenate([off_diff, def_diff])
        rows.append(row)
        targets.append(float(home_pts) - float(away_pts))
    sample_size = len(rows)
    if sample_size < min_rows:
        return None
    X = np.vstack(rows)
    y = np.array(targets)
    # Add bias column and ridge regularization (no penalty on intercept).
    ones = np.ones((sample_size, 1))
    X_aug = np.hstack([ones, X])
    A = X_aug.T @ X_aug
    reg = l2 * np.eye(A.shape[0])
    reg[0, 0] = 0.0
    b = X_aug.T @ y
    try:
        solution = np.linalg.solve(A + reg, b)
    except np.linalg.LinAlgError:
        return None
    weights = solution[1:]
    off_weights = weights[: len(offense_cols)]
    def_weights = weights[len(offense_cols) :]
    # Coverage-based shrinkage toward zero for unreliable feeds.
    coverage_off = np.array([teams[col].notna().mean() if col in teams else 0.0 for col in offense_cols])
    coverage_def = np.array([teams[col].notna().mean() if col in teams else 0.0 for col in defense_cols])
    off_weights = off_weights * np.clip(coverage_off, 0.0, 1.0)
    def_weights = def_weights * np.clip(coverage_def, 0.0, 1.0)
    return off_weights, def_weights, sample_size


def _fit_power_weights(
    teams: pd.DataFrame,
    games: Iterable[dict],
    feature_cols: Sequence[str],
    *,
    l2: float = 6.0,
    min_rows: int = 40,
) -> Optional[tuple[dict[str, float], int]]:
    if not games:
        return None
    if not feature_cols:
        return None
    if "team" not in teams.columns:
        return None
    lookup = teams.set_index("team")
    existing_cols = [col for col in feature_cols if col in lookup.columns]
    if not existing_cols:
        return None
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for game in games:
        if game.get("completed") is not True:
            continue
        if game.get("homeClassification") != "fbs" or game.get("awayClassification") != "fbs":
            continue
        home = game.get("homeTeam")
        away = game.get("awayTeam")
        if home not in lookup.index or away not in lookup.index:
            continue
        home_pts = game.get("homePoints")
        away_pts = game.get("awayPoints")
        if home_pts is None or away_pts is None:
            continue
        try:
            home_row = lookup.loc[home, existing_cols].astype(float)
            away_row = lookup.loc[away, existing_cols].astype(float)
        except KeyError:
            continue
        diff = (
            home_row.reindex(existing_cols).fillna(0.0).to_numpy(dtype=float)
            - away_row.reindex(existing_cols).fillna(0.0).to_numpy(dtype=float)
        )
        rows.append(diff)
        targets.append(float(home_pts) - float(away_pts))
    sample_size = len(rows)
    if sample_size < min_rows:
        return None
    X = np.vstack(rows)
    y = np.array(targets)
    ones = np.ones((sample_size, 1))
    X_aug = np.hstack([ones, X])
    A = X_aug.T @ X_aug
    reg = l2 * np.eye(A.shape[0])
    reg[0, 0] = 0.0
    b = X_aug.T @ y
    try:
        solution = np.linalg.solve(A + reg, b)
    except np.linalg.LinAlgError:
        return None
    weights = solution[1:]
    weight_map = {col: float(weight) for col, weight in zip(existing_cols, weights)}
    return weight_map, sample_size


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


def _load_rating_weights(
    offense_cols: Sequence[str],
    defense_cols: Sequence[str],
    power_cols: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    payload: dict[str, dict[str, float]] = {}
    if RATING_WEIGHTS_PATH.exists():
        try:
            with RATING_WEIGHTS_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                payload = data
        except (json.JSONDecodeError, OSError):
            payload = {}
    offense_map = payload.get("offense", {}) if isinstance(payload.get("offense"), dict) else {}
    defense_map = payload.get("defense", {}) if isinstance(payload.get("defense"), dict) else {}
    power_map = payload.get("power", {}) if isinstance(payload.get("power"), dict) else {}
    off_weights = np.array([float(offense_map.get(col, 0.0)) for col in offense_cols], dtype=float)
    def_weights = np.array([float(defense_map.get(col, 0.0)) for col in defense_cols], dtype=float)
    power_weights = np.array([float(power_map.get(col, 0.0)) for col in power_cols], dtype=float)
    meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
    return off_weights, def_weights, power_weights, meta


def _persist_rating_weights(
    offense_cols: Sequence[str],
    defense_cols: Sequence[str],
    power_cols: Sequence[str],
    offense_weights: np.ndarray,
    defense_weights: np.ndarray,
    power_weights: np.ndarray,
    *,
    offense_rows: Optional[int],
    power_rows: Optional[int],
) -> None:
    payload = {
        "generated": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "offense": {col: float(weight) for col, weight in zip(offense_cols, offense_weights)},
        "defense": {col: float(weight) for col, weight in zip(defense_cols, defense_weights)},
        "power": {col: float(weight) for col, weight in zip(power_cols, power_weights)},
        "meta": {
            "offense_rows": offense_rows,
            "defense_rows": offense_rows,
            "power_rows": power_rows,
        },
    }
    try:
        RATING_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RATING_WEIGHTS_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    except OSError:
        # Non-fatal: continue without persisting if filesystem unavailable.
        pass


def load_pff_team_metrics(
    data_dir: Optional[Path] = None,
    *,
    combined_dir: Optional[Path] = None,
) -> pd.DataFrame:
    primary_dir = data_dir or PFF_DATA_DIR
    combined = combined_dir or PFF_COMBINED_DATA_DIR
    source_dirs: list[Path] = []
    if combined:
        combined_expanded = combined.expanduser()
        if combined_expanded.exists():
            source_dirs.append(combined_expanded)
    if primary_dir:
        primary_expanded = primary_dir.expanduser()
        if primary_expanded.exists():
            source_dirs.append(primary_expanded)
        else:
            source_dirs.append(primary_expanded)
    if not source_dirs:
        return pd.DataFrame(columns=["team"])

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
    pff = pff.rename(columns={"team_name": "team"})
    pff["team_key"] = pff["team"].map(_team_key)
    pff = pff.drop_duplicates(subset="team_key", keep="first")
    return pff


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

    game_counts = pd.DataFrame(columns=["team", "games_played"])
    try:
        game_records = client.get("/ppa/games", year=year, seasonType=season_type)
        games = _flatten_game_ppa(game_records)
        if through_week is not None:
            games = games[games["week"].fillna(0) <= through_week]
        adj = compute_opponent_adjusted_ppa(ppa, games)
        if not games.empty:
            game_counts = (
                games.groupby("team")["game_id"]
                .nunique()
                .reset_index()
                .rename(columns={"game_id": "games_played"})
            )
    except requests.HTTPError:
        adj = pd.DataFrame(columns=["team"])

    if not adj.empty:
        teams = teams.merge(adj, on="team", how="left")

    # Prefer cached season advanced stats; fall back to live call if needed.
    try:
        advanced_records = cached_cfbd.load_advanced_team(
            year,
            season_type=season_type,
            fetch_if_missing=True,
            api_key=api_key or os.environ.get("CFBD_API_KEY"),
        )
        if not advanced_records.empty:
            advanced_df = _flatten_season_advanced(advanced_records.to_dict("records"))
            teams = teams.merge(advanced_df, on="team", how="left")
    except Exception:
        advanced_df = pd.DataFrame(columns=["team"])

    if not game_counts.empty:
        teams = teams.merge(game_counts, on="team", how="left")

    pff = load_pff_team_metrics()
    if not pff.empty:
        teams["team_key"] = teams["team"].map(_team_key)
        pff = pff.rename(columns={"team": "pff_team"})
        teams = teams.merge(pff, on="team_key", how="left")

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

    if "games_played" not in teams.columns:
        teams["games_played"] = 0.0
    else:
        teams["games_played"] = teams["games_played"].fillna(0.0)

    if "team_key" in teams.columns:
        teams = teams.drop(columns=["team_key"])

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
    provider: Optional[str] = None,
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
    provider : Optional[str], optional
        Legacy single-provider filter.
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
    provider_names_input: list[str] = [name for name in FBS_MARKET_PROVIDERS]
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
    provider_filter = {_normalize_provider_name(name) for name in provider_names_input} if provider_names_input else None

    entries = list(client.get("/lines", **params))

    lookup: Dict[int, dict] = {}
    date_keys: set[dt.date] = set()
    cfbd_games: Dict[int, dict] = {}
    cfbd_index: Dict[int, dict] = {}
    for entry in entries:
        game_id = entry.get("id")
        if not game_id:
            continue
        start_date_raw = entry.get("startDate")
        kickoff_dt = None
        kickoff_py = None
        if start_date_raw:
            try:
                kickoff_dt = pd.to_datetime(start_date_raw, utc=True)
            except (TypeError, ValueError):
                kickoff_dt = pd.NaT
            if isinstance(kickoff_dt, pd.Timestamp) and not pd.isna(kickoff_dt):
                kickoff_py = kickoff_dt.to_pydatetime()
                date_keys.add(kickoff_py.date())

        cfbd_games[game_id] = entry
        lines = entry.get("lines") or []
        provider_names: set[str] = set()
        provider_lines: Dict[str, dict] = {}
        for line in lines:
            provider = line.get("provider")
            provider_normalized = str(provider or "").strip()
            if not provider_normalized:
                continue
            normalized_name = _normalize_provider_name(provider_normalized)
            if provider_filter and normalized_name not in provider_filter:
                continue
            provider_names.add(provider_normalized)

            spread_raw = _coerce_float(line.get("spread"))
            total_raw = _coerce_float(line.get("overUnder"))
            home_ml = _coerce_float(line.get("homeMoneyline"))
            away_ml = _coerce_float(line.get("awayMoneyline"))
            last_updated = line.get("lastUpdated") or line.get("updated")

            if spread_raw is not None:
                spread_home = float(spread_raw)
            else:
                spread_home = None
            provider_lines[provider_normalized] = {
                "spread_home": spread_home,
                "spread_away": -spread_home if spread_home is not None else None,
                "total": total_raw,
                "home_moneyline": home_ml,
                "away_moneyline": away_ml,
                "last_updated": last_updated,
            }
        if provider_filter and not provider_names:
            continue

        spreads = [
            -float(info.get("spread_home"))
            for info in provider_lines.values()
            if info.get("spread_home") is not None
        ]
        totals = [
            float(info.get("total"))
            for info in provider_lines.values()
            if info.get("total") is not None
        ]
        primary_key = _select_primary_provider(provider_names)
        primary_provider = None
        if primary_key:
            for name in provider_names:
                if _normalize_provider_name(name) == primary_key:
                    primary_provider = name
                    break
            if primary_provider is None:
                primary_provider = primary_key
        spread_value: Optional[float] = None
        total_value: Optional[float] = None
        if primary_provider and primary_provider in provider_lines:
            primary_info = provider_lines.get(primary_provider, {})
            primary_spread = primary_info.get("spread_home")
            if primary_spread is not None:
                spread_value = -float(primary_spread)
            primary_total = primary_info.get("total")
            if primary_total is not None:
                total_value = float(primary_total)
        if spread_value is None and spreads:
            spread_value = float(np.mean(spreads))
        if total_value is None and totals:
            total_value = float(np.mean(totals))

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
            "primary_provider": primary_provider,
            "provider_weights": {name: 1.0 for name in provider_names},
        }

        cfbd_index[game_id] = {
            "home_tokens": _tokenize_match_label(entry.get("homeTeam")),
            "away_tokens": _tokenize_match_label(entry.get("awayTeam")),
            "kickoff": kickoff_py,
        }

    if _THE_ODDS_API_ENABLED and THE_ODDS_API_SPORT_KEY:
        try:
            odds_events = the_odds_api.fetch_current_odds(
                THE_ODDS_API_SPORT_KEY,
                regions=THE_ODDS_API_REGIONS,
                markets=THE_ODDS_API_MARKETS,
                bookmakers=THE_ODDS_API_BOOKMAKERS,
                odds_format=THE_ODDS_API_FORMAT,
            )
        except the_odds_api.TheOddsAPIError as exc:
            logger.warning("The Odds API fetch failed: %s", exc)
            odds_events = []
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Unexpected error fetching The Odds API odds: %s", exc)
            odds_events = []
        else:
            _merge_the_odds_api_events(
                lookup,
                cfbd_index,
                odds_events,
                provider_filter=provider_filter,
            )

    live_lookup: Dict[Tuple[dt.date, str, str], Dict[str, object]] = {}
    if date_keys:
        live_lookup = oddslogic_io.fetch_live_market_lookup(
            date_keys,
            classification=classification or "fbs",
            providers=provider_names_input or None,
        )

    if live_lookup:
        for game_id, data in lookup.items():
            entry = cfbd_games.get(game_id, {})
            start_raw = entry.get("startDate")
            kickoff_date = None
            if start_raw:
                try:
                    kickoff_dt = pd.to_datetime(start_raw)
                except (TypeError, ValueError):
                    kickoff_dt = pd.NaT
                if not pd.isna(kickoff_dt):
                    kickoff_date = kickoff_dt.date()
            home_norm = oddslogic_loader.normalize_label(data.get("home_team") or "")
            away_norm = oddslogic_loader.normalize_label(data.get("away_team") or "")
            live_key = (kickoff_date, home_norm, away_norm)
            invert = False
            live_entry = live_lookup.get(live_key)
            if not live_entry:
                alt_key = (kickoff_date, away_norm, home_norm)
                live_entry = live_lookup.get(alt_key)
                if live_entry:
                    invert = True
            if not live_entry:
                continue

            provider_lines = data.setdefault("provider_lines", {})
            for provider_name, payload in live_entry["providers"].items():
                spread_value = payload.get("spread_value")
                if spread_value is not None and invert:
                    spread_value = -spread_value
                total_value = payload.get("total_value")
                last_updated = payload.get("spread_updated") or payload.get("total_updated")
                if not last_updated:
                    ts = payload.get("spread_updated") or payload.get("total_updated")
                existing = provider_lines.get(provider_name, {})
                provider_lines[provider_name] = {
                    "spread_home": spread_value,
                    "spread_away": -spread_value if spread_value is not None else None,
                    "total": total_value,
                    "home_moneyline": existing.get("home_moneyline"),
                    "away_moneyline": existing.get("away_moneyline"),
                    "last_updated": payload.get("spread_updated") or payload.get("total_updated"),
                    "source": "OddsLogic",
                    "open_spread_home": (
                        -payload["open_spread_value"] if invert and payload.get("open_spread_value") is not None else payload.get("open_spread_value")
                    ),
                    "open_total": payload.get("open_total_value"),
                }
            _summarize_provider_lines(data)
    for data in lookup.values():
        _summarize_provider_lines(data)
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


def build_ratings(teams: pd.DataFrame, calibration_games: Optional[Iterable[dict]] = None) -> pd.DataFrame:
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
        "adv_off_ppa",
        "adv_off_success_rate",
        "adv_off_standard_down_success",
        "adv_off_passing_down_success",
        "adv_off_explosiveness",
        "adv_off_power_success",
        "adv_off_stuff_rate",
        "adv_off_line_yards",
        "adv_off_second_level_yards",
        "adv_off_open_field_yards",
        "adv_off_rush_success_rate",
        "adv_off_rush_explosiveness",
        "adv_off_pass_success_rate",
        "adv_off_pass_explosiveness",
        "adv_def_ppa",
        "adv_def_success_rate",
        "adv_def_standard_down_success",
        "adv_def_passing_down_success",
        "adv_def_explosiveness",
        "adv_def_power_success",
        "adv_def_stuff_rate",
        "adv_def_line_yards",
        "adv_def_second_level_yards",
        "adv_def_open_field_yards",
        "adv_def_rush_success_rate",
        "adv_def_rush_explosiveness",
        "adv_def_pass_success_rate",
        "adv_def_pass_explosiveness",
    ]
    add_z_scores(teams, candidates)

    for metric in candidates:
        z_col = f"{metric}_z"
        if z_col in teams.columns:
            teams[z_col] = teams[z_col].fillna(0.0)

    offense_feature_cols = [
        "ppa_offense_z",
        "ppa_passing_z",
        "ppa_rushing_z",
        "sp_offense_z",
        "fpi_offense_z",
        "sp_special_z",
        "adj_ppa_offense_z",
        "adj_ppa_offense_pass_z",
        "adj_ppa_offense_rush_z",
        "receiving_grade_route_z",
        "receiving_yprr_z",
        "blocking_grade_pass_z",
        "blocking_grade_run_z",
        "blocking_pbe_z",
        "adv_off_success_rate_z",
        "adv_off_standard_down_success_z",
        "adv_off_passing_down_success_z",
        "adv_off_line_yards_z",
        "adv_off_second_level_yards_z",
        "adv_off_open_field_yards_z",
        "adv_off_rush_explosiveness_z",
        "adv_off_pass_explosiveness_z",
        "adv_off_power_success_z",
        "adv_off_stuff_rate_z",
    ]
    defense_feature_cols = [
        "ppa_defense_z",
        "ppa_def_pass_z",
        "ppa_def_rush_z",
        "sp_defense_z",
        "fpi_defense_z",
        "adj_ppa_defense_z",
        "adj_ppa_defense_pass_z",
        "adj_ppa_defense_rush_z",
        "defense_grade_overall_z",
        "defense_grade_pass_rush_z",
        "defense_grade_coverage_z",
        "defense_grade_run_z",
        "adv_def_success_rate_z",
        "adv_def_standard_down_success_z",
        "adv_def_passing_down_success_z",
        "adv_def_explosiveness_z",
        "adv_def_line_yards_z",
        "adv_def_second_level_yards_z",
        "adv_def_open_field_yards_z",
        "adv_def_rush_explosiveness_z",
        "adv_def_pass_explosiveness_z",
        "adv_def_power_success_z",
        "adv_def_stuff_rate_z",
    ]

    available_off_cols = [col for col in offense_feature_cols if col in teams]
    available_def_cols = [col for col in defense_feature_cols if col in teams]
    power_feature_cols = ["offense_rating", "defense_rating", "sp_rating_z", "fpi_z", "elo_z"]

    stored_off_weights, stored_def_weights, stored_power_weights, _ = _load_rating_weights(
        available_off_cols,
        available_def_cols,
        power_feature_cols,
    )

    off_weights = stored_off_weights.copy()
    def_weights = stored_def_weights.copy()
    offense_rows = None
    if calibration_games:
        fitted = _fit_feature_weights(
            teams,
            calibration_games,
            available_off_cols,
            available_def_cols,
        )
        if fitted:
            off_weights, def_weights, offense_rows = fitted

    if available_off_cols:
        off_matrix = teams.reindex(columns=available_off_cols).fillna(0.0).to_numpy(dtype=float)
        offense_series = pd.Series(off_matrix @ off_weights, index=teams.index)
        offense_series -= float(offense_series.mean())
    else:
        offense_series = pd.Series(0.0, index=teams.index, dtype=float)

    if available_def_cols:
        def_matrix = teams.reindex(columns=available_def_cols).fillna(0.0).to_numpy(dtype=float)
        defense_series = pd.Series(-(def_matrix @ def_weights), index=teams.index)
        defense_series -= float(defense_series.mean())
    else:
        defense_series = pd.Series(0.0, index=teams.index, dtype=float)

    teams["offense_rating"] = offense_series
    teams["defense_rating"] = defense_series

    teams["offense_rating"] = teams["offense_rating"] - teams.get("injury_offense_penalty", 0.0)
    teams["defense_rating"] = teams["defense_rating"] + teams.get("injury_defense_penalty", 0.0)

    for col in ("sp_rating_z", "fpi_z", "elo_z"):
        if col not in teams.columns:
            teams[col] = 0.0

    if stored_power_weights.shape != (len(power_feature_cols),):
        stored_power_weights = np.zeros(len(power_feature_cols), dtype=float)
    power_weights = stored_power_weights.copy()
    power_rows = None
    if calibration_games:
        fitted_power = _fit_power_weights(teams, calibration_games, power_feature_cols)
        if fitted_power:
            power_map, power_rows = fitted_power
            power_weights = np.array([float(power_map.get(col, 0.0)) for col in power_feature_cols], dtype=float)

    power_matrix = teams.reindex(columns=power_feature_cols).fillna(0.0).to_numpy(dtype=float)
    power_series = pd.Series(power_matrix @ power_weights, index=teams.index)
    power_series -= float(power_series.mean())
    teams["power_rating"] = power_series

    if (offense_rows and offense_rows > 0) or (power_rows and power_rows > 0):
        _persist_rating_weights(
            available_off_cols,
            available_def_cols,
            power_feature_cols,
            off_weights,
            def_weights,
            power_weights,
            offense_rows=offense_rows,
            power_rows=power_rows,
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
    raw_residuals: list[float] = []
    if {"team", "injury_offense_penalty", "injury_defense_penalty"}.issubset(ratings.columns):
        injury_lookup = (
            ratings[["team", "injury_offense_penalty", "injury_defense_penalty"]]
            .set_index("team")
            .fillna(0.0)
        )
    else:
        injury_lookup = pd.DataFrame()
    decay = 0.35
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
        raw_residuals.append(float(residual))
        base_weight = float(game.get("__cal_weight", 1.0))
        weeks_since = max(0, (up_to_week or game.get("week") or 0) - (game.get("week") or 0))
        recency = math.exp(-decay * weeks_since)
        home_injury = 0.0
        away_injury = 0.0
        if not injury_lookup.empty:
            if home in injury_lookup.index:
                home_injury = abs(float(injury_lookup.at[home, "injury_offense_penalty"])) + abs(float(injury_lookup.at[home, "injury_defense_penalty"]))
            if away in injury_lookup.index:
                away_injury = abs(float(injury_lookup.at[away, "injury_offense_penalty"])) + abs(float(injury_lookup.at[away, "injury_defense_penalty"]))
        injury_factor = 1.0 / (1.0 + 0.35 * (home_injury + away_injury))
        weight = max(1e-3, base_weight * recency * injury_factor)
        sqrt_weight = math.sqrt(weight)
        row = np.zeros(len(teams))
        row[index[home]] = 1.0
        row[index[away]] = -1.0
        rows.append(row * sqrt_weight)
        residuals.append(float(residual) * sqrt_weight)

    if not rows:
        return {}

    A = np.vstack(rows)
    b = np.array(residuals)
    reg_matrix = math.sqrt(reg) * np.eye(len(teams))
    A_reg = np.vstack([A, reg_matrix])
    b_reg = np.concatenate([b, np.zeros(len(teams))])
    adjustments = np.linalg.lstsq(A_reg, b_reg, rcond=None)[0]
    adjustments = adjustments - float(np.mean(adjustments))
    adj_std = float(np.std(adjustments)) if adjustments.size else 0.0
    residual_std = float(np.std(raw_residuals)) if raw_residuals else 0.0
    if adj_std > 1e-6 and residual_std > 0.0:
        scale = float(np.clip((residual_std / adj_std) * 0.5, 0.15, 0.35))
    else:
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
    if not calibration_list:
        key_for_games = api_key or os.environ.get("CFBD_API_KEY")
        if key_for_games:
            try:
                calibration_list = fetch_games(year, key_for_games)
            except Exception:
                calibration_list = []
    teams_raw = fetch_team_metrics(
        year,
        api_key=api_key,
        through_week=adjust_week,
    )
    ratings = build_ratings(teams_raw, calibration_list if calibration_list else None)
    team_games_map: Dict[str, float] = {}
    if "games_played" in ratings.columns:
        for row in ratings[["team", "games_played"]].itertuples(index=False):
            team = str(row.team).lower()
            try:
                games_value = float(row.games_played)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(games_value):
                continue
            team_games_map[team] = max(0.0, games_value)
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
    bayesian_cfg = _bayesian_config_from_config()
    if bayesian_cfg and bayesian_cfg.total_prior_mean is None:
        bayesian_cfg.total_prior_mean = constants.avg_total

    if anchor_config and anchor_config.archive_path.exists():
        anchor_source = games_for_anchor or cfbd_games
        if anchor_source:
            anchor_book = RatingBook(
                ratings,
                constants,
                power_adjustments=adjustments,
                team_games=team_games_map,
                bayesian_config=bayesian_cfg,
            )
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
    book = RatingBook(
        ratings,
        constants,
        power_adjustments=combined_adjustments,
        team_games=team_games_map,
        bayesian_config=bayesian_cfg,
    )
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
