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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests

PFF_DATA_DIR = Path("~/Desktop/PFFMODEL_FBS").expanduser()


BASE_URL = "https://api.collegefootballdata.com"

WEATHER_REG_COEFFS = {
    "wind_high": -0.1425,
    "cold": -0.0161,
    "heat": -0.2142,
    "rain": 0.0,  # Historical fit was positive; clamp to zero to avoid inflating totals
    "snow": -0.0911,
    "humid": -0.0574,
    "dewpos": -0.0187,
}
WEATHER_CLAMP_BOUNDS = (-12.0, 6.0)

MARKET_SPREAD_WEIGHT = 0.8
MARKET_TOTAL_WEIGHT = 0.8


class CFBDClient:
    """Minimal helper around CFBD HTTP endpoints."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        api_key = api_key or os.environ.get("CFBD_API_KEY")
        if not api_key:
            raise RuntimeError(
                "CFBD API key not provided. Set CFBD_API_KEY env var or use --api-key."
            )
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def get(self, path: str, **params) -> list[dict]:
        resp = requests.get(BASE_URL + path, headers=self.headers, params=params, timeout=30)
        if resp.status_code == 401:
            raise RuntimeError(
                "CFBD API rejected the key (401). Double-check the token and Bearer prefix."
            )
        resp.raise_for_status()
        return resp.json()


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
    try:
        receiving = pd.read_csv(data_dir / "receiving_summary_FBS.csv")
        blocking = pd.read_csv(data_dir / "offense_blocking FBS.csv")
        defense = pd.read_csv(data_dir / "defense_summary FBS.csv")
        special = pd.read_csv(data_dir / "special_teams_summary_FBS.csv")
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


def fetch_team_metrics(year: int, api_key: Optional[str] = None) -> pd.DataFrame:
    client = CFBDClient(api_key)

    ppa = _flatten_ppa(client.get("/ppa/teams", year=year))
    sp = _flatten_sp(client.get("/ratings/sp", year=year))
    elo = _flatten_elo(client.get("/ratings/elo", year=year))
    fpi = _flatten_fpi(client.get("/ratings/fpi", year=year))

    teams = ppa.merge(sp, on="team", how="outer")
    teams = teams.merge(elo, on="team", how="outer")
    teams = teams.merge(fpi, on="team", how="outer")
    pff = load_pff_team_metrics()
    if not pff.empty:
        teams = teams.merge(pff, on="team", how="left")

    for col in teams.columns:
        if col == "team":
            continue
        teams[col] = pd.to_numeric(teams[col], errors="coerce")

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
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(
        BASE_URL + "/games",
        headers=headers,
        params={"year": year, "seasonType": season_type},
        timeout=60,
    )
    if resp.status_code == 401:
        raise RuntimeError("CFBD API rejected the key (401).")
    resp.raise_for_status()
    return resp.json()


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

    headers = {"Authorization": f"Bearer {api_key}"}
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

    resp = requests.get(
        BASE_URL + "/games/weather",
        headers=headers,
        params=params,
        timeout=60,
    )
    if resp.status_code == 401:
        raise RuntimeError("CFBD API rejected the key when fetching weather (401).")
    resp.raise_for_status()
    data = resp.json()
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
) -> Dict[int, dict]:
    """Return aggregated sportsbook lines keyed by game id."""

    headers = {"Authorization": f"Bearer {api_key}"}
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

    resp = requests.get(
        BASE_URL + "/lines",
        headers=headers,
        params=params,
        timeout=60,
    )
    if resp.status_code == 401:
        raise RuntimeError("CFBD API rejected the key when fetching sportsbook lines (401).")
    resp.raise_for_status()
    lookup: Dict[int, dict] = {}
    for entry in resp.json():
        game_id = entry.get("id")
        if not game_id:
            continue
        lines = entry.get("lines") or []
        spreads = []
        totals = []
        providers: set[str] = set()
        for line in lines:
            provider = line.get("provider")
            if provider:
                providers.add(provider)
            spread_raw = _coerce_float(line.get("spread"))
            if spread_raw is not None:
                # CFBD spreads are away-centric; convert to home-minus-away.
                spreads.append(-spread_raw)
            total_raw = _coerce_float(line.get("overUnder"))
            if total_raw is not None:
                totals.append(total_raw)
        lookup[game_id] = {
            "spread": float(np.mean(spreads)) if spreads else None,
            "total": float(np.mean(totals)) if totals else None,
            "providers": sorted(providers),
        }
    return lookup


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


def _moneyline_from_prob(prob: float) -> Optional[float]:
    if prob <= 0.0 or prob >= 1.0:
        return None
    return -100.0 * prob / (1 - prob) if prob >= 0.5 else 100.0 * (1 - prob) / prob


def _weather_features(weather: dict) -> dict:
    temp = _coerce_float(weather.get("temperature"))
    wind = _coerce_float(weather.get("windSpeed"))
    humidity = _coerce_float(weather.get("humidity"))
    dew_point = _coerce_float(weather.get("dewPoint"))
    condition_raw = (
        weather.get("displayValue")
        or weather.get("condition")
        or weather.get("weatherCondition")
        or weather.get("summary")
        or ""
    )
    condition = str(condition_raw).lower()
    features = {
        "wind_high": max((wind or 0.0) - 10.0, 0.0),
        "cold": max(50.0 - (temp or 50.0), 0.0),
        "heat": max((temp or 50.0) - 85.0, 0.0),
        "rain": 1.0 if ("rain" in condition or (_coerce_float(weather.get("precipitation")) or 0.0) > 0) else 0.0,
        "snow": 1.0 if ("snow" in condition or (_coerce_float(weather.get("snowfall")) or 0.0) > 0) else 0.0,
        "humid": max((humidity or 0.0) - 75.0, 0.0),
        "dewpos": max((dew_point or 0.0) - 65.0, 0.0),
    }
    return features


def _weather_effect_delta(weather: Optional[dict]) -> float:
    if not weather:
        return 0.0
    feats = _weather_features(weather)
    delta = sum(WEATHER_REG_COEFFS.get(name, 0.0) * value for name, value in feats.items())
    # Do not allow positive adjustments (weather should not inflate totals).
    delta = min(delta, 0.0)
    lo, hi = WEATHER_CLAMP_BOUNDS
    return max(lo, min(hi, delta))


def apply_market_prior(
    result: Dict[str, float],
    market: Optional[dict],
    *,
    prob_sigma: float,
    spread_weight: float = MARKET_SPREAD_WEIGHT,
    total_weight: float = MARKET_TOTAL_WEIGHT,
) -> Dict[str, float]:
    """Blend model projections with market lines when available."""

    updated = result.copy()
    spread = updated["spread_home_minus_away"]
    total = updated["total_points"]
    market_spread = None
    market_total = None
    provider_list: list[str] = []

    if market:
        market_spread = market.get("spread")
        market_total = market.get("total")
        provider_list = market.get("providers", [])

        if market_spread is not None:
            w = min(max(spread_weight, 0.0), 1.0)
            spread = (1.0 - w) * spread + w * market_spread
        if market_total is not None:
            w = min(max(total_weight, 0.0), 1.0)
            total = (1.0 - w) * total + w * market_total

    home_points = (total + spread) / 2.0
    away_points = total - home_points
    win_prob = 0.5 * (1.0 + math.erf(spread / (prob_sigma * math.sqrt(2))))

    updated["spread_home_minus_away"] = spread
    updated["total_points"] = total
    updated["home_points"] = home_points
    updated["away_points"] = away_points
    updated["home_win_prob"] = win_prob
    updated["away_win_prob"] = 1.0 - win_prob
    updated["home_moneyline"] = _moneyline_from_prob(win_prob)
    updated["away_moneyline"] = _moneyline_from_prob(1.0 - win_prob)
    updated["market_spread"] = market_spread
    updated["market_total"] = market_total
    updated["market_providers"] = provider_list
    updated["spread_vs_market"] = (
        spread - market_spread if market_spread is not None else None
    )
    updated["total_vs_market"] = (
        total - market_total if market_total is not None else None
    )
    return updated


def apply_weather_adjustment(result: Dict[str, float], weather: Optional[dict]) -> Dict[str, float]:
    if not weather:
        return result

    temp = _coerce_float(weather.get("temperature"))
    wind = _coerce_float(weather.get("windSpeed"))
    condition_raw = (
        weather.get("displayValue")
        or weather.get("condition")
        or weather.get("weatherCondition")
        or weather.get("summary")
        or ""
    )
    condition = str(condition_raw).lower()

    total_adj = _weather_effect_delta(weather)

    total_adj = max(min(total_adj, 10.0), -10.0)
    if abs(total_adj) < 1e-6:
        return result

    spread = result["spread_home_minus_away"]
    base_total = result["total_points"] + total_adj
    total = max(20.0, base_total)
    home_points = (total + spread) / 2.0
    away_points = total - home_points

    adjusted = result.copy()
    adjusted["total_points"] = total
    adjusted["home_points"] = home_points
    adjusted["away_points"] = away_points
    adjusted["weather_total_adj"] = total_adj
    adjusted["weather_condition"] = condition
    adjusted["weather_temp"] = temp
    adjusted["weather_wind"] = wind
    if adjusted.get("market_total") is not None:
        adjusted["total_vs_market"] = adjusted["total_points"] - adjusted["market_total"]
    return adjusted


def build_ratings(teams: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "ppa_offense",
        "ppa_passing",
        "ppa_rushing",
        "ppa_defense",
        "ppa_def_pass",
        "ppa_def_rush",
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
        0.30 * teams.get("ppa_offense_z", 0.0)
        + 0.15 * teams.get("ppa_passing_z", 0.0)
        + 0.15 * teams.get("ppa_rushing_z", 0.0)
        + 0.20 * teams.get("sp_offense_z", 0.0)
        + 0.10 * teams.get("fpi_offense_z", 0.0)
        + 0.05 * teams.get("sp_special_z", 0.0)
        + 0.10 * teams.get("receiving_grade_route_z", 0.0)
        + 0.05 * teams.get("receiving_yprr_z", 0.0)
        + 0.08 * teams.get("blocking_grade_pass_z", 0.0)
        + 0.05 * teams.get("blocking_grade_run_z", 0.0)
        + 0.02 * teams.get("blocking_pbe_z", 0.0)
    )

    teams["defense_rating"] = (
        -0.35 * teams.get("ppa_defense_z", 0.0)
        - 0.20 * teams.get("ppa_def_pass_z", 0.0)
        - 0.15 * teams.get("ppa_def_rush_z", 0.0)
        - 0.20 * teams.get("sp_defense_z", 0.0)
        - 0.10 * teams.get("fpi_defense_z", 0.0)
        - 0.08 * teams.get("defense_grade_overall_z", 0.0)
        - 0.05 * teams.get("defense_grade_pass_rush_z", 0.0)
        - 0.05 * teams.get("defense_grade_coverage_z", 0.0)
        - 0.02 * teams.get("defense_grade_run_z", 0.0)
    )

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


def build_rating_book(
    year: int,
    *,
    api_key: Optional[str] = None,
    adjust_week: Optional[int] = None,
    calibration_games: Optional[Iterable[dict]] = None,
    constants: Optional[RatingConstants] = None,
) -> tuple[pd.DataFrame, RatingBook]:
    constants = constants or RatingConstants()
    teams_raw = fetch_team_metrics(year, api_key=api_key)
    ratings = build_ratings(teams_raw)
    adjustments: Dict[str, float] = {}
    if adjust_week is not None and adjust_week > 1:
        key = api_key or os.environ.get("CFBD_API_KEY")
        if not key:
            raise RuntimeError("CFBD API key required for opponent adjustments.")
        games = fetch_games(year, key)
        adjustments = compute_power_adjustments(
            ratings,
            games,
            constants=constants,
            up_to_week=adjust_week,
        )
    book = RatingBook(ratings, constants, power_adjustments=adjustments)
    if calibration_games:
        sigma = fit_probability_sigma(book, calibration_games)
        if sigma:
            book.prob_sigma = sigma
    return ratings, book


@dataclass
class RatingConstants:
    avg_total: float = 56.0
    home_field_advantage: float = 2.4
    offense_factor: float = 5.2
    defense_factor: float = 4.5
    power_factor: float = 1.2
    spread_sigma: float = 15.0

    @property
    def avg_team_points(self) -> float:
        return self.avg_total / 2.0


class RatingBook:
    def __init__(
        self,
        teams: pd.DataFrame,
        constants: RatingConstants,
        power_adjustments: Optional[Dict[str, float]] = None,
    ) -> None:
        self.teams = teams
        self.constants = constants
        self.power_adjustments = power_adjustments or {}
        # Calibration parameters derived from training data (Weeks 5-10, 2025).
        self.spread_calibration = (-0.283736, 1.018197)
        self.total_calibration = (-11.81207161, 1.16842142)
        self.prob_sigma = constants.spread_sigma

    def _lookup(self, team: str) -> pd.Series:
        df = self.teams
        mask = df["team"].str.lower() == team.lower()
        if mask.sum() == 1:
            return df.loc[mask].iloc[0]
        mask = df["team"].str.contains(team, case=False, regex=False)
        if mask.sum() == 1:
            return df.loc[mask].iloc[0]
        raise KeyError(f"Team '{team}' not found; available: {sorted(df['team'].unique())[:10]}...")

    def get_rating(self, team: str) -> Dict[str, float]:
        row = self._lookup(team)
        return {
            "team": row["team"],
            "offense_rating": row["offense_rating"],
            "defense_rating": row["defense_rating"],
            "power_rating": row["power_rating"] + self.power_adjustments.get(row["team"], 0.0),
        }

    def predict(self, home: str, away: str, neutral_site: bool = False) -> Dict[str, float]:
        home_row = self._lookup(home)
        away_row = self._lookup(away)
        const = self.constants
        home_power = home_row["power_rating"] + self.power_adjustments.get(home_row["team"], 0.0)
        away_power = away_row["power_rating"] + self.power_adjustments.get(away_row["team"], 0.0)

        def expected_points(off_row: pd.Series, def_row: pd.Series, power_diff: float) -> float:
            return (
                const.avg_team_points
                + const.offense_factor * off_row["offense_rating"]
                - const.defense_factor * def_row["defense_rating"]
                + const.power_factor * power_diff / 2.0
            )

        home_points = expected_points(home_row, away_row, home_power - away_power)
        away_points = expected_points(away_row, home_row, away_power - home_power)

        spread = home_points - away_points
        if not neutral_site:
            spread += const.home_field_advantage
            home_points += const.home_field_advantage / 2.0
            away_points -= const.home_field_advantage / 2.0

        total = home_points + away_points

        # Apply linear calibration to align predictions with historical margins/totals.
        a_s, b_s = self.spread_calibration
        a_t, b_t = self.total_calibration
        spread = a_s + b_s * spread
        total = a_t + b_t * total

        # Recompute implied team totals from calibrated spread/total.
        home_points = (total + spread) / 2.0
        away_points = total - home_points
        win_prob = 0.5 * (1.0 + math.erf(spread / (self.prob_sigma * math.sqrt(2))))

        home_win_prob = win_prob
        away_win_prob = 1.0 - win_prob

        return {
            "home_team": home_row["team"],
            "away_team": away_row["team"],
            "home_points": home_points,
            "away_points": away_points,
            "spread_home_minus_away": spread,
            "total_points": total,
            "home_win_prob": home_win_prob,
            "away_win_prob": away_win_prob,
            "home_moneyline": _moneyline_from_prob(home_win_prob),
            "away_moneyline": _moneyline_from_prob(away_win_prob),
        }


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
