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
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import pandas as pd
import numpy as np
import requests

import ncaa_stats
from cfb import market as market_utils
from cfb.config import load_config
from cfb.model import FCSRatingBook, FCSRatingConstants
from cfb import market as market_utils
from cfb.config import load_config

DATA_DIR_DEFAULT = Path("~/Desktop/PFFMODEL_FBS/FCS_DATA").expanduser()
ADJUSTED_DATA_GLOB = "fcs_adjusted*/fcs_adjusted_{kind}_*.csv"
CFBD_BASE_URL = "https://api.collegefootballdata.com"

CONFIG = load_config()
FCS_CONFIG = CONFIG.get("fcs", {}) if isinstance(CONFIG.get("fcs"), dict) else {}
_MARKET_CONFIG = FCS_CONFIG.get("market", {}) if isinstance(FCS_CONFIG.get("market"), dict) else {}

_RATING_CONFIG = FCS_CONFIG.get("ratings", {}) if isinstance(FCS_CONFIG.get("ratings"), dict) else {}

FCS_MARKET_SPREAD_WEIGHT = float(_MARKET_CONFIG.get("spread_weight", 0.7))
FCS_MARKET_TOTAL_WEIGHT = float(_MARKET_CONFIG.get("total_weight", 0.7))
PFF_COMBINED_DATA_DIR = Path("~/Desktop/POST WEEK 9 FBS & FCS DATA").expanduser()


def _rating_constants_from_config() -> FCSRatingConstants:
    return FCSRatingConstants(
        avg_total=float(_RATING_CONFIG.get("avg_total", 52.7442)),
        home_field_advantage=float(_RATING_CONFIG.get("home_field_advantage", 4.9105)),
        offense_factor=float(_RATING_CONFIG.get("offense_factor", 0.7333)),
        defense_factor=float(_RATING_CONFIG.get("defense_factor", 0.2318)),
        special_teams_factor=float(_RATING_CONFIG.get("special_teams_factor", 1.4226)),
        spread_sigma=float(_RATING_CONFIG.get("spread_sigma", 16.0)),
    )

FCS_SPREAD_CALIBRATION = (0.01194, 0.99456)
FCS_TOTAL_CALIBRATION = (0.0, 1.0)
FCS_PROB_SIGMA = 16.0
FCS_SPREAD_REG_INTERCEPT = 5.013813339075651
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
FCS_SPREAD_LINEAR_COEFFS = {
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


def _read_first_available(data_dir: Path, candidates: Iterable[str]) -> pd.DataFrame:
    directories = []
    combined = PFF_COMBINED_DATA_DIR.expanduser()
    if combined.exists():
        directories.append(combined)
    directories.append(data_dir.expanduser())
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
    provider_filter: set[str] = set()
    if providers:
        provider_filter.update(str(name).strip().lower() for name in providers if str(name).strip())
    if provider:
        provider_filter.add(str(provider).strip().lower())

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
    for entry in resp.json():
        if entry.get("homeClassification") != "fcs" or entry.get("awayClassification") != "fcs":
            continue
        lines = entry.get("lines") or []
        provider_names: set[str] = set()
        provider_lines: Dict[str, dict] = {}
        for line in lines:
            prov_raw = line.get("provider")
            provider_name = str(prov_raw or "").strip()
            if not provider_name:
                continue
            if provider_filter and provider_name.lower() not in provider_filter:
                continue

            spread_raw = _coerce_float(line.get("spread"))
            total_raw = _coerce_float(line.get("overUnder"))
            home_ml = _coerce_float(line.get("homeMoneyline"))
            away_ml = _coerce_float(line.get("awayMoneyline"))
            last_updated = line.get("lastUpdated") or line.get("updated")

            provider_names.add(provider_name)
            provider_lines[provider_name] = {
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

        records.append(
            {
                "game_id": entry.get("id"),
                "home_team": entry.get("homeTeam"),
                "away_team": entry.get("awayTeam"),
                "start_date": entry.get("startDate"),
                "spread": float(np.mean(spread_values)) if spread_values else None,
                "total": float(np.mean(total_values)) if total_values else None,
                "providers": sorted(provider_names),
                "provider_lines": provider_lines,
            }
        )
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


# --- Rating construction --------------------------------------------------


def build_rating_book(
    data_dir: Path = DATA_DIR_DEFAULT,
    *,
    season_year: Optional[int] = None,
    spread_calibration: tuple[float, float] = FCS_SPREAD_CALIBRATION,
    total_calibration: tuple[float, float] = FCS_TOTAL_CALIBRATION,
    prob_sigma: float = FCS_PROB_SIGMA,
) -> tuple[pd.DataFrame, FCSRatingBook]:
    teams = load_team_ratings(data_dir, season_year=season_year)
    spread_model = dict(FCS_SPREAD_LINEAR_COEFFS)
    if "intercept" not in spread_model:
        spread_model["intercept"] = FCS_SPREAD_REG_INTERCEPT
    book = FCSRatingBook(
        teams,
        _rating_constants_from_config(),
        spread_calibration=spread_calibration,
        total_calibration=total_calibration,
        prob_sigma=prob_sigma,
        spread_model=spread_model,
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
