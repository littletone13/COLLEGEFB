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
from typing import Dict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import pandas as pd
import numpy as np
import requests

import ncaa_stats

DATA_DIR_DEFAULT = Path("~/Desktop/PFFMODEL_FBS/FCS_DATA").expanduser()
ADJUSTED_DATA_GLOB = "fcs_adjusted*/fcs_adjusted_{kind}_*.csv"
CFBD_BASE_URL = "https://api.collegefootballdata.com"

FCS_MARKET_SPREAD_WEIGHT = 0.7
FCS_MARKET_TOTAL_WEIGHT = 0.7
PFF_COMBINED_DATA_DIR = Path("~/Desktop/POST WEEK 9 FBS & FCS DATA").expanduser()

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
    if not market:
        result.setdefault("market_spread", None)
        result.setdefault("market_total", None)
        result.setdefault("market_providers", [])
        result.setdefault("market_provider_count", 0)
        result.setdefault("market_provider_lines", {})
        result.setdefault("spread_vs_market", None)
        result.setdefault("total_vs_market", None)
        return result

    updated = result.copy()
    spread = updated.get("spread_team_one_minus_team_two")
    total = updated.get("total_points")
    if spread is None or total is None:
        return updated

    market_spread = market.get("spread")
    market_total = market.get("total")
    provider_lines = market.get("provider_lines") or {}
    providers = sorted(provider_lines) if provider_lines else (market.get("providers") or [])

    if market_spread is not None:
        w = min(max(spread_weight, 0.0), 1.0)
        spread = (1.0 - w) * spread + w * market_spread
    if market_total is not None:
        w = min(max(total_weight, 0.0), 1.0)
        total = (1.0 - w) * total + w * market_total

    home_points = (total + spread) / 2.0
    away_points = total - home_points
    win_prob = 0.5 * (1.0 + math.erf(spread / (FCS_PROB_SIGMA * math.sqrt(2))))
    win_prob = min(max(win_prob, 0.0), 1.0)

    updated["spread_team_one_minus_team_two"] = spread
    updated["total_points"] = max(20.0, total)
    updated["team_one_points"] = home_points
    updated["team_two_points"] = away_points
    updated["team_one_win_prob"] = win_prob
    updated["team_two_win_prob"] = 1.0 - win_prob
    updated["team_one_moneyline"] = RatingBook._prob_to_moneyline(win_prob)
    updated["team_two_moneyline"] = RatingBook._prob_to_moneyline(1.0 - win_prob)
    updated["market_spread"] = market_spread
    updated["market_total"] = market_total
    updated["market_provider_lines"] = provider_lines
    updated["market_providers"] = providers
    updated["market_provider_count"] = len(providers)
    updated["spread_vs_market"] = (spread - market_spread) if market_spread is not None else None
    updated["total_vs_market"] = (total - market_total) if market_total is not None else None
    return updated

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

@dataclass
class RatingConstants:
    avg_total: float = 52.7442
    home_field_advantage: float = 4.9105
    offense_factor: float = 0.7333
    defense_factor: float = 0.2318
    special_teams_factor: float = 1.4226
    spread_sigma: float = 16.0

    @property
    def avg_team_points(self) -> float:
        return self.avg_total / 2.0


@dataclass
class TeamRatings:
    team_name: str
    offense_rating: float
    defense_rating: float
    special_teams_rating: float
    power_rating: float
    spread_rating: float


class RatingBook:
    def __init__(
        self,
        teams: pd.DataFrame,
        constants: RatingConstants,
        *,
        spread_calibration: Optional[tuple[float, float]] = None,
        total_calibration: Optional[tuple[float, float]] = None,
        prob_sigma: Optional[float] = None,
        spread_intercept: Optional[float] = None,
        spread_coeffs: Optional[Dict[str, float]] = None,
    ):
        self.teams = teams
        self.constants = constants
        self.spread_calibration = spread_calibration or FCS_SPREAD_CALIBRATION
        self.total_calibration = total_calibration or FCS_TOTAL_CALIBRATION
        self.prob_sigma = prob_sigma or FCS_PROB_SIGMA
        self.spread_intercept = spread_intercept if spread_intercept is not None else FCS_SPREAD_REG_INTERCEPT
        self.spread_coeffs = spread_coeffs or FCS_SPREAD_LINEAR_COEFFS

    def get(self, team: str) -> TeamRatings:
        row = self._find_row(team)
        return TeamRatings(
            team_name=row["team_name"],
            offense_rating=float(row["offense_rating"]),
            defense_rating=float(row["defense_rating"]),
            special_teams_rating=float(row["special_rating"]),
            power_rating=float(row["power_rating"]),
            spread_rating=float(row.get("spread_rating", row["power_rating"])),
        )

    def _find_row(self, team: str) -> pd.Series:
        df = self.teams
        # Case-insensitive lookup with defensive fallbacks.
        mask = df["team_name"].str.lower() == team.lower()
        if mask.sum() == 1:
            return df.loc[mask].iloc[0]
        # Allow partial matches when unique.
        mask = df["team_name"].str.contains(team, case=False, regex=False)
        if mask.sum() == 1:
            return df.loc[mask].iloc[0]
        available = ", ".join(sorted(df["team_name"].unique()))
        raise KeyError(f"Team '{team}' not found. Available teams: {available}")

    def predict(
        self,
        team_one: str,
        team_two: str,
        *,
        neutral_site: bool = False,
    ) -> Dict[str, float]:
        a = self.get(team_one)
        b = self.get(team_two)
        const = self.constants

        # Expected points for each side.
        a_points = self._expected_points(a, b)
        b_points = self._expected_points(b, a)

        spread_rating_home = getattr(a, "spread_rating", None)
        spread_rating_away = getattr(b, "spread_rating", None)
        if spread_rating_home is not None and spread_rating_away is not None:
            coeffs = self.spread_coeffs or {}
            spread_diff = spread_rating_home - spread_rating_away
            power_diff = a.power_rating - b.power_rating
            offense_diff = a.offense_rating - b.offense_rating
            defense_diff = b.defense_rating - a.defense_rating
            special_diff = a.special_teams_rating - b.special_teams_rating
            sr_power = spread_diff * power_diff
            off_def = offense_diff * defense_diff
            power_special = power_diff * special_diff
            power_sq = power_diff * power_diff
            spread_raw = (
                coeffs.get("intercept", self.spread_intercept)
                + coeffs.get("spread_rating_diff", 1.0) * spread_diff
                + coeffs.get("power_diff", 0.0) * power_diff
                + coeffs.get("offense_diff", 0.0) * offense_diff
                + coeffs.get("defense_diff", 0.0) * defense_diff
                + coeffs.get("special_diff", 0.0) * special_diff
                + coeffs.get("sr_x_power", 0.0) * sr_power
                + coeffs.get("off_x_def", 0.0) * off_def
                + coeffs.get("power_x_special", 0.0) * power_special
                + coeffs.get("power_sq", 0.0) * power_sq
            )
        else:
            spread_raw = a_points - b_points
        if not neutral_site:
            spread_raw += const.home_field_advantage
            a_points += const.home_field_advantage / 2.0
            b_points -= const.home_field_advantage / 2.0

        total_raw = a_points + b_points

        spread = self.spread_calibration[0] + self.spread_calibration[1] * spread_raw
        total = self.total_calibration[0] + self.total_calibration[1] * total_raw
        total = max(20.0, total)

        a_points = (total + spread) / 2.0
        b_points = total - a_points

        win_prob_a = self._spread_to_win_prob(spread)
        win_prob_b = 1.0 - win_prob_a

        moneyline_a = self._prob_to_moneyline(win_prob_a)
        moneyline_b = self._prob_to_moneyline(win_prob_b)

        return {
            "team_one": a.team_name,
            "team_two": b.team_name,
            "team_one_points": a_points,
            "team_two_points": b_points,
            "spread_team_one_minus_team_two": spread,
            "total_points": total,
            "team_one_win_prob": win_prob_a,
            "team_two_win_prob": win_prob_b,
            "team_one_moneyline": moneyline_a,
            "team_two_moneyline": moneyline_b,
        }

    def _expected_points(self, offense: TeamRatings, defense: TeamRatings) -> float:
        const = self.constants
        return (
            const.avg_team_points
            + const.offense_factor * offense.offense_rating
            - const.defense_factor * defense.defense_rating
            + const.special_teams_factor * (offense.special_teams_rating - defense.special_teams_rating) / 2.0
        )

    def _spread_to_win_prob(self, spread: float) -> float:
        sigma = self.prob_sigma
        # Convert spread edge into a win probability using a normal model.
        z = spread / (sigma * 2 ** 0.5)
        return 0.5 * (1.0 + math.erf(z))

    @staticmethod
    def _prob_to_moneyline(prob: float) -> Optional[float]:
        if prob <= 0.0 or prob >= 1.0:
            return None
        if prob >= 0.5:
            return -100.0 * prob / (1.0 - prob)
        return 100.0 * (1.0 - prob) / prob


# --- Data loading ---------------------------------------------------------

def load_team_ratings(
    data_dir: Path = DATA_DIR_DEFAULT,
    *,
    season_year: Optional[int] = None,
) -> pd.DataFrame:
    data_dir = data_dir.expanduser()
    receiving = _read_first_available(
        data_dir,
        (
            "receiving_summary*FBS*FCS*.csv",
            "receiving_summary_FCS.csv",
            "receiving_summary_FBS.csv",
            "receiving_summary.csv",
        ),
    )
    blocking = _read_first_available(
        data_dir,
        (
            "offense_blocking*FBS*FCS*.csv",
            "offense_blocking_FCS.csv",
            "offense_blocking FBS.csv",
            "offense_blocking.csv",
        ),
    )
    defense = _read_first_available(
        data_dir,
        (
            "defense_summary*FBS*FCS*.csv",
            "defense_summary_FCS.csv",
            "defense_summary.csv",
            "defense_summary FBS.csv",
            "defense_summary_FBS.csv",
        ),
    )
    special = _read_first_available(
        data_dir,
        (
            "special_teams_summary*FBS*FCS*.csv",
            "special_teams_summary_FCS.csv",
            "special_teams_summary_FBS.csv",
            "special_teams_summary.csv",
        ),
    )

    adj_offense, adj_defense = _load_adjusted_metrics(season_year)
    adj_offense = adj_offense.rename(columns={
        next((c for c in adj_offense.columns if c.startswith("points")), "points_offense"): "adj_points_for",
        next((c for c in adj_offense.columns if c.startswith("totalYards")), "totalYards_offense"): "adj_total_yards_for",
        next((c for c in adj_offense.columns if c.startswith("netPassingYards")), "netPassingYards_offense"): "adj_net_passing_for",
        next((c for c in adj_offense.columns if c.startswith("rushingYards")), "rushingYards_offense"): "adj_rushing_for",
        next((c for c in adj_offense.columns if c.startswith("yardsPerPass")), "yardsPerPass_offense"): "adj_yppass_for",
        next((c for c in adj_offense.columns if c.startswith("yardsPerRushAttempt")), "yardsPerRushAttempt_offense"): "adj_yprush_for",
        next((c for c in adj_offense.columns if c.startswith("turnovers")), "turnovers_offense"): "adj_turnovers_for",
        next((c for c in adj_offense.columns if c.startswith("thirdDownEff")), "thirdDownEff_offense"): "adj_third_down_for",
    }) if not adj_offense.empty else adj_offense
    if not adj_offense.empty and "team" not in adj_offense.columns:
        adj_offense = adj_offense.rename(columns={adj_offense.columns[1]: "team"})

    adj_defense = adj_defense.rename(columns={
        next((c for c in adj_defense.columns if c.startswith("points")), "points_defense"): "adj_points_against",
        next((c for c in adj_defense.columns if c.startswith("totalYards")), "totalYards_defense"): "adj_total_yards_against",
        next((c for c in adj_defense.columns if c.startswith("netPassingYards")), "netPassingYards_defense"): "adj_net_passing_against",
        next((c for c in adj_defense.columns if c.startswith("rushingYards")), "rushingYards_defense"): "adj_rushing_against",
        next((c for c in adj_defense.columns if c.startswith("yardsPerPass")), "yardsPerPass_defense"): "adj_yppass_against",
        next((c for c in adj_defense.columns if c.startswith("yardsPerRushAttempt")), "yardsPerRushAttempt_defense"): "adj_yprush_against",
        next((c for c in adj_defense.columns if c.startswith("turnovers")), "turnovers_defense"): "adj_turnovers_against",
        next((c for c in adj_defense.columns if c.startswith("thirdDownEff")), "thirdDownEff_defense"): "adj_third_down_against",
    }) if not adj_defense.empty else adj_defense
    if not adj_defense.empty and "team" not in adj_defense.columns:
        adj_defense = adj_defense.rename(columns={adj_defense.columns[1]: "team"})

    receiving_teams = aggregate_team_metrics(
        receiving,
        team_col="team_name",
        weight_col="routes" if "routes" in receiving.columns else None,
        weighted_metrics={
            "receiving_grade_offense": "grades_offense",
            "receiving_grade_route": "grades_pass_route",
            "receiving_grade_pass_block": "grades_pass_block",
            "receiving_target_qb_rating": "targeted_qb_rating",
            "receiving_yprr": "yprr",
            "receiving_catch_rate": "caught_percent",
        },
        sum_metrics={
            "receiving_targets_total": "targets",
            "receiving_yards_total": "yards",
            "receiving_routes_total": "routes" if "routes" in receiving.columns else "targets",
        },
    )

    blocking_teams = aggregate_team_metrics(
        blocking,
        team_col="team_name",
        weight_col="snap_counts_offense" if "snap_counts_offense" in blocking.columns else None,
        weighted_metrics={
            "blocking_grade_offense": "grades_offense",
            "blocking_grade_pass": "grades_pass_block",
            "blocking_grade_run": "grades_run_block",
            "blocking_pbe": "pbe",
        },
        sum_metrics={
            "pressures_allowed_total": "pressures_allowed",
            "sacks_allowed_total": "sacks_allowed",
        },
    )

    defense_teams = aggregate_team_metrics(
        defense,
        team_col="team_name",
        weight_col="snap_counts_defense" if "snap_counts_defense" in defense.columns else None,
        weighted_metrics={
            "defense_grade_overall": "grades_defense",
            "defense_grade_coverage": "grades_coverage_defense",
            "defense_grade_run": "grades_run_defense",
            "defense_grade_pass_rush": "grades_pass_rush_defense",
            "defense_missed_tackle_rate": "missed_tackle_rate",
            "defense_qb_rating_against": "qb_rating_against",
        },
        sum_metrics={
            "defense_turnovers": "interceptions",
            "defense_sacks": "sacks",
            "defense_pressures": "total_pressures",
        },
    )

    special_teams = aggregate_team_metrics(
        special,
        team_col="team_name",
        weight_col=None,
        weighted_metrics={
            "special_grade_misc": "grades_misc_st",
            "special_grade_return": "grades_kick_return",
            "special_grade_punt_return": "grades_punt_return",
            "special_grade_kickoff": "grades_kickoff_kicker",
            "special_grade_fg_offense": "grades_fgep_offense",
        },
        sum_metrics={
            "special_tackles": "tackles",
        },
    )

    teams = (
        receiving_teams
        .join(blocking_teams, how="outer")
        .join(defense_teams, how="outer")
        .join(special_teams, how="outer")
        .reset_index()
    )

    if not adj_offense.empty and "team" in adj_offense.columns:
        off_cols = [c for c in adj_offense.columns if c.startswith("adj_")]
        off_df = adj_offense[["team", *off_cols]].drop_duplicates(subset="team")
        teams = teams.merge(off_df.rename(columns={"team": "team_name"}), on="team_name", how="left")

    if not adj_defense.empty and "team" in adj_defense.columns:
        def_cols = [c for c in adj_defense.columns if c.startswith("adj_")]
        def_df = adj_defense[["team", *def_cols]].drop_duplicates(subset="team")
        teams = teams.merge(def_df.rename(columns={"team": "team_name"}), on="team_name", how="left")

    # Append NCAA advanced team metrics when available.
    try:
        ncaa_features = ncaa_stats.build_team_feature_frame(season_year=season_year)
    except Exception as exc:  # pragma: no cover - network failure resilience
        warnings.warn(f"NCAA stats fetch failed: {exc}")
        ncaa_features = pd.DataFrame(columns=["team_name"])

    if not ncaa_features.empty:
        existing = set(teams["team_name"])
        additional = sorted(set(ncaa_features["team_name"]) - existing)
        if additional:
            filler = pd.DataFrame(index=range(len(additional)))
            for col in teams.columns:
                if col == "team_name":
                    filler[col] = additional
                else:
                    filler[col] = np.nan
            teams = pd.concat([teams, filler], ignore_index=True)
        merge_cols = [col for col in ncaa_features.columns if col not in teams.columns or col == "team_name"]
        teams = teams.merge(ncaa_features[merge_cols], on="team_name", how="left")

    # Fill numeric NaNs with column means for stability.
    teams = teams.copy()
    numeric_cols = teams.select_dtypes(include="number").columns
    teams[numeric_cols] = teams[numeric_cols].apply(lambda col: col.fillna(col.mean()))

    # Build z-scores.
    def add_z(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
        valid_cols = [col for col in cols if col in df.columns]
        if not valid_cols:
            return df
        sub = df[valid_cols].apply(pd.to_numeric, errors="coerce")
        means = sub.mean()
        stds = sub.std(ddof=0)
        stds = stds.replace(0.0, np.nan)
        z_scores = (sub - means).divide(stds).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        z_scores.columns = [f"{col}_z" for col in valid_cols]
        return pd.concat([df, z_scores], axis=1)

    teams = add_z(teams, [
        "receiving_grade_offense",
        "receiving_grade_route",
        "receiving_grade_pass_block",
        "receiving_target_qb_rating",
        "receiving_yprr",
        "blocking_grade_offense",
        "blocking_grade_pass",
        "blocking_grade_run",
        "blocking_pbe",
        "defense_grade_overall",
        "defense_grade_coverage",
        "defense_grade_run",
        "defense_grade_pass_rush",
        "defense_missed_tackle_rate",
        "defense_qb_rating_against",
        "special_grade_misc",
        "special_grade_return",
        "special_grade_punt_return",
        "special_grade_kickoff",
        "special_grade_fg_offense",
        "plays_per_game",
        "offense_ypp",
        "offense_ypg",
        "third_down_pct",
        "team_pass_eff",
        "points_per_game",
        "avg_time_of_possession",
        "defense_ypp_allowed",
        "defense_ypg_allowed",
        "third_down_def_pct",
        "opp_pass_eff",
        "points_allowed_per_game",
        "qb_pass_eff",
        "rb_rush_ypg",
        "wr_rec_yards",
        "offense_ypp_adj",
        "points_per_game_adj",
        "team_pass_eff_adj",
        "defense_ypp_allowed_adj",
        "points_allowed_per_game_adj",
        "red_zone_pct",
        "red_zone_def_pct",
        "red_zone_attempts",
        "red_zone_def_attempts",
        "turnover_gain",
        "turnover_loss",
        "turnover_margin_total",
        "turnover_margin_avg",
        "penalties_per_game",
        "penalty_yards_per_game",
        "rush_yards_per_game",
        "rush_yards_allowed_per_game",
        "pass_yards_per_game",
        "pass_yards_allowed_per_game",
        "sacks_per_game",
        "tfl_per_game",
        "sacks_allowed_per_game",
        "tfl_allowed_per_game",
        "kick_return_avg",
        "kick_return_defense_avg",
        "punt_return_avg",
        "punt_return_defense_avg",
        "net_punting_avg",
        "adj_points_for",
        "adj_total_yards_for",
        "adj_net_passing_for",
        "adj_rushing_for",
        "adj_yppass_for",
        "adj_yprush_for",
        "adj_turnovers_for",
        "adj_third_down_for",
        "adj_points_against",
        "adj_total_yards_against",
        "adj_net_passing_against",
        "adj_rushing_against",
        "adj_yppass_against",
        "adj_yprush_against",
        "adj_turnovers_against",
        "adj_third_down_against",
    ])

    spread_components: list[np.ndarray] = []
    for col, weight in FCS_SPREAD_REG_WEIGHTS.items():
        if col in teams.columns:
            spread_components.append(
                weight * teams[col].fillna(0.0).to_numpy()
            )
    spread_vals = np.sum(spread_components, axis=0) if spread_components else np.zeros(len(teams), dtype=float)

    offense_rating = (
        0.15 * teams.get("receiving_grade_route_z", 0.0)
        + 0.14 * teams.get("receiving_grade_offense_z", 0.0)
        + 0.08 * teams.get("receiving_yprr_z", 0.0)
        + 0.16 * teams.get("blocking_grade_pass_z", 0.0)
        + 0.10 * teams.get("blocking_grade_run_z", 0.0)
        + 0.08 * teams.get("adj_points_for_z", 0.0)
        + 0.08 * teams.get("adj_total_yards_for_z", 0.0)
        + 0.07 * teams.get("adj_yppass_for_z", 0.0)
        + 0.07 * teams.get("adj_yprush_for_z", 0.0)
        + 0.05 * teams.get("adj_third_down_for_z", 0.0)
        - 0.05 * teams.get("adj_turnovers_for_z", 0.0)
        + 0.05 * teams.get("offense_ypp_z", 0.0)
        + 0.04 * teams.get("third_down_pct_z", 0.0)
        + 0.04 * teams.get("team_pass_eff_z", 0.0)
        + 0.04 * teams.get("points_per_game_z", 0.0)
        + 0.04 * teams.get("rush_yards_per_game_z", 0.0)
        + 0.04 * teams.get("pass_yards_per_game_z", 0.0)
        + 0.03 * teams.get("red_zone_pct_z", 0.0)
        + 0.03 * teams.get("offense_ypp_adj_z", 0.0)
        + 0.03 * teams.get("points_per_game_adj_z", 0.0)
        + 0.03 * teams.get("team_pass_eff_adj_z", 0.0)
        + 0.03 * teams.get("turnover_margin_avg_z", 0.0)
        - 0.04 * teams.get("penalties_per_game_z", 0.0)
        - 0.04 * teams.get("sacks_allowed_per_game_z", 0.0)
        - 0.03 * teams.get("tfl_allowed_per_game_z", 0.0)
        + 0.03 * teams.get("qb_pass_eff_z", 0.0)
        + 0.02 * teams.get("rb_rush_ypg_z", 0.0)
        + 0.02 * teams.get("wr_rec_yards_z", 0.0)
    )

    defense_rating = (
        0.24 * teams.get("defense_grade_overall_z", 0.0)
        + 0.20 * teams.get("defense_grade_coverage_z", 0.0)
        + 0.18 * teams.get("defense_grade_run_z", 0.0)
        + 0.14 * teams.get("defense_grade_pass_rush_z", 0.0)
        - 0.12 * teams.get("adj_points_against_z", 0.0)
        - 0.12 * teams.get("adj_total_yards_against_z", 0.0)
        - 0.08 * teams.get("adj_yppass_against_z", 0.0)
        - 0.08 * teams.get("adj_yprush_against_z", 0.0)
        - 0.06 * teams.get("adj_third_down_against_z", 0.0)
        + 0.06 * teams.get("adj_turnovers_against_z", 0.0)
        - 0.08 * teams.get("defense_qb_rating_against_z", 0.0)
        - 0.06 * teams.get("defense_missed_tackle_rate_z", 0.0)
        - 0.08 * teams.get("defense_ypp_allowed_z", 0.0)
        - 0.08 * teams.get("defense_ypg_allowed_z", 0.0)
        - 0.08 * teams.get("rush_yards_allowed_per_game_z", 0.0)
        - 0.08 * teams.get("pass_yards_allowed_per_game_z", 0.0)
        - 0.06 * teams.get("third_down_def_pct_z", 0.0)
        - 0.06 * teams.get("opp_pass_eff_z", 0.0)
        - 0.06 * teams.get("points_allowed_per_game_z", 0.0)
        - 0.05 * teams.get("red_zone_def_pct_z", 0.0)
        - 0.05 * teams.get("defense_ypp_allowed_adj_z", 0.0)
        - 0.05 * teams.get("points_allowed_per_game_adj_z", 0.0)
        + 0.04 * teams.get("sacks_per_game_z", 0.0)
        + 0.04 * teams.get("tfl_per_game_z", 0.0)
        + 0.04 * teams.get("turnover_gain_z", 0.0)
    )

    special_rating = (
        0.40 * teams.get("special_grade_misc_z", 0.0)
        + 0.18 * teams.get("special_grade_return_z", 0.0)
        + 0.12 * teams.get("special_grade_punt_return_z", 0.0)
        + 0.10 * teams.get("special_grade_kickoff_z", 0.0)
        + 0.05 * teams.get("special_grade_fg_offense_z", 0.0)
        + 0.05 * teams.get("kick_return_avg_z", 0.0)
        - 0.05 * teams.get("kick_return_defense_avg_z", 0.0)
        + 0.05 * teams.get("punt_return_avg_z", 0.0)
        - 0.05 * teams.get("punt_return_defense_avg_z", 0.0)
        + 0.05 * teams.get("net_punting_avg_z", 0.0)
    )

    power_rating = offense_rating + defense_rating + 0.2 * special_rating

    teams = teams.assign(
        spread_rating=spread_vals,
        offense_rating=offense_rating,
        defense_rating=defense_rating,
        special_rating=special_rating,
        power_rating=power_rating,
    )

    return teams


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
    teams = load_team_ratings(args.data_dir, season_year=args.season_year)
    book = RatingBook(teams, RatingConstants())

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
