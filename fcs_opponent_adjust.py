"""Compute opponent-adjusted FCS team metrics using ridge regression.

This module fetches per-game team statistics from the CFBD ``/games/teams``
endpoint (classification ``fcs``) and applies the ridge-regression opponent
adjustment methodology described by CFBD (see
``https://blog.collegefootballdata.com/opponent-adjusted-stats-ridge-regression/``).

Two helper entry points are provided:

* ``collect_fcs_game_stats`` – download and assemble per-game stats for a
  target season.
* ``compute_opponent_adjustments`` – perform ridge regression for a
  rectangular set of stats (offense vs defense) and return the adjusted
  ratings for each team.

The script can also be executed directly to generate season-level outputs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge, RidgeCV


CFBD_API = "https://api.collegefootballdata.com"


# Default stat categories pulled from the CFBD /games/teams endpoint. These are
# all numeric fields that exist for FCS contests.
DEFAULT_STATS = (
    "points",
    "totalYards",
    "netPassingYards",
    "rushingYards",
    "yardsPerPass",
    "yardsPerRushAttempt",
    "turnovers",
    "thirdDownEff",
)


@dataclass
class AdjustmentResult:
    """Container for a single stat's ridge regression output."""

    stat: str
    alpha: float
    homefield: float
    offense: pd.DataFrame
    defense: pd.DataFrame


def _ensure_api_key(value: Optional[str]) -> str:
    if value:
        return value
    from os import environ

    key = environ.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError("CFBD API key required (set CFBD_API_KEY or pass --api-key).")
    return key


def collect_fcs_game_stats(
    year: int,
    *,
    api_key: Optional[str] = None,
    max_week: Optional[int] = None,
    season_type: str = "regular",
    stats: Iterable[str] = DEFAULT_STATS,
) -> pd.DataFrame:
    """Fetch per-game stats for FCS matchups.

    Returns a dataframe with columns ``offense``, ``defense``, ``hfa`` and one
    column per requested stat.
    """

    api_key = _ensure_api_key(api_key)
    headers = {"Authorization": f"Bearer {api_key}"}
    stat_set = set(stats)
    # Restrict to true FCS vs FCS games.
    teams_resp = requests.get(
        f"{CFBD_API}/teams",
        headers=headers,
        params={"classification": "fcs"},
        timeout=60,
    )
    teams_resp.raise_for_status()
    fcs_team_names = {
        entry["school"]
        for entry in teams_resp.json()
        if entry.get("classification") == "fcs"
    }

    records: list[dict] = []

    def process_payload(payload, *, week_label: Optional[int] = None) -> None:
        for game in payload:
            teams = game.get("teams") or []
            if len(teams) != 2:
                continue
            home_team = next((team for team in teams if team.get("homeAway") == "home"), None)
            away_team = next((team for team in teams if team.get("homeAway") == "away"), None)
            if not home_team or not away_team:
                continue
            if home_team.get("team") not in fcs_team_names or away_team.get("team") not in fcs_team_names:
                continue
            for offense_entry, defense_entry, hfa in (
                (home_team, away_team, 1.0),
                (away_team, home_team, -1.0),
            ):
                stat_map = {item["category"]: item.get("stat") for item in offense_entry.get("stats", [])}
                row: dict[str, float | str] = {
                    "week": game.get("week") if week_label is None else week_label,
                    "offense": offense_entry.get("team"),
                    "defense": defense_entry.get("team"),
                    "hfa": hfa,
                }
                row["points"] = float(offense_entry.get("points") or 0.0)
                for category in stat_set:
                    if category == "points":
                        continue
                    value = stat_map.get(category)
                    if value is None or value == "":
                        row[category] = np.nan
                        continue
                    if category == "thirdDownEff":
                        made, _, attempted = value.partition("-")
                        try:
                            made_f = float(made)
                            attempted_f = float(attempted) if attempted else np.nan
                            row[category] = np.nan if not attempted_f else made_f / attempted_f
                        except ValueError:
                            row[category] = np.nan
                        continue
                    if category == "completionAttempts":
                        made, _, attempted = value.partition("-")
                        try:
                            row[category] = float(attempted)
                        except ValueError:
                            row[category] = np.nan
                        continue
                    try:
                        row[category] = float(value)
                    except ValueError:
                        row[category] = np.nan
                records.append(row)

    # Attempt to fetch the entire dataset in one call when max_week is not set
    if max_week is None:
        params = {"year": year, "seasonType": season_type, "classification": "fcs"}
        try:
            resp = requests.get(f"{CFBD_API}/games/teams", headers=headers, params=params, timeout=120)
            if resp.status_code != 404:
                resp.raise_for_status()
                payload = resp.json()
                if payload:
                    process_payload(payload)
                    df = pd.DataFrame(records)
                    if not df.empty:
                        return df
        except requests.HTTPError:
            records.clear()

    # Otherwise fetch week-by-week (or when the single call returned nothing)
    week = 1
    while True:
        if max_week is not None and week > max_week:
            break
        params = {
            "year": year,
            "week": week,
            "seasonType": season_type,
            "classification": "fcs",
        }
        resp = requests.get(f"{CFBD_API}/games/teams", headers=headers, params=params, timeout=60)
        if resp.status_code == 404:
            break
        resp.raise_for_status()
        payload = resp.json()
        if not payload:
            if week == 1:
                break
            week += 1
            continue
        process_payload(payload)
        week += 1

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No FCS game data found for season {year} (max_week={max_week}).")
    return df


def _ridge_adjust_single_stat(
    df: pd.DataFrame,
    stat: str,
    alphas: Iterable[float],
) -> AdjustmentResult:
    df_valid = df.dropna(subset=[stat]).copy()
    if df_valid.empty:
        raise RuntimeError(f"No valid rows for stat '{stat}'.")

    X = pd.get_dummies(df_valid[["offense", "hfa", "defense"]], dtype=float)
    y = df_valid[stat].astype(float)

    ridge_cv = RidgeCV(alphas=list(alphas), fit_intercept=True, scoring="neg_mean_squared_error")
    ridge_cv.fit(X, y)
    alpha = float(ridge_cv.alpha_)

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)

    coef_df = pd.DataFrame({"name": X.columns, "coef": model.coef_})
    coef_df["value"] = coef_df["coef"] + model.intercept_

    offense = (
        coef_df[coef_df["name"].str.startswith("offense_")]
        .assign(team=lambda d: d["name"].str.replace("offense_", "", regex=False))
        .rename(columns={"value": stat})
        .drop(columns=["name", "coef"])
        .reset_index(drop=True)
    )
    defense = (
        coef_df[coef_df["name"].str.startswith("defense_")]
        .assign(team=lambda d: d["name"].str.replace("defense_", "", regex=False))
        .rename(columns={"value": stat})
        .drop(columns=["name", "coef"])
        .reset_index(drop=True)
    )
    hfa_coef = coef_df[coef_df["name"].str.startswith("hfa_")]
    homefield = float(hfa_coef["coef"].sum()) if not hfa_coef.empty else 0.0

    return AdjustmentResult(stat=stat, alpha=alpha, homefield=homefield, offense=offense, defense=defense)


def compute_opponent_adjustments(
    df: pd.DataFrame,
    stats: Iterable[str],
    *,
    alpha_grid: Iterable[float] | None = None,
) -> dict[str, AdjustmentResult]:
    if alpha_grid is None:
        alpha_grid = [25, 50, 75, 100, 125, 150, 200, 250, 300]

    results: dict[str, AdjustmentResult] = {}
    for stat in stats:
        if stat not in df.columns:
            continue
        try:
            result = _ridge_adjust_single_stat(df, stat, alpha_grid)
        except RuntimeError:
            continue
        results[stat] = result
    return results


def _merge_results(results: dict[str, AdjustmentResult]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    offense_frames: list[pd.DataFrame] = []
    defense_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict] = []
    for stat, result in results.items():
        offense_frames.append(result.offense.rename(columns={stat: f"{stat}_offense"}))
        defense_frames.append(result.defense.rename(columns={stat: f"{stat}_defense"}))
        metadata_rows.append({
            "stat": stat,
            "alpha": result.alpha,
            "homefield": result.homefield,
        })
    offense = offense_frames[0]
    for frame in offense_frames[1:]:
        offense = offense.merge(frame, on="team", how="outer")
    defense = defense_frames[0]
    for frame in defense_frames[1:]:
        defense = defense.merge(frame, on="team", how="outer")
    metadata = pd.DataFrame(metadata_rows)
    return offense, defense, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Opponent-adjusted FCS metrics via ridge regression.")
    parser.add_argument("year", type=int, help="Season year (e.g. 2024).")
    parser.add_argument("--api-key", type=str, help="CFBD API key (defaults to env CFBD_API_KEY).")
    parser.add_argument("--max-week", type=int, help="Optional upper bound on week number to include.")
    parser.add_argument(
        "--stats",
        type=str,
        nargs="*",
        default=list(DEFAULT_STATS),
        help="Stat categories to adjust (default subset of yards/points metrics).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("fcs_adjusted"), help="Directory for CSV outputs.")
    args = parser.parse_args()

    df = collect_fcs_game_stats(
        args.year,
        api_key=args.api_key,
        max_week=args.max_week,
        stats=args.stats,
    )
    results = compute_opponent_adjustments(df, args.stats)
    if not results:
        raise RuntimeError("No stats were adjusted; check input categories.")
    offense, defense, metadata = _merge_results(results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    offense.to_csv(args.out_dir / f"fcs_adjusted_offense_{args.year}.csv", index=False)
    defense.to_csv(args.out_dir / f"fcs_adjusted_defense_{args.year}.csv", index=False)
    metadata.to_csv(args.out_dir / f"fcs_adjusted_metadata_{args.year}.csv", index=False)

    print(json.dumps({stat: {"alpha": res.alpha, "homefield": res.homefield} for stat, res in results.items()}, indent=2))


if __name__ == "__main__":
    main()
