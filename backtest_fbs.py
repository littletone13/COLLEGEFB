"""Backtesting harness for the FBS CFBD-based ratings.

For each week of an FBS season, the script:
1. Pulls completed game results from the CFBD API.
2. Builds the current rating book (using season-to-date CFBD metrics).
3. Compares predicted spreads/totals/win probabilities against actual
   outcomes and closing market lines from the provided historical CSVs.

Outputs aggregate accuracy metrics (MAE for spread/total, Brier score for
win probability) plus a simple ATS record/ROI using closing spreads.

Usage:
    export CFBD_API_KEY="..."
    python3 backtest_fbs.py --year 2024 \
        --historical ~/Desktop/PFFMODEL_FBS/2024_CFB_FBS_HISTORICALs.csv

Assumptions/Limitations:
* CFBD advanced stats are used as-is for the entire season (CFBD does not
  expose week-by-week splits), so ratings do not strictly update weekly.
* Closing lines are taken from the supplied CSV and filtered by provider
  (default DraftKings). If multiple rows exist per game, the last one is
  treated as closing.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests

import fbs

CFBD_API = "https://api.collegefootballdata.com"


def _fetch_cfbd(path: str, *, api_key: str, params: Optional[Dict[str, str]] = None) -> list[dict]:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(CFBD_API + path, headers=headers, params=params or {}, timeout=60)
    if response.status_code == 401:
        raise RuntimeError("CFBD API rejected the key (401). Verify CFBD_API_KEY.")
    response.raise_for_status()
    return response.json()


def load_closing_lines(path: Optional[Path], provider: str, year: int, api_key: str) -> pd.DataFrame:
    if path is not None and path.exists():
        df = pd.read_csv(path)
        df.columns = [c.strip().strip('"').replace('\ufeff', '') for c in df.columns]
        if provider:
            df = df[df["LineProvider"].str.lower() == provider.lower()]
        df = df.drop_duplicates(subset="Id", keep="last")
        df["Id"] = df["Id"].astype(str).str.replace('"', '').astype(int)
        df = df.set_index("Id")
        out = df[["Spread", "OverUnder", "HomeMoneyline", "AwayMoneyline"]]
        for col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    # Fallback to CFBD lines endpoint.
    lines = _fetch_cfbd(
        "/lines",
        api_key=api_key,
        params={"year": year, "seasonType": "regular", "provider": provider.lower()},
    )
    records = []
    for entry in lines:
        game_id = entry.get("id")
        for line in entry.get("lines", []):
            records.append(
                {
                    "Id": game_id,
                    "Spread": line.get("spread"),
                    "OverUnder": line.get("overUnder"),
                    "HomeMoneyline": line.get("homeMoneyline"),
                    "AwayMoneyline": line.get("awayMoneyline"),
                }
            )
    df = pd.DataFrame(records)
    if df.empty or "Id" not in df.columns:
        return pd.DataFrame(columns=["Spread", "OverUnder", "HomeMoneyline", "AwayMoneyline"])
    df = df.drop_duplicates(subset="Id", keep="last").set_index("Id")
    return df


@dataclass
class BacktestResult:
    spread_mae: float
    total_mae: float
    brier: float
    games: int
    ats_record: tuple[int, int, int]
    ats_roi: float


def evaluate_season(
    year: int,
    *,
    api_key: str,
    hist_path: Optional[Path],
    provider: str,
    max_week: Optional[int] = None,
) -> BacktestResult:
    games = _fetch_cfbd("/games", api_key=api_key, params={"year": year, "seasonType": "regular"})
    if max_week is not None:
        games = [game for game in games if (game.get("week") or 0) <= max_week]
    ratings, book = fbs.build_rating_book(
        year,
        api_key=api_key,
        calibration_games=games,
    )
    weather_lookup = {game.get("id"): game.get("weather") for game in games}
    lines = load_closing_lines(hist_path, provider, year, api_key)

    rows: list[dict] = []
    ats_wins = ats_losses = pushes = 0
    roi = 0.0

    for game in games:
        if game.get("completed") is not True:
            continue
        if game.get("homeClassification") != "fbs" or game.get("awayClassification") != "fbs":
            continue
        home_team = game["homeTeam"]
        away_team = game["awayTeam"]
        home_points = game.get("homePoints")
        away_points = game.get("awayPoints")
        if home_points is None or away_points is None:
            continue

        identifier = game.get("id")
        try:
            line_row = lines.loc[identifier]
        except KeyError:
            continue

        try:
            pred = book.predict(home_team, away_team, neutral_site=game.get("neutralSite", False))
        except KeyError:
            continue
        pred = fbs.apply_weather_adjustment(pred, weather_lookup.get(identifier))

        actual_margin = home_points - away_points
        actual_total = home_points + away_points

        spread = float(line_row["Spread"]) if not pd.isna(line_row["Spread"]) else None
        total_line = float(line_row["OverUnder"]) if not pd.isna(line_row["OverUnder"]) else None

        if spread is not None:
            edge = pred["spread_home_minus_away"] - spread
            # Determine ATS pick.
            if abs(edge) > 1e-6:
                pick_home = edge > 0
                cover = actual_margin - spread
                if pick_home:
                    if cover > 0:
                        ats_wins += 1
                        roi += 0.909
                    elif cover < 0:
                        ats_losses += 1
                        roi -= 1.0
                    else:
                        pushes += 1
                else:
                    cover = actual_margin + spread  # away cover condition mirrored
                    if cover < 0:
                        ats_wins += 1
                        roi += 0.909
                    elif cover > 0:
                        ats_losses += 1
                        roi -= 1.0
                    else:
                        pushes += 1

        rows.append(
            {
                "spread_error": pred["spread_home_minus_away"] - actual_margin,
                "total_error": pred["total_points"] - actual_total if total_line is not None else np.nan,
                "brier": (pred["home_win_prob"] - (1.0 if actual_margin > 0 else 0.0)) ** 2,
            }
        )

    if not rows:
        raise RuntimeError("No games evaluated; verify data availability.")

    df = pd.DataFrame(rows)
    spread_mae = df["spread_error"].abs().mean()
    total_mae = df["total_error"].abs().mean(skipna=True)
    brier = df["brier"].mean()
    games_eval = len(df)

    total_bets = ats_wins + ats_losses
    ats_roi = (roi / total_bets) if total_bets else 0.0

    return BacktestResult(
        spread_mae=spread_mae,
        total_mae=total_mae,
        brier=brier,
        games=games_eval,
        ats_record=(ats_wins, ats_losses, pushes),
        ats_roi=ats_roi,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the FBS rating model against historical lines.")
    parser.add_argument("--year", type=int, required=True, help="Season year (e.g., 2024).")
    parser.add_argument("--historical", type=Path, help="Optional CSV with closing lines (fallback to API).")
    parser.add_argument("--provider", type=str, default="DraftKings", help="Sportsbook provider to filter.")
    parser.add_argument("--api-key", type=str, help="Optional CFBD API key (defaults to CFBD_API_KEY env var).")
    parser.add_argument("--max-week", type=int, help="Optional limit on the maximum week to include.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD API key not provided. Use --api-key or set CFBD_API_KEY env var.")

    result = evaluate_season(
        args.year,
        api_key=api_key,
        hist_path=args.historical,
        provider=args.provider,
        max_week=args.max_week,
    )
    print(f"Season {args.year} backtest over {result.games} games")
    print(f"Spread MAE: {result.spread_mae:.2f} pts")
    print(f"Total MAE: {result.total_mae:.2f} pts")
    print(f"Brier score: {result.brier:.4f}")
    w, l, p = result.ats_record
    print(f"ATS record: {w}-{l}-{p}")
    print(f"ATS ROI (per bet, -110 assumed): {result.ats_roi*100:0.2f}%")


if __name__ == "__main__":
    main()
