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
import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import requests

import fbs
import oddslogic_loader
from cfb.config import load_config
from cfb.market import edges as edge_utils

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
    bets: int
    total_record: tuple[int, int, int]
    total_roi: float
    total_bets: int


def evaluate_season(
    year: int,
    *,
    api_key: str,
    hist_path: Optional[Path],
    provider: str,
    max_week: Optional[int] = None,
    oddslogic_lookup: Optional[Dict[Tuple[dt.date, str, str], Dict[str, object]]] = None,
    spread_edge_min: float = 0.0,
    min_provider_count: int = 0,
    total_edge_min: float = 0.0,
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
    lines = None
    if oddslogic_lookup is None:
        lines = load_closing_lines(hist_path, provider, year, api_key)

    rows: list[dict] = []
    ats_wins = ats_losses = pushes = 0
    roi = 0.0
    bets = 0
    missing_closings = 0
    total_wins = total_losses = total_pushes = 0
    total_roi = 0.0
    total_bets = 0

    edge_config = edge_utils.EdgeFilterConfig(
        spread_edge_min=spread_edge_min,
        total_edge_min=total_edge_min,
        min_provider_count=min_provider_count,
    )

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

        kickoff_date = None
        spread = None
        total_line = None
        provider_count = 0
        edge = None

        closing = None
        invert = False
        if oddslogic_lookup is not None:
            kickoff_str = game.get("startDate") or game.get("startTime")
            if kickoff_str:
                try:
                    kickoff_date = pd.to_datetime(kickoff_str).date()
                except Exception:  # pylint: disable=broad-except
                    kickoff_date = None
            if kickoff_date is None:
                missing_closings += 1
                continue
            home_key = oddslogic_loader.normalize_label(home_team)
            away_key = oddslogic_loader.normalize_label(away_team)
            lookup_key = (kickoff_date, home_key, away_key)
            closing = oddslogic_lookup.get(lookup_key)
            if closing is None:
                alt_key = (kickoff_date, away_key, home_key)
                closing = oddslogic_lookup.get(alt_key)
                if closing is not None:
                    invert = True
            if closing is None:
                missing_closings += 1
                continue
            provider_count = len((closing.get("providers") or {}))
            spread_raw = closing.get("spread_value")
            if spread_raw is not None and not pd.isna(spread_raw):
                spread = float(spread_raw)
                if invert:
                    spread = -spread
            total_raw = closing.get("total_value")
            if total_raw is not None and not pd.isna(total_raw):
                total_line = float(total_raw)
        else:
            try:
                line_row = lines.loc[identifier]
            except KeyError:
                continue
            spread = float(line_row["Spread"]) if not pd.isna(line_row["Spread"]) else None
            total_line = float(line_row["OverUnder"]) if not pd.isna(line_row["OverUnder"]) else None
            provider_count = 1 if (spread is not None or total_line is not None) else 0

        if spread is None and total_line is None:
            continue

        try:
            pred = book.predict(home_team, away_team, neutral_site=game.get("neutralSite", False))
        except KeyError:
            continue
        pred = fbs.apply_weather_adjustment(pred, weather_lookup.get(identifier))

        actual_margin = home_points - away_points
        actual_total = home_points + away_points

        if spread is not None:
            edge = pred["spread_home_minus_away"] - spread
            bet_allowed = edge_utils.allow_spread_bet(edge, provider_count, edge_config)
            if bet_allowed:
                bets += 1
                pick_home = edge > 0
                cover_margin = actual_margin + spread
                if pick_home:
                    if cover_margin > 0:
                        ats_wins += 1
                        roi += 0.909
                    elif cover_margin < 0:
                        ats_losses += 1
                        roi -= 1.0
                    else:
                        pushes += 1
                else:
                    if cover_margin < 0:
                        ats_wins += 1
                        roi += 0.909
                    elif cover_margin > 0:
                        ats_losses += 1
                        roi -= 1.0
                    else:
                        pushes += 1

        total_edge = None
        if total_line is not None:
            total_edge = pred["total_points"] - total_line
            bet_allowed = edge_utils.allow_total_bet(total_edge, provider_count, edge_config)
            if bet_allowed:
                total_bets += 1
                pick_over = total_edge > 0
                margin = actual_total - total_line
                if pick_over:
                    if margin > 0:
                        total_wins += 1
                        total_roi += 0.909
                    elif margin < 0:
                        total_losses += 1
                        total_roi -= 1.0
                    else:
                        total_pushes += 1
                else:
                    if margin < 0:
                        total_wins += 1
                        total_roi += 0.909
                    elif margin > 0:
                        total_losses += 1
                        total_roi -= 1.0
                    else:
                        total_pushes += 1

        rows.append(
            {
                "spread_error": pred["spread_home_minus_away"] - actual_margin,
                "total_error": pred["total_points"] - actual_total if total_line is not None else np.nan,
                "brier": (pred["home_win_prob"] - (1.0 if actual_margin > 0 else 0.0)) ** 2,
                "spread_edge": edge if edge is not None else np.nan,
                "total_edge": total_edge if total_edge is not None else np.nan,
                "market_spread": spread,
                "market_total": total_line,
                "market_provider_count": provider_count,
            }
        )

    if not rows:
        raise RuntimeError("No games evaluated; verify data availability.")

    if oddslogic_lookup is not None and missing_closings:
        print(f"Warning: {missing_closings} games skipped due to missing OddsLogic closings.")

    df = pd.DataFrame(rows)
    spread_mae = df["spread_error"].abs().mean()
    total_mae = df["total_error"].abs().mean(skipna=True)
    brier = df["brier"].mean()
    games_eval = len(df)

    ats_roi = (roi / bets) if bets else 0.0
    total_roi_per_bet = (total_roi / total_bets) if total_bets else 0.0

    return BacktestResult(
        spread_mae=spread_mae,
        total_mae=total_mae,
        brier=brier,
        games=games_eval,
        ats_record=(ats_wins, ats_losses, pushes),
        ats_roi=ats_roi,
        bets=bets,
        total_record=(total_wins, total_losses, total_pushes),
        total_roi=total_roi_per_bet,
        total_bets=total_bets,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the FBS rating model against historical lines.")
    parser.add_argument("--year", type=int, required=True, help="Season year (e.g., 2024).")
    parser.add_argument("--historical", type=Path, help="Optional CSV with closing lines (fallback to API).")
    parser.add_argument("--provider", type=str, default="DraftKings", help="Sportsbook provider to filter.")
    parser.add_argument("--api-key", type=str, help="Optional CFBD API key (defaults to CFBD_API_KEY env var).")
    parser.add_argument("--max-week", type=int, help="Optional limit on the maximum week to include.")
    parser.add_argument(
        "--oddslogic-dir",
        type=Path,
        help="Optional path to OddsLogic archive output (overrides --historical / CFBD lines).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD API key not provided. Use --api-key or set CFBD_API_KEY env var.")

    config = load_config()
    backtest_cfg = config.get("fbs", {}).get("backtest", {}) if isinstance(config.get("fbs"), dict) else {}
    spread_edge_min = float(backtest_cfg.get("spread_edge_min", 0.0))
    total_edge_min = float(backtest_cfg.get("total_edge_min", 0.0))
    min_provider_count = int(backtest_cfg.get("min_provider_count", 0))

    oddslogic_lookup = None
    if args.oddslogic_dir:
        df_archive = oddslogic_loader.load_archive_dataframe(args.oddslogic_dir)
        providers = [args.provider] if args.provider else None
        oddslogic_lookup = oddslogic_loader.build_closing_lookup(df_archive, "fbs", providers=providers)
        if not oddslogic_lookup:
            raise RuntimeError("No OddsLogic lines matched the requested configuration.")
        # When using archive data we do not rely on CSV/API fallbacks.
        hist_path = None
    else:
        hist_path = args.historical

    result = evaluate_season(
        args.year,
        api_key=api_key,
        hist_path=hist_path,
        provider=args.provider,
        max_week=args.max_week,
        oddslogic_lookup=oddslogic_lookup,
        spread_edge_min=spread_edge_min,
        min_provider_count=min_provider_count,
        total_edge_min=total_edge_min,
    )
    print(f"Season {args.year} backtest over {result.games} games")
    print(f"Spread MAE: {result.spread_mae:.2f} pts")
    print(f"Total MAE: {result.total_mae:.2f} pts")
    print(f"Brier score: {result.brier:.4f}")
    w, l, p = result.ats_record
    print(f"ATS record: {w}-{l}-{p}")
    print(f"ATS ROI (per bet, -110 assumed): {result.ats_roi*100:0.2f}%")
    print(f"Bets placed: {result.bets} (spread edge ≥ {spread_edge_min}, providers ≥ {min_provider_count})")
    tw, tl, tp = result.total_record
    print(f"Totals record: {tw}-{tl}-{tp}")
    print(f"Totals ROI (per bet, -110 assumed): {result.total_roi*100:0.2f}%")
    print(f"Totals bets: {result.total_bets} (total edge ≥ {total_edge_min}, providers ≥ {min_provider_count})")


if __name__ == "__main__":
    main()
