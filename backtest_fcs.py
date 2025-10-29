"""Backtest the FCS PFF-based model using CFBD historical game results."""
from __future__ import annotations

import argparse
import difflib
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

import fcs
import ncaa_stats
from simulate_fcs_week import TEAM_NAME_ALIASES, _normalize_label

CFBD_API = "https://api.collegefootballdata.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FCS projections against NCAA.com historical results.")
    parser.add_argument("start_year", type=int, help="First season year to backtest (e.g., 2022).")
    parser.add_argument("end_year", type=int, help="Last season year to backtest (e.g., 2024).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fcs_backtest_games.csv"),
        help="CSV path for per-game backtest rows.",
    )
    parser.add_argument(
        "--season-summary",
        type=Path,
        help="Optional CSV path for per-season accuracy metrics.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=fcs.DATA_DIR_DEFAULT,
        help="Path to the FCS PFF data directory (default matches fcs.py).",
    )
    parser.add_argument(
        "--season-type",
        choices=["regular", "postseason", "both"],
        default="regular",
        help="Season type(s) to include from CFBD games (default regular).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="CFBD API key (falls back to CFBD_API_KEY env var).",
    )
    parser.add_argument(
        "--max-week",
        type=int,
        help="Optional limit on maximum week to include when evaluating seasons.",
    )
    return parser.parse_args()


def fetch_cfbd_games(year: int, api_key: str, season_type: str) -> List[dict]:
    season_param = "regular" if season_type == "regular" else season_type
    params = {
        "year": year,
        "division": "fcs",
        "seasonType": season_param,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(CFBD_API + "/games", headers=headers, params=params, timeout=60)
    if resp.status_code == 401:
        raise RuntimeError("CFBD API rejected the key (401). Confirm CFBD_API_KEY.")
    resp.raise_for_status()
    games = resp.json()
    if season_type == "both":
        params["seasonType"] = "postseason"
        post = requests.get(CFBD_API + "/games", headers=headers, params=params, timeout=60)
        post.raise_for_status()
        games.extend(post.json())
    return games


PFF_NAMES = list(ncaa_stats.SLUG_TO_PFF.values())
PFF_SET = set(PFF_NAMES)


def map_team(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    raw = name.upper()
    if raw in PFF_SET:
        return raw
    normalized = _normalize_label(raw)
    alias = TEAM_NAME_ALIASES.get(normalized)
    if alias:
        return alias
    match = difflib.get_close_matches(raw, PFF_NAMES, n=1, cutoff=0.5)
    return match[0] if match else None


def evaluate_season(
    year: int,
    data_dir: Path,
    api_key: str,
    season_type: str,
    max_week: Optional[int] = None,
) -> List[Dict[str, object]]:
    ratings = fcs.load_team_ratings(data_dir=data_dir, season_year=year)
    book = fcs.RatingBook(ratings, fcs.RatingConstants())
    games = fetch_cfbd_games(year, api_key, season_type)
    if max_week is not None:
        games = [game for game in games if (game.get("week") or 0) <= max_week]
    if not games:
        return []

    rows: List[Dict[str, object]] = []
    for game in games:
        if game.get("completed") is not True:
            continue
        home = map_team(game.get("homeTeam"))
        away = map_team(game.get("awayTeam"))
        if not home or not away:
            continue
        actual_home = game.get("homePoints")
        actual_away = game.get("awayPoints")
        if actual_home is None or actual_away is None:
            continue
        try:
            pred = book.predict(home, away, neutral_site=False)
        except KeyError:
            continue
        actual_home = float(actual_home)
        actual_away = float(actual_away)
        actual_margin = actual_home - actual_away
        actual_total = actual_home + actual_away
        home_flag = 1.0 if actual_margin > 0 else (0.0 if actual_margin < 0 else 0.5)
        rows.append(
            {
                "season": year,
                "date": game.get("startDate"),
                "week": game.get("week"),
                "season_type": game.get("seasonType"),
                "home_team": home,
                "away_team": away,
                "actual_home_points": actual_home,
                "actual_away_points": actual_away,
                "actual_margin": actual_margin,
                "actual_total": actual_total,
                "pred_home_points": pred.get("home_points") or pred.get("team_one_points"),
                "pred_away_points": pred.get("away_points") or pred.get("team_two_points"),
                "pred_spread": pred.get("spread_home_minus_away") or pred.get("spread_team_one_minus_team_two"),
                "pred_total": pred["total_points"],
                "pred_home_win_prob": pred.get("home_win_prob") or pred.get("team_one_win_prob"),
                "spread_error": (pred.get("spread_home_minus_away") or pred.get("spread_team_one_minus_team_two")) - actual_margin,
                "total_error": pred["total_points"] - actual_total,
                "brier": (pred.get("home_win_prob") or pred.get("team_one_win_prob") - home_flag) ** 2,
            }
        )
    return rows


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for season, group in df.groupby("season"):
        summaries.append(
            {
                "season": season,
                "games": len(group),
                "spread_mae": group["spread_error"].abs().mean(),
                "total_mae": group["total_error"].abs().mean(),
                "brier": group["brier"].mean(),
            }
        )
    return pd.DataFrame(summaries).sort_values("season")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    start = min(args.start_year, args.end_year)
    end = max(args.start_year, args.end_year)
    api_key = args.api_key or os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD API key required. Set CFBD_API_KEY or pass --api-key.")

    all_rows: List[Dict[str, object]] = []
    for season in range(start, end + 1):
        rows = evaluate_season(season, data_dir, api_key, args.season_type, max_week=args.max_week)
        if not rows:
            print(f"No games processed for season {season}; check data availability.")
            continue
        all_rows.extend(rows)
        print(f"Season {season}: processed {len(rows)} games")

    if not all_rows:
        print("No games available across requested seasons.")
        return

    df = pd.DataFrame(all_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved per-game backtest data to {args.output}")

    summary = build_summary(df)
    if args.season_summary:
        args.season_summary.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.season_summary, index=False)
        print(f"Saved per-season summary to {args.season_summary}")

    overall = summary.assign(games=summary["games"].astype(int))
    total_games = df.shape[0]
    print(f"Overall across {total_games} games:")
    print(f"  Spread MAE: {df['spread_error'].abs().mean():.2f}")
    print(f"  Total MAE: {df['total_error'].abs().mean():.2f}")
    print(f"  Brier score: {df['brier'].mean():.4f}")


if __name__ == "__main__":
    main()
