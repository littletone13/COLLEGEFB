"""Backtest the FBS model across large historical samples using CFBD data."""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

import fbs

CFBD_API = "https://api.collegefootballdata.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a multi-season dataset that compares CFBD closing lines to the FBS "
            "model and grades ATS performance."
        )
    )
    parser.add_argument("start_date", type=str, help="Earliest kickoff date to include (YYYY-MM-DD).")
    parser.add_argument("end_date", type=str, help="Latest kickoff date to include (YYYY-MM-DD).")
    parser.add_argument("--provider", type=str, default="DraftKings", help="Sportsbook provider tag (default DraftKings).")
    parser.add_argument(
        "--season-type",
        choices=["regular", "postseason", "both"],
        default="regular",
        help="Which CFBD season type(s) to include.",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.0,
        help=(
            "Minimum absolute spread edge (in points) required before logging an ATS bet/ROI. "
            "All games are still saved to the output CSV."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="CFBD API key; defaults to CFBD_API_KEY environment variable if omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("historic_edges.csv"),
        help="CSV path for the per-game backtest output (default historic_edges.csv).",
    )
    parser.add_argument(
        "--season-summary",
        type=Path,
        help="Optional CSV path to store per-season accuracy + ATS metrics.",
    )
    return parser.parse_args()


def season_year_for_date(day: datetime) -> int:
    """Map a calendar date to its NCAA season year (Jan-Jun belongs to previous season)."""

    return day.year if day.month >= 7 else day.year - 1


def parse_game_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def normalize_utc(dt: datetime) -> datetime:
    """Return a timezone-naive UTC datetime for comparisons."""

    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _fetch_cfbd(path: str, *, api_key: str, params: Optional[Dict[str, str]] = None) -> list[dict]:
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(CFBD_API + path, headers=headers, params=params or {}, timeout=60)
    if resp.status_code == 401:
        raise RuntimeError("CFBD API rejected the key (401). Confirm CFBD_API_KEY.")
    resp.raise_for_status()
    return resp.json()


def fetch_games(year: int, season_types: Iterable[str], api_key: str) -> list[dict]:
    games: list[dict] = []
    for kind in season_types:
        games.extend(fbs.fetch_games(year, api_key, season_type=kind))
    return games


def fetch_lines(year: int, season_types: Iterable[str], provider: str, api_key: str) -> pd.DataFrame:
    records: list[dict] = []
    for kind in season_types:
        data = _fetch_cfbd(
            "/lines",
            api_key=api_key,
            params={"year": year, "seasonType": kind, "provider": provider.lower()},
        )
        for entry in data:
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
    if df.empty:
        return df
    df = df.drop_duplicates(subset="Id", keep="last").set_index("Id")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_rating_book(
    year: int,
    api_key: str,
    cache: Dict[int, tuple[pd.DataFrame, fbs.RatingBook]],
) -> tuple[pd.DataFrame, fbs.RatingBook]:
    if year not in cache:
        cache[year] = fbs.build_rating_book(year, api_key=api_key)
    return cache[year]


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD API key required; set CFBD_API_KEY or pass --api-key.")

    start = normalize_utc(datetime.strptime(args.start_date, "%Y-%m-%d"))
    end = normalize_utc(datetime.strptime(args.end_date, "%Y-%m-%d"))
    if end < start:
        raise ValueError("end_date cannot be earlier than start_date.")

    season_start = season_year_for_date(start)
    season_end = season_year_for_date(end)
    if season_end < season_start:
        raise RuntimeError("No overlapping NCAA seasons for the provided date range.")

    season_types: List[str]
    if args.season_type == "both":
        season_types = ["regular", "postseason"]
    else:
        season_types = [args.season_type]

    cache: Dict[int, tuple[pd.DataFrame, fbs.RatingBook]] = {}
    rows: list[dict] = []
    bet_wins = bet_losses = bet_pushes = 0
    roi_units = 0.0

    for season in range(season_start, season_end + 1):
        try:
            _, book = get_rating_book(season, api_key, cache)
        except RuntimeError as exc:
            print(f"Skipping season {season}: {exc}")
            continue

        games = fetch_games(season, season_types, api_key)
        if not games:
            continue
        lines = fetch_lines(season, season_types, args.provider, api_key)
        line_lookup = lines.to_dict(orient="index") if not lines.empty else {}
        weather_lookup = {game.get("id"): game.get("weather") for game in games}

        for game in games:
            kickoff = parse_game_datetime(game.get("startDate"))
            if kickoff is None:
                continue
            kickoff = normalize_utc(kickoff)
            if kickoff < start or kickoff > end:
                continue
            if game.get("completed") is not True:
                continue
            if game.get("homeClassification") != "fbs" or game.get("awayClassification") != "fbs":
                continue
            home_points = game.get("homePoints")
            away_points = game.get("awayPoints")
            if home_points is None or away_points is None:
                continue

            identifier = game.get("id")
            line_row = line_lookup.get(identifier)

            try:
                pred = book.predict(game["homeTeam"], game["awayTeam"], neutral_site=game.get("neutralSite", False))
            except KeyError:
                continue
            pred = fbs.apply_weather_adjustment(pred, weather_lookup.get(identifier))

            market_spread = fbs._coerce_float(line_row.get("Spread")) if line_row else None
            market_total = fbs._coerce_float(line_row.get("OverUnder")) if line_row else None
            home_ml = fbs._coerce_float(line_row.get("HomeMoneyline")) if line_row else None
            away_ml = fbs._coerce_float(line_row.get("AwayMoneyline")) if line_row else None

            actual_margin = home_points - away_points
            actual_total = home_points + away_points
            model_spread = pred["spread_home_minus_away"]
            model_total = pred["total_points"]

            spread_edge = None if market_spread is None else model_spread - market_spread
            total_edge = None if market_total is None else model_total - market_total
            model_pick = None
            bet_active = False
            bet_result = None

            if spread_edge is not None and abs(spread_edge) > 0:
                model_pick = "home" if spread_edge > 0 else "away"
                if abs(spread_edge) >= args.min_edge:
                    bet_active = True
                    cover_margin = actual_margin - market_spread
                    if abs(cover_margin) < 1e-6:
                        bet_result = "push"
                        bet_pushes += 1
                    elif (cover_margin > 0 and model_pick == "home") or (cover_margin < 0 and model_pick == "away"):
                        bet_result = "win"
                        bet_wins += 1
                        roi_units += 0.909
                    else:
                        bet_result = "loss"
                        bet_losses += 1
                        roi_units -= 1.0

            home_win_flag = 1.0 if actual_margin > 0 else (0.0 if actual_margin < 0 else 0.5)

            rows.append(
                {
                    "season": season,
                    "season_type": game.get("seasonType"),
                    "week": game.get("week"),
                    "game_id": identifier,
                    "start_date": kickoff.isoformat(),
                    "home_team": pred["home_team"],
                    "away_team": pred["away_team"],
                    "actual_home_points": home_points,
                    "actual_away_points": away_points,
                    "actual_margin": actual_margin,
                    "actual_total": actual_total,
                    "model_spread": model_spread,
                    "model_total": model_total,
                    "home_win_prob": pred["home_win_prob"],
                    "market_spread": market_spread,
                    "market_total": market_total,
                    "home_moneyline": home_ml,
                    "away_moneyline": away_ml,
                    "spread_edge": spread_edge,
                    "total_edge": total_edge,
                    "spread_error": model_spread - actual_margin,
                    "total_error": model_total - actual_total,
                    "brier": (pred["home_win_prob"] - home_win_flag) ** 2,
                    "model_pick": model_pick,
                    "bet_tracked": bet_active,
                    "bet_result": bet_result,
                }
            )

    if not rows:
        print("No games matched the requested filters.")
        return

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    spread_mae = df["spread_error"].abs().mean()
    total_mae = df["total_error"].abs().mean()
    brier = df["brier"].mean()

    print(f"Saved {len(df)} games spanning seasons {season_start}-{season_end} to {args.output}")
    print(f"Spread MAE: {spread_mae:.2f} pts")
    print(f"Total MAE: {total_mae:.2f} pts")
    print(f"Brier score: {brier:.4f}")

    bets_graded = bet_wins + bet_losses
    if bets_graded > 0:
        roi = roi_units / bets_graded
        print(
            f"ATS (|edge|>={args.min_edge:.1f}): {bet_wins}-{bet_losses}-{bet_pushes} | ROI {roi*100:0.2f}%"
        )
    else:
        print(
            "No spreads met the edge threshold for ATS grading. Set --min-edge lower to record bets."
        )

    if args.season_summary:
        summary_rows = []
        for season, group in df.groupby("season"):
            games = len(group)
            spread_mae_s = group["spread_error"].abs().mean()
            total_mae_s = group["total_error"].abs().mean()
            brier_s = group["brier"].mean()
            bets = group[group["bet_tracked"]]
            wins = (bets["bet_result"] == "win").sum()
            losses = (bets["bet_result"] == "loss").sum()
            pushes = (bets["bet_result"] == "push").sum()
            bet_count = wins + losses
            roi = ((0.909 * wins) - losses) / bet_count if bet_count else 0.0
            summary_rows.append(
                {
                    "season": season,
                    "games": games,
                    "spread_mae": spread_mae_s,
                    "total_mae": total_mae_s,
                    "brier": brier_s,
                    "bets": len(bets),
                    "ats_record": f"{wins}-{losses}-{pushes}",
                    "ats_roi": roi,
                }
            )
        summary_df = pd.DataFrame(summary_rows).sort_values("season")
        args.season_summary.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.season_summary, index=False)
        print(f"Per-season summary saved to {args.season_summary}")


if __name__ == "__main__":
    main()
