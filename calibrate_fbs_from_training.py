"""Recalibrate FBS spread regression using the weekly Patreon training data."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import fbs
from cfb.io import odds_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit linear spread calibration based on training data.")
    parser.add_argument("training_dir", type=Path, help="Directory containing training_data_2025_week*.csv files.")
    parser.add_argument("--year", type=int, default=2025, help="Season year (default 2025).")
    parser.add_argument("--api-key", type=str, help="CFBD API key (otherwise CFBD_API_KEY env var).")
    parser.add_argument("--adjust-week", type=int, default=10, help="Week to use for opponent adjustments (default 10).")
    parser.add_argument("--output", type=Path, help="Optional CSV dump of raw vs vegas spreads.")
    parser.add_argument(
        "--odds-api-dir",
        type=Path,
        help="Optional directory of The Odds API history (uses FanDuel closings when provided).",
    )
    parser.add_argument(
        "--bookmaker",
        type=str,
        help="Deprecated: single bookmaker key (use --bookmakers).",
    )
    parser.add_argument(
        "--bookmakers",
        type=str,
        nargs="+",
        help="One or more bookmaker keys from the odds-history archive (default: FanDuel + BetOnlineAG).",
    )
    return parser.parse_args()


def load_training_rows(training_dir: Path) -> pd.DataFrame:
    files: Iterable[Path] = sorted(training_dir.glob("training_data_2025_week*.csv"))
    frames = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            continue
        if df.empty:
            continue
        df = df[[
            "id",
            "week",
            "home_team",
            "away_team",
            "neutral_site",
            "spread",
        ]].copy()
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
        # CFBD training pack spreads reflect opening numbers, so persist that metadata.
        df["vegas_spread_opening"] = df["spread"]
        df["vegas_line_type"] = "opening"
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No training_data_2025_week*.csv found in {training_dir}.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD API key required via --api-key or CFBD_API_KEY env var.")

    rows = load_training_rows(args.training_dir)
    ratings, book = fbs.build_rating_book(
        args.year,
        api_key=api_key,
        adjust_week=args.adjust_week,
        calibration_games=None,
    )
    # Remove previous calibration so we can fit a new one.
    book.spread_calibration = (0.0, 1.0)

    default_books = ["fanduel", "betonlineag"]
    bookmakers = args.bookmakers or ([args.bookmaker] if args.bookmaker else default_books)
    closing_lookup: dict[int, dict[str, float]] = {}
    closing_counts: dict[str, int] = {}
    if args.odds_api_dir:
        for bookmaker in bookmakers:
            closing_df = odds_history.load_closing_prices(
                args.odds_api_dir,
                year=args.year,
                provider=bookmaker,
                classification="fbs",
            )
            if closing_df.empty:
                continue
            closing_counts[bookmaker] = len(closing_df)
            for game_id, row in closing_df.iterrows():
                idx = int(game_id)
                entry = closing_lookup.get(idx, {})
                if "Spread" not in entry or pd.isna(entry.get("Spread")):
                    payload = row.to_dict()
                    payload["_bookmaker"] = bookmaker
                    closing_lookup[idx] = payload
                else:
                    closing_lookup.setdefault(idx, entry)
    if closing_counts:
        total_closings = sum(closing_counts.values())
        joined = ", ".join(f"{book}:{count}" for book, count in sorted(closing_counts.items()))
        print(f"Loaded {total_closings} closing rows from bookmakers -> {joined}")

    records = []
    missing = 0
    missing_closing = 0
    for row in rows.itertuples(index=False):
        try:
            pred = book.predict(row.home_team, row.away_team, neutral_site=bool(row.neutral_site))
        except KeyError:
            missing += 1
            continue
        opening_spread = getattr(row, "vegas_spread_opening", row.spread)
        vegas_home_minus_away = -float(opening_spread)
        closing_home_minus_away = None
        if closing_lookup:
            row_id = getattr(row, "id", None)
            closing_entry = closing_lookup.get(int(row_id)) if row_id is not None else None
            if closing_entry:
                closing_value = closing_entry.get("Spread")
                if closing_value is not None and not pd.isna(closing_value):
                    closing_home_minus_away = float(closing_value)
                else:
                    missing_closing += 1
            else:
                missing_closing += 1
        records.append(
            {
                "game_id": getattr(row, "id", None),
                "week": row.week,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "vegas_spread": vegas_home_minus_away,
                "vegas_spread_closing": closing_home_minus_away,
                "raw_spread": pred["spread_home_minus_away"],
            }
        )
    if not records:
        raise RuntimeError("No overlapping games between training data and rating book.")

    df = pd.DataFrame(records)
    df = df.dropna(subset=["vegas_spread", "raw_spread"])
    X = df["raw_spread"].values
    y = df["vegas_spread"].values
    slope, intercept = np.polyfit(X, y, 1)

    print(f"Fit on {len(df)} games (skipped {missing} mismatches).")
    print(f"Spread calibration -> intercept: {intercept:.6f}, slope: {slope:.6f}")

    closing_stats: Optional[tuple[float, float]] = None
    if df["vegas_spread_closing"].notna().any():
        df_closing = df.dropna(subset=["vegas_spread_closing"])
        if not df_closing.empty:
            slope_close, intercept_close = np.polyfit(
                df_closing["raw_spread"].values,
                df_closing["vegas_spread_closing"].values,
                1,
            )
            closing_stats = (intercept_close, slope_close)
            print(
                f"Closing calibration ({len(df_closing)} games, skipped {missing_closing}): "
                f"intercept {intercept_close:.6f}, slope {slope_close:.6f}"
            )
        else:
            print("No closing spreads aligned with the training rows.")
    elif closing_lookup:
        print(f"Closing spreads provided but none matched training rows (missing {missing_closing}).")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved calibration dataset to {args.output}")

    if closing_stats:
        intercept_close, slope_close = closing_stats
        print(
            f"Difference vs opening: intercept shift {intercept_close - intercept:+.6f}, "
            f"slope shift {slope_close - slope:+.6f}"
        )


if __name__ == "__main__":
    main()
