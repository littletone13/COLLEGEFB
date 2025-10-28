"""Recalibrate FBS spread regression using the weekly Patreon training data."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import fbs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit linear spread calibration based on training data.")
    parser.add_argument("training_dir", type=Path, help="Directory containing training_data_2025_week*.csv files.")
    parser.add_argument("--year", type=int, default=2025, help="Season year (default 2025).")
    parser.add_argument("--api-key", type=str, help="CFBD API key (otherwise CFBD_API_KEY env var).")
    parser.add_argument("--adjust-week", type=int, default=10, help="Week to use for opponent adjustments (default 10).")
    parser.add_argument("--output", type=Path, help="Optional CSV dump of raw vs vegas spreads.")
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
            "week",
            "home_team",
            "away_team",
            "neutral_site",
            "spread",
        ]].copy()
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

    records = []
    missing = 0
    for row in rows.itertuples(index=False):
        try:
            pred = book.predict(row.home_team, row.away_team, neutral_site=bool(row.neutral_site))
        except KeyError:
            missing += 1
            continue
        vegas_home_minus_away = -float(row.spread)
        records.append(
            {
                "week": row.week,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "vegas_spread": vegas_home_minus_away,
                "raw_spread": pred["spread_home_minus_away"],
            }
        )
    if not records:
        raise RuntimeError("No overlapping games between training data and rating book.")

    df = pd.DataFrame(records).dropna()
    X = df["raw_spread"].values
    y = df["vegas_spread"].values
    slope, intercept = np.polyfit(X, y, 1)

    print(f"Fit on {len(df)} games (skipped {missing} mismatches).")
    print(f"Spread calibration -> intercept: {intercept:.6f}, slope: {slope:.6f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved calibration dataset to {args.output}")


if __name__ == "__main__":
    main()
