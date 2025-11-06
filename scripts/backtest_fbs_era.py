#!/usr/bin/env python3
"""Backtest FBS results with era splits and exponential decay weights.

Usage:
    python scripts/backtest_fbs_era.py --in data/backtest.parquet --format parquet --out out/
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="inpath",
        required=True,
        help="Input CSV/Parquet file with per-game or per-bet rows.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="parquet",
        help="Input file format (default parquet).",
    )
    parser.add_argument("--out", dest="outdir", default="out", help="Directory for outputs.")
    parser.add_argument("--current-year", type=int, default=2025)
    parser.add_argument(
        "--post-clock-start",
        type=int,
        default=2023,
        help="Season marking the start of the Post-ClockRule era.",
    )
    parser.add_argument(
        "--era-split",
        action="store_true",
        help="Include era-level summaries.",
    )
    parser.add_argument(
        "--decay-lambda",
        type=float,
        default=0.5,
        help="Exponential decay lambda per season (default 0.5).",
    )
    parser.add_argument(
        "--exclude-covid",
        action="store_true",
        help="Drop the 2020 season from metrics.",
    )
    parser.add_argument(
        "--min-season",
        type=int,
        default=2021,
        help="Drop seasons before this value after loading.",
    )
    parser.add_argument(
        "--id-col",
        default="game_id",
        help="Grouping identifier (if present).",
    )
    return parser.parse_args()


def load_df(path: str, fmt: str) -> pd.DataFrame:
    if fmt == "csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    if "market" in df.columns:
        df["market"] = df["market"].astype(str).str.lower()
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
    return df


def assign_era(season: float, post_clock_start: int) -> str:
    if pd.isna(season):
        return "Unknown"
    if season <= 2020:
        return "COVID"
    if season < post_clock_start:
        return "Pre-ClockRule"
    return "Post-ClockRule"


def era_weight(season: float, current_year: int, lam: float) -> float:
    if pd.isna(season):
        return 0.0
    seasons_ago = max(0.0, current_year - float(season))
    return float(np.exp(-seasons_ago * lam))


def compute_roi(df: pd.DataFrame) -> Tuple[float, int]:
    if {"bet_outcome", "bet_price"}.issubset(df.columns):
        stake = df["stake"] if "stake" in df.columns else 1.0

        def price_to_payout(american: object) -> float:
            try:
                value = float(american)
            except Exception:  # pylint: disable=broad-except
                return np.nan
            if value > 0:
                return value / 100.0
            return 100.0 / abs(value)

        payout_mult = df["bet_price"].apply(price_to_payout)

        def profit(row: pd.Series) -> float:
            outcome = row["bet_outcome"]
            multiplier = row["payout_mult"]
            if pd.isna(outcome) or pd.isna(multiplier):
                return np.nan
            if outcome == 1:
                return multiplier
            if outcome == 0:
                return 0.0
            if outcome == -1:
                return -1.0
            return np.nan

        tmp = df.copy()
        tmp["payout_mult"] = payout_mult
        tmp["profit_units"] = tmp.apply(profit, axis=1)

        if "stake" in tmp.columns:
            total_staked = tmp["stake"].sum()
            total_profit = (tmp["profit_units"] * tmp["stake"]).sum()
        else:
            total_staked = len(tmp)
            total_profit = tmp["profit_units"].sum()

        if total_staked and not pd.isna(total_staked):
            return float(total_profit / total_staked), int(len(tmp))

    return float("nan"), 0


def compute_metrics(df: pd.DataFrame) -> dict:
    metrics: dict = {}
    if {"model_spread", "close_spread"}.issubset(df.columns):
        metrics["spread_mae"] = (df["model_spread"] - df["close_spread"]).abs().mean()
    if {"model_total", "close_total"}.issubset(df.columns):
        metrics["total_mae"] = (df["model_total"] - df["close_total"]).abs().mean()

    roi_value, roi_count = compute_roi(df)
    if roi_count:
        metrics["roi"] = roi_value
        metrics["n_bets_for_roi"] = roi_count

    if {"bet_line", "close_line"}.issubset(df.columns):
        deltas = pd.to_numeric(df["close_line"], errors="coerce") - pd.to_numeric(
            df["bet_line"], errors="coerce"
        )
        metrics["clv_line_delta_mean"] = deltas.mean()
        metrics["clv_line_delta_median"] = deltas.median()

    if {"bet_price", "close_price"}.issubset(df.columns):
        price_delta = pd.to_numeric(df["close_price"], errors="coerce") - pd.to_numeric(
            df["bet_price"], errors="coerce"
        )
        metrics["clv_price_delta_mean"] = price_delta.mean()
        metrics["clv_price_delta_median"] = price_delta.median()

    metrics["rows"] = int(len(df))
    return metrics


def summarize(
    df: pd.DataFrame,
    *,
    current_year: int,
    post_clock_start: int,
    lam: float,
    era_split: bool,
    min_season: Optional[int],
    exclude_covid: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    if "season" in working.columns:
        if min_season is not None:
            working = working[working["season"] >= min_season]
        if exclude_covid:
            working = working[working["season"] != 2020]
    else:
        working["season"] = np.nan

    working["era"] = working["season"].apply(lambda s: assign_era(s, post_clock_start))
    working["era_weight"] = working["season"].apply(lambda s: era_weight(s, current_year, lam))

    overall_metrics = compute_metrics(working)
    overall_df = pd.DataFrame([overall_metrics])

    weighted_metrics: dict = {}
    weights = working["era_weight"].fillna(0.0)
    weight_sum = weights.sum()
    if {"model_spread", "close_spread"}.issubset(working.columns) and weight_sum:
        diffs = (working["model_spread"] - working["close_spread"]).abs()
        weighted_metrics["spread_mae_weighted"] = float((diffs * weights).sum() / weight_sum)
    if {"model_total", "close_total"}.issubset(working.columns) and weight_sum:
        diffs = (working["model_total"] - working["close_total"]).abs()
        weighted_metrics["total_mae_weighted"] = float((diffs * weights).sum() / weight_sum)
    weighted_df = pd.DataFrame([weighted_metrics])

    clv_metrics: dict = {}
    if {"bet_line", "close_line"}.issubset(working.columns):
        delta = pd.to_numeric(working["close_line"], errors="coerce") - pd.to_numeric(
            working["bet_line"], errors="coerce"
        )
        clv_metrics["clv_line_delta_mean"] = delta.mean()
        clv_metrics["clv_line_delta_median"] = delta.median()
    if {"bet_price", "close_price"}.issubset(working.columns):
        delta = pd.to_numeric(working["close_price"], errors="coerce") - pd.to_numeric(
            working["bet_price"], errors="coerce"
        )
        clv_metrics["clv_price_delta_mean"] = delta.mean()
        clv_metrics["clv_price_delta_median"] = delta.median()
    clv_df = pd.DataFrame([clv_metrics]) if clv_metrics else pd.DataFrame()

    era_rows = []
    if era_split:
        for era, group in working.groupby("era"):
            result = {"era": era}
            result.update(compute_metrics(group))
            era_rows.append(result)
    era_df = pd.DataFrame(era_rows) if era_rows else pd.DataFrame()

    return overall_df, weighted_df, clv_df, era_df, working


def main() -> None:
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    dataset = load_df(args.inpath, args.format)
    overall_df, weighted_df, clv_df, era_df, weighted_out = summarize(
        dataset,
        current_year=args.current_year,
        post_clock_start=args.post_clock_start,
        lam=args.decay_lambda,
        era_split=args.era_split,
        min_season=args.min_season,
        exclude_covid=args.exclude_covid,
    )

    overall_df.to_csv(Path(args.outdir) / "fbs_overall_summary.csv", index=False)
    weighted_df.to_csv(Path(args.outdir) / "fbs_overall_weighted_summary.csv", index=False)
    if not clv_df.empty:
        clv_df.to_csv(Path(args.outdir) / "fbs_clv_summary.csv", index=False)
    if not era_df.empty:
        era_df.to_csv(Path(args.outdir) / "fbs_era_summary.csv", index=False)
    try:
        weighted_out.to_parquet(Path(args.outdir) / "fbs_weighted_dataset.parquet", index=False)
    except Exception:  # pylint: disable=broad-except
        weighted_out.to_csv(Path(args.outdir) / "fbs_weighted_dataset.csv", index=False)

    print(f"Wrote summaries to {args.outdir}")


if __name__ == "__main__":
    main()
