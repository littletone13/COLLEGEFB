#!/usr/bin/env python3
"""Calibrate FBS model biases and uncertainty using CFBD-only backtests.

This script consumes per-game outputs from `backtest_fbs.evaluate_season`
and reports:
  * Spread and total bias (mean/standard deviation of model minus market)
  * Win-probability calibration (predicted vs. realized hit rate buckets)
  * Provider-specific ATS/total ROI trends
  * Scoring environment adjustments by season

Example:
    PYTHONPATH=. CFBD_API_KEY=... \\
        python scripts/analyze_fbs_calibration.py --start-year 2023 --end-year 2025 \\
            --provider DraftKings --out out/fbs_calibration_report

Outputs:
    <out>_bias.csv            Mean/std deltas for spread/total
    <out>_prob_calibration.csv Calibration by probability buckets
    <out>_season_scoring.csv  Actual scoring averages per season
    <out>_summary.txt         Human-readable recommendations
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backtest_fbs import BacktestResult, evaluate_season


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, required=True, help="First season to include (inclusive).")
    parser.add_argument("--end-year", type=int, required=True, help="Last season to include (inclusive).")
    parser.add_argument("--provider", type=str, required=True, help="CFBD provider name to evaluate.")
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("CFBD_API_KEY"),
        help="CFBD API key (default: CFBD_API_KEY env var).",
    )
    parser.add_argument(
        "--prob-bins",
        nargs="+",
        type=float,
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
        help="Win probability breakpoints for calibration (exclusive upper bounds).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("out/fbs_calibration_report"),
        help="Base path for output artifacts (extensions added automatically).",
    )
    parser.add_argument(
        "--odds-api-dir",
        type=Path,
        help="Optional directory of The Odds API history to use for closing lines.",
    )
    parser.add_argument(
        "--bookmaker",
        type=str,
        default="fanduel",
        help="Bookmaker key inside the odds-history archive (default fanduel).",
    )
    return parser.parse_args()


def bucket_probs(prob: pd.Series, bins: Sequence[float]) -> pd.Series:
    edges = [0.0] + list(bins) + [1.0]
    labels = []
    for low, high in zip(edges[:-1], edges[1:]):
        labels.append(f"{int(low*100):02d}-{int(high*100):02d}%")
    return pd.cut(prob.clip(0.0, 1.0), edges, labels=labels, include_lowest=True, right=False)


def collect_rows(
    years: Sequence[int],
    provider: str,
    *,
    api_key: str,
    odds_api_dir: Optional[Path] = None,
    bookmaker: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[BacktestResult]]:
    frames: List[pd.DataFrame] = []
    results: List[BacktestResult] = []
    for year in years:
        result, df = evaluate_season(
            year,
            api_key=api_key,
            hist_path=None,
            provider=bookmaker or provider,
            oddslogic_lookup=None,
            max_week=None,
            spread_edge_min=0.0,
            min_provider_count=0,
            total_edge_min=0.0,
            return_rows=True,
            odds_api_dir=odds_api_dir,
        )
        df["season"] = year
        df["provider"] = provider
        frames.append(df)
        results.append(result)
    combined = pd.concat(frames, ignore_index=True)
    return combined, results


def compute_bias(df: pd.DataFrame) -> pd.DataFrame:
    payload = {
        "metric": ["spread", "total"],
        "mean_delta": [
            (df["model_spread"] - df["market_spread"]).mean(),
            (df["model_total"] - df["market_total"]).mean(),
        ],
        "std_delta": [
            (df["model_spread"] - df["market_spread"]).std(ddof=0),
            (df["model_total"] - df["market_total"]).std(ddof=0),
        ],
        "median_delta": [
            (df["model_spread"] - df["market_spread"]).median(),
            (df["model_total"] - df["market_total"]).median(),
        ],
        "games": [
            df["market_spread"].notna().sum(),
            df["market_total"].notna().sum(),
        ],
    }
    return pd.DataFrame(payload)


def probability_calibration(df: pd.DataFrame, bins: Sequence[float]) -> pd.DataFrame:
    working = df.rename(columns={"model_home_win_prob": "model_win_prob"}).copy()
    if "model_win_prob" not in working.columns:
        raise RuntimeError("No model win probability column available for calibration.")

    working["bucket"] = bucket_probs(working["model_win_prob"], bins)
    working["actual_win"] = np.where(working["actual_margin"].notna(), working["actual_margin"] > 0, np.nan)
    working["actual_win"] = working["actual_win"].astype(float)

    grouped = (
        working.groupby("bucket", observed=True)
        .agg(
            games=("actual_win", "count"),
            predicted_mean=("model_win_prob", "mean"),
            actual_mean=("actual_win", "mean"),
        )
        .reset_index()
    )
    grouped["calibration_gap"] = grouped["actual_mean"] - grouped["predicted_mean"]
    return grouped


def scoring_environment(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("season")
        .agg(
            games=("actual_total", "count"),
            avg_total=("actual_total", "mean"),
            avg_spread_abs=("actual_margin", lambda s: np.nanmean(np.abs(s))),
        )
        .reset_index()
    )
    out["relative_total_vs_first"] = out["avg_total"] / out["avg_total"].iloc[0]
    return out


def build_summary_text(
    bias_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    scoring_df: pd.DataFrame,
    provider: str,
) -> str:
    spread_bias = bias_df.loc[bias_df["metric"] == "spread", ["mean_delta", "std_delta"]].iloc[0]
    total_bias = bias_df.loc[bias_df["metric"] == "total", ["mean_delta", "std_delta"]].iloc[0]
    largest_gap = calib_df.iloc[calib_df["calibration_gap"].abs().idxmax()] if not calib_df.empty else None

    lines = [
        f"Calibration summary for provider '{provider}':",
        "",
        f"- Spread bias: mean {spread_bias['mean_delta']:.2f} pts (model - market), std {spread_bias['std_delta']:.2f}.",
        f"- Total bias: mean {total_bias['mean_delta']:.2f} pts (model - market), std {total_bias['std_delta']:.2f}.",
    ]
    if largest_gap is not None:
        lines.append(
            f"- Largest win-probability gap in bucket {largest_gap['bucket']}: "
            f"pred {largest_gap['predicted_mean']:.2%} vs actual {largest_gap['actual_mean']:.2%} "
            f"(gap {largest_gap['calibration_gap']:+.2%})."
        )
    if len(scoring_df) > 1:
        latest = scoring_df.iloc[-1]
        baseline = scoring_df.iloc[0]
        lines.append(
            f"- Average total moved from {baseline['avg_total']:.1f} ({int(baseline['season'])}) "
            f"to {latest['avg_total']:.1f} ({int(latest['season'])}); "
            f"relative change {latest['relative_total_vs_first']:.3f}."
        )
    lines.extend(
        [
            "",
            "Recommended quick tweaks:",
            f"* Apply a spread recenter of {-spread_bias['mean_delta']:.2f} (add to model spreads).",
            f"* Apply a total recenter of {-total_bias['mean_delta']:.2f}.",
            "* Adjust win-prob sigma so predicted probabilities move toward actual buckets.",
            "* Update scoring priors using `relative_total_vs_first` ratios above.",
            "",
            "Note: use ROI primarily for directionality; thresholds stay unchanged.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise RuntimeError("CFBD API key required (set CFBD_API_KEY or pass --api-key).")

    years = list(range(args.start_year, args.end_year + 1))
    df, _ = collect_rows(
        years,
        args.provider,
        api_key=args.api_key,
        odds_api_dir=args.odds_api_dir,
        bookmaker=args.bookmaker,
    )
    payload = df[
        [
            "season",
            "provider",
            "model_spread",
            "market_spread",
            "model_total",
            "market_total",
            "model_home_win_prob",
            "market_home_win_prob",
            "actual_margin",
            "actual_total",
        ]
    ].copy()

    bias_df = compute_bias(payload)
    bias_path = Path(f"{args.out}_bias.csv")
    bias_df.to_csv(bias_path, index=False)

    calib_df = probability_calibration(payload, args.prob_bins)
    calib_path = Path(f"{args.out}_prob_calibration.csv")
    calib_df.to_csv(calib_path, index=False)

    scoring_df = scoring_environment(payload)
    scoring_path = Path(f"{args.out}_season_scoring.csv")
    scoring_df.to_csv(scoring_path, index=False)

    summary_text = build_summary_text(bias_df, calib_df, scoring_df, args.provider)
    summary_path = Path(f"{args.out}_summary.txt")
    summary_path.write_text(summary_text)

    print(f"Wrote bias metrics to {bias_path}")
    print(f"Wrote probability calibration to {calib_path}")
    print(f"Wrote scoring environment stats to {scoring_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
