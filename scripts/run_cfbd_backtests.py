#!/usr/bin/env python3
"""Batch runner for FBS backtests using only CFBD API data."""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from backtest_fbs import BacktestResult, evaluate_season
from cfb.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, required=True, help="First season year (inclusive).")
    parser.add_argument("--end-year", type=int, required=True, help="Last season year (inclusive).")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["DraftKings"],
        help="Sportsbooks to test (exact CFBD provider names). Run multiple passes if needed.",
    )
    parser.add_argument("--out", type=Path, default=Path("out/fbs_backtests_cfbd.csv"), help="Output CSV path.")
    parser.add_argument(
        "--api-key",
        type=str,
        help="CFBD API key (defaults to CFBD_API_KEY env var).",
    )
    parser.add_argument("--max-week", type=int, help="Optional max week filter.")
    return parser.parse_args()


def load_edge_config() -> dict:
    config = load_config()
    if not isinstance(config.get("fbs"), dict):
        return {}
    backtest_cfg = config["fbs"].get("backtest", {})
    return {
        "spread_edge_min": float(backtest_cfg.get("spread_edge_min", 0.0)),
        "total_edge_min": float(backtest_cfg.get("total_edge_min", 0.0)),
        "min_provider_count": int(backtest_cfg.get("min_provider_count", 0)),
    }


def run_backtests(
    years: Sequence[int],
    providers: Sequence[str],
    *,
    api_key: str,
    max_week: Optional[int],
) -> pd.DataFrame:
    edge_cfg = load_edge_config()
    rows: List[dict] = []
    for year in years:
        for provider in providers:
            result: BacktestResult = evaluate_season(
                year,
                api_key=api_key,
                hist_path=None,
                provider=provider,
                max_week=max_week,
                oddslogic_lookup=None,
                spread_edge_min=edge_cfg.get("spread_edge_min", 0.0),
                min_provider_count=edge_cfg.get("min_provider_count", 0),
                total_edge_min=edge_cfg.get("total_edge_min", 0.0),
            )
            payload = asdict(result)
            payload["year"] = year
            payload["provider"] = provider
            rows.append(payload)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD API key not provided. Use --api-key or set CFBD_API_KEY env var.")

    years = range(args.start_year, args.end_year + 1)
    df = run_backtests(years, args.providers, api_key=api_key, max_week=args.max_week)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote CFBD-only FBS backtest summary to {args.out}")


if __name__ == "__main__":
    main()
