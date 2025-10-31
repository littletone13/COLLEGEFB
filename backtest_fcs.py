"""Backtest the FCS PFF-based model using CFBD historical game results."""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

import fcs
import ncaa_stats
import oddslogic_loader
from cfb.config import load_config
from cfb.fcs_aliases import TEAM_NAME_ALIASES, normalize_label as _normalize_label
from cfb.market import edges as edge_utils

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
    parser.add_argument(
        "--oddslogic-dir",
        type=Path,
        help="Optional OddsLogic archive directory to supply multi-book closing lines.",
    )
    parser.add_argument(
        "--providers",
        type=str,
        help="Optional comma-separated sportsbook providers to filter when using OddsLogic data.",
    )
    return parser.parse_args()


def _parse_providers(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    providers = [part.strip() for part in value.split(",") if part.strip()]
    return providers or None


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
    closing_lookup: Optional[Dict[Tuple[dt.date, str, str], Dict[str, object]]] = None,
    *,
    spread_edge_min: float = 0.0,
    total_edge_min: float = 0.0,
    min_provider_count: int = 0,
) -> Tuple[List[Dict[str, object]], int]:
    _, book = fcs.build_rating_book(data_dir=data_dir, season_year=year)
    games = fetch_cfbd_games(year, api_key, season_type)
    if max_week is not None:
        games = [game for game in games if (game.get("week") or 0) <= max_week]
    if not games:
        return [], 0

    rows: List[Dict[str, object]] = []
    missing_closings = 0
    edge_config = edge_utils.EdgeFilterConfig(
        spread_edge_min=spread_edge_min,
        total_edge_min=total_edge_min,
        min_provider_count=min_provider_count,
    )
    total_wins = total_losses = total_pushes = 0
    total_roi = 0.0
    total_bets = 0
    for game in games:
        if game.get("completed") is not True:
            continue
        cfbd_home = game.get("homeTeam")
        cfbd_away = game.get("awayTeam")
        home = map_team(cfbd_home)
        away = map_team(cfbd_away)
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
        closing_spread = None
        closing_total = None
        closing_price = None
        closing_total_price = None
        closing_book = None
        closing_book_id = None
        kickoff_date = None
        provider_count = 0
        if closing_lookup is not None:
            kickoff_raw = game.get("startDate") or game.get("startTime")
            if kickoff_raw:
                try:
                    kickoff_date = pd.to_datetime(kickoff_raw).date()
                except Exception:  # pylint: disable=broad-except
                    kickoff_date = None
            if kickoff_date is not None:
                home_key = oddslogic_loader.normalize_label(cfbd_home or "")
                away_key = oddslogic_loader.normalize_label(cfbd_away or "")
                lookup_key = (kickoff_date, home_key, away_key)
                closing = closing_lookup.get(lookup_key)
                invert = False
                if closing is None:
                    alt_key = (kickoff_date, away_key, home_key)
                    closing = closing_lookup.get(alt_key)
                    if closing is not None:
                        invert = True
                if closing is None:
                    missing_closings += 1
                else:
                    providers_payload = closing.get("providers") or {}
                    if providers_payload:
                        provider_count = len(providers_payload)
                    elif closing.get("sportsbook_name"):
                        provider_count = 1
                    spread_raw = closing.get("spread_value")
                    if spread_raw is not None and not pd.isna(spread_raw):
                        closing_spread = float(spread_raw)
                        if invert:
                            closing_spread = -closing_spread
                    total_raw = closing.get("total_value")
                    if total_raw is not None and not pd.isna(total_raw):
                        closing_total = float(total_raw)
                    price_raw = closing.get("spread_price")
                    if price_raw is not None and not pd.isna(price_raw):
                        closing_price = float(price_raw)
                    total_price_raw = closing.get("total_price")
                    if total_price_raw is not None and not pd.isna(total_price_raw):
                        closing_total_price = float(total_price_raw)
                    closing_book = closing.get("sportsbook_name")
                    closing_book_id = closing.get("sportsbook_id")
            else:
                missing_closings += 1

        actual_home = float(actual_home)
        actual_away = float(actual_away)
        actual_margin = actual_home - actual_away
        actual_total = actual_home + actual_away
        home_flag = 1.0 if actual_margin > 0 else (0.0 if actual_margin < 0 else 0.5)
        pred_spread_value = pred.get("spread_home_minus_away") or pred.get("spread_team_one_minus_team_two")
        spread_edge_value = (
            pred_spread_value - closing_spread if closing_spread is not None else np.nan
        )
        total_edge_value = (
            pred["total_points"] - closing_total if closing_total is not None else np.nan
        )
        edge_allowed = edge_utils.allow_spread_bet(spread_edge_value, provider_count, edge_config)
        total_allowed = edge_utils.allow_total_bet(total_edge_value, provider_count, edge_config)

        if total_allowed:
            total_bets += 1
            pick_over = total_edge_value > 0
            margin = actual_total - (closing_total if closing_total is not None else 0.0)
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
                "pred_spread": pred_spread_value,
                "pred_total": pred["total_points"],
                "pred_home_win_prob": pred.get("home_win_prob") or pred.get("team_one_win_prob"),
                "spread_error": pred_spread_value - actual_margin if pred_spread_value is not None else np.nan,
                "total_error": pred["total_points"] - actual_total,
                "brier": (pred.get("home_win_prob") or pred.get("team_one_win_prob") - home_flag) ** 2,
                "closing_spread": closing_spread,
                "closing_spread_price": closing_price,
                "closing_total": closing_total,
                "closing_total_price": closing_total_price,
                "closing_sportsbook": closing_book,
                "closing_sportsbook_id": closing_book_id,
                "kickoff_date": kickoff_date,
                "market_provider_count": provider_count,
                "spread_edge": spread_edge_value,
                "total_edge": total_edge_value,
                "spread_edge_allowed": edge_allowed,
                "total_edge_allowed": total_allowed,
            }
        )
    return rows, missing_closings, (total_wins, total_losses, total_pushes, total_bets, total_roi)


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

    config = load_config()
    backtest_cfg = config.get("fcs", {}).get("backtest", {}) if isinstance(config.get("fcs"), dict) else {}
    spread_edge_min = float(backtest_cfg.get("spread_edge_min", 0.0))
    total_edge_min = float(backtest_cfg.get("total_edge_min", 0.0))
    min_provider_count = int(backtest_cfg.get("min_provider_count", 0))

    oddslogic_lookup = None
    providers = _parse_providers(args.providers)
    if args.oddslogic_dir:
        df_archive = oddslogic_loader.load_archive_dataframe(args.oddslogic_dir)
        class_filters = ["fcs", "fbs", "other"]
        oddslogic_lookup = oddslogic_loader.build_closing_lookup(
            df_archive,
            class_filters,
            providers=providers,
        )
        if not oddslogic_lookup:
            print("Warning: no OddsLogic records matched the requested configuration.")

    all_rows: List[Dict[str, object]] = []
    total_missing = 0
    total_metric_rows: list[tuple[int, int, int, int, float]] = []
    for season in range(start, end + 1):
        rows, missing, total_metrics = evaluate_season(
            season,
            data_dir,
            api_key,
            args.season_type,
            max_week=args.max_week,
            closing_lookup=oddslogic_lookup,
            spread_edge_min=spread_edge_min,
            total_edge_min=total_edge_min,
            min_provider_count=min_provider_count,
        )
        if not rows:
            print(f"No games processed for season {season}; check data availability.")
            continue
        all_rows.extend(rows)
        total_missing += missing
        total_metric_rows.append(total_metrics)
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

    if oddslogic_lookup is not None and total_missing:
        print(f"Warning: {total_missing} games missing OddsLogic closings across requested span.")

    overall = summary.assign(games=summary["games"].astype(int))
    total_games = df.shape[0]
    print(f"Overall across {total_games} games:")
    print(f"  Spread MAE: {df['spread_error'].abs().mean():.2f}")
    print(f"  Total MAE: {df['total_error'].abs().mean():.2f}")
    print(f"  Brier score: {df['brier'].mean():.4f}")

    if total_metric_rows:
        total_wins = sum(metrics[0] for metrics in total_metric_rows)
        total_losses = sum(metrics[1] for metrics in total_metric_rows)
        total_pushes = sum(metrics[2] for metrics in total_metric_rows)
        total_bets = sum(metrics[3] for metrics in total_metric_rows)
        total_roi_sum = sum(metrics[4] for metrics in total_metric_rows)
        total_roi_per_bet = (total_roi_sum / total_bets) if total_bets else 0.0
        print(
            f"  Totals record: {total_wins}-{total_losses}-{total_pushes} "
            f"(edge ≥ {total_edge_min}, providers ≥ {min_provider_count})"
        )
        print(f"  Totals ROI (per bet, -110 assumed): {total_roi_per_bet*100:0.2f}%")


if __name__ == "__main__":
    main()
