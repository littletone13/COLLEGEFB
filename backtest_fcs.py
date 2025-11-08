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
    parser.add_argument(
        "--odds-api-dir",
        type=Path,
        help="Optional path to archived The Odds API history (FanDuel, etc.).",
    )
    parser.add_argument(
        "--bookmaker",
        type=str,
        default="fanduel",
        help="Bookmaker key to use with The Odds API history (default fanduel).",
    )
    return parser.parse_args()


def _parse_providers(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    providers = [part.strip() for part in value.split(",") if part.strip()]
    return providers or None


def build_odds_api_lookup(
    base_dir: Optional[Path],
    seasons: Iterable[int],
    bookmaker: str,
) -> Dict[Tuple[dt.date, str, str], Dict[str, object]]:
    if base_dir is None:
        return {}
    base_dir = base_dir.expanduser()
    if not base_dir.exists():
        return {}
    mapping: Dict[Tuple[dt.date, str, str], Dict[str, object]] = {}
    bookmaker_key = bookmaker.strip().lower()
    for season in seasons:
        season_dir = base_dir / f"season={season}"
        if not season_dir.exists():
            continue
        for csv_path in season_dir.glob("week=*/*"):
            if bookmaker_key not in csv_path.name.lower():
                continue
            try:
                df = pd.read_csv(csv_path)
            except FileNotFoundError:
                continue
            if df.empty:
                continue
            for row in df.itertuples(index=False):
                commence = pd.to_datetime(getattr(row, "commence_time", None), utc=True, errors="coerce")
                if pd.isna(commence):
                    continue
                kickoff_date = commence.date()
                home_team = getattr(row, "home_team", "")
                away_team = getattr(row, "away_team", "")
                home_key = oddslogic_loader.normalize_label(home_team or "")
                away_key = oddslogic_loader.normalize_label(away_team or "")
                key = (kickoff_date, home_key, away_key)
                entry = mapping.setdefault(
                    key,
                    {
                        "providers": {},
                        "sportsbook_name": bookmaker,
                        "sportsbook_id": None,
                        "spread_value": None,
                        "total_value": None,
                        "spread_price": None,
                        "total_price": None,
                    },
                )
                providers = entry.setdefault("providers", {})
                payload = providers.setdefault(bookmaker, {})
                market = getattr(row, "market", "")
                if market == "spread":
                    point = getattr(row, "close_point", None)
                    if point is not None and not pd.isna(point):
                        entry["spread_value"] = float(point)
                        payload["spread_value"] = float(point)
                    price_home = getattr(row, "close_price_home", None)
                    if price_home is not None and not pd.isna(price_home):
                        entry["spread_price"] = float(price_home)
                        payload["spread_price"] = float(price_home)
                elif market == "total":
                    point = getattr(row, "close_point", None)
                    if point is not None and not pd.isna(point):
                        entry["total_value"] = float(point)
                        payload["total_value"] = float(point)
                    price_over = getattr(row, "close_price_over", None)
                    if price_over is not None and not pd.isna(price_over):
                        entry["total_price"] = float(price_over)
                        payload["total_price"] = float(price_over)
                elif market == "moneyline":
                    price_home = getattr(row, "close_price_home", None)
                    price_away = getattr(row, "close_price_away", None)
                    if price_home is not None and not pd.isna(price_home):
                        payload["home_moneyline"] = float(price_home)
                    if price_away is not None and not pd.isna(price_away):
                        payload["away_moneyline"] = float(price_away)
    return mapping


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
    pure_spread_wins = pure_spread_losses = pure_spread_pushes = 0
    pure_spread_roi = 0.0
    pure_spread_bets = 0
    blended_spread_wins = blended_spread_losses = blended_spread_pushes = 0
    blended_spread_roi = 0.0
    blended_spread_bets = 0
    pure_total_wins = pure_total_losses = pure_total_pushes = 0
    pure_total_roi = 0.0
    pure_total_bets = 0
    blended_total_wins = blended_total_losses = blended_total_pushes = 0
    blended_total_roi = 0.0
    blended_total_bets = 0
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
        provider_lines: Dict[str, dict] = {}
        market_payload: Optional[dict] = None
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
                        provider_lines = providers_payload
                    elif closing.get("sportsbook_name"):
                        provider_count = 1
                        provider_name = closing.get("sportsbook_name") or "provider"
                        provider_lines = {
                            provider_name: {
                                "spread_value": closing.get("spread_value"),
                                "total_value": closing.get("total_value"),
                            }
                        }
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
                    if invert and provider_lines:
                        adjusted: Dict[str, dict] = {}
                        for name, payload in provider_lines.items():
                            payload_copy = dict(payload)
                            spread_val = payload_copy.get("spread_value")
                            if spread_val is not None and not pd.isna(spread_val):
                                payload_copy["spread_value"] = -float(spread_val)
                            adjusted[name] = payload_copy
                        provider_lines = adjusted
            else:
                missing_closings += 1

        actual_home = float(actual_home)
        actual_away = float(actual_away)
        actual_margin = actual_home - actual_away
        actual_total = actual_home + actual_away
        home_flag = 1.0 if actual_margin > 0 else (0.0 if actual_margin < 0 else 0.5)

        if closing_spread is not None or closing_total is not None or provider_lines:
            market_payload = {
                "spread": closing_spread,
                "total": closing_total,
                "providers": sorted(provider_lines.keys()),
                "provider_lines": provider_lines,
            }

        pure_pred = pred.copy()
        blended_pred = fcs.apply_market_prior(pure_pred.copy(), market_payload)

        pure_spread_value = pure_pred.get("spread_home_minus_away") or pure_pred.get("spread_team_one_minus_team_two")
        blended_spread_value = (
            blended_pred.get("spread_home_minus_away") or blended_pred.get("spread_team_one_minus_team_two")
        )
        pure_total_value = pure_pred["total_points"]
        blended_total_value = blended_pred["total_points"]
        pure_prob = pure_pred.get("home_win_prob") or pure_pred.get("team_one_win_prob")
        blended_prob = blended_pred.get("home_win_prob") or blended_pred.get("team_one_win_prob")

        pure_spread_edge = (
            pure_spread_value - closing_spread if closing_spread is not None and pure_spread_value is not None else np.nan
        )
        blended_spread_edge = (
            blended_spread_value - closing_spread
            if closing_spread is not None and blended_spread_value is not None
            else np.nan
        )
        pure_total_edge = (
            pure_total_value - closing_total if closing_total is not None else np.nan
        )
        blended_total_edge = (
            blended_total_value - closing_total if closing_total is not None else np.nan
        )

        spread_allowed_pure = edge_utils.allow_spread_bet(pure_spread_edge, provider_count, edge_config)
        spread_allowed_blended = edge_utils.allow_spread_bet(blended_spread_edge, provider_count, edge_config)
        total_allowed_pure = edge_utils.allow_total_bet(pure_total_edge, provider_count, edge_config)
        total_allowed_blended = edge_utils.allow_total_bet(blended_total_edge, provider_count, edge_config)

        if spread_allowed_pure and closing_spread is not None and pure_spread_value is not None:
            pure_spread_bets += 1
            pick_home = pure_spread_edge > 0
            cover_margin = actual_margin + closing_spread
            if pick_home:
                if cover_margin > 0:
                    pure_spread_wins += 1
                    pure_spread_roi += 0.909
                elif cover_margin < 0:
                    pure_spread_losses += 1
                    pure_spread_roi -= 1.0
                else:
                    pure_spread_pushes += 1
            else:
                if cover_margin < 0:
                    pure_spread_wins += 1
                    pure_spread_roi += 0.909
                elif cover_margin > 0:
                    pure_spread_losses += 1
                    pure_spread_roi -= 1.0
                else:
                    pure_spread_pushes += 1

        if spread_allowed_blended and closing_spread is not None and blended_spread_value is not None:
            blended_spread_bets += 1
            pick_home_blended = blended_spread_edge > 0
            cover_margin = actual_margin + closing_spread
            if pick_home_blended:
                if cover_margin > 0:
                    blended_spread_wins += 1
                    blended_spread_roi += 0.909
                elif cover_margin < 0:
                    blended_spread_losses += 1
                    blended_spread_roi -= 1.0
                else:
                    blended_spread_pushes += 1
            else:
                if cover_margin < 0:
                    blended_spread_wins += 1
                    blended_spread_roi += 0.909
                elif cover_margin > 0:
                    blended_spread_losses += 1
                    blended_spread_roi -= 1.0
                else:
                    blended_spread_pushes += 1

        if total_allowed_pure and closing_total is not None:
            pure_total_bets += 1
            pick_over = pure_total_edge > 0
            margin = actual_total - closing_total
            if pick_over:
                if margin > 0:
                    pure_total_wins += 1
                    pure_total_roi += 0.909
                elif margin < 0:
                    pure_total_losses += 1
                    pure_total_roi -= 1.0
                else:
                    pure_total_pushes += 1
            else:
                if margin < 0:
                    pure_total_wins += 1
                    pure_total_roi += 0.909
                elif margin > 0:
                    pure_total_losses += 1
                    pure_total_roi -= 1.0
                else:
                    pure_total_pushes += 1

        if total_allowed_blended and closing_total is not None:
            blended_total_bets += 1
            pick_over_blended = blended_total_edge > 0
            margin = actual_total - closing_total
            if pick_over_blended:
                if margin > 0:
                    blended_total_wins += 1
                    blended_total_roi += 0.909
                elif margin < 0:
                    blended_total_losses += 1
                    blended_total_roi -= 1.0
                else:
                    blended_total_pushes += 1
            else:
                if margin < 0:
                    blended_total_wins += 1
                    blended_total_roi += 0.909
                elif margin > 0:
                    blended_total_losses += 1
                    blended_total_roi -= 1.0
                else:
                    blended_total_pushes += 1

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
                "pred_home_points": pure_pred.get("home_points") or pure_pred.get("team_one_points"),
                "pred_away_points": pure_pred.get("away_points") or pure_pred.get("team_two_points"),
                "pred_home_points_blended": blended_pred.get("home_points") or blended_pred.get("team_one_points"),
                "pred_away_points_blended": blended_pred.get("away_points") or blended_pred.get("team_two_points"),
                "pred_spread": pure_spread_value,
                "pred_spread_blended": blended_spread_value,
                "pred_total": pure_total_value,
                "pred_total_blended": blended_total_value,
                "pred_home_win_prob": pure_prob,
                "pred_home_win_prob_blended": blended_prob,
                "spread_error": pure_spread_value - actual_margin if pure_spread_value is not None else np.nan,
                "spread_error_blended": blended_spread_value - actual_margin
                if blended_spread_value is not None
                else np.nan,
                "total_error": pure_total_value - actual_total,
                "total_error_blended": blended_total_value - actual_total,
                "brier": (pure_prob - home_flag) ** 2 if pure_prob is not None else np.nan,
                "brier_blended": (blended_prob - home_flag) ** 2 if blended_prob is not None else np.nan,
                "closing_spread": closing_spread,
                "closing_spread_price": closing_price,
                "closing_total": closing_total,
                "closing_total_price": closing_total_price,
                "closing_sportsbook": closing_book,
                "closing_sportsbook_id": closing_book_id,
                "kickoff_date": kickoff_date,
                "market_provider_count": provider_count,
                "market_providers": ", ".join(sorted(provider_lines.keys())) if provider_lines else None,
                "spread_edge": pure_spread_edge,
                "spread_edge_blended": blended_spread_edge,
                "total_edge": pure_total_edge,
                "total_edge_blended": blended_total_edge,
                "spread_edge_allowed": spread_allowed_pure,
                "spread_edge_allowed_blended": spread_allowed_blended,
                "total_edge_allowed": total_allowed_pure,
                "total_edge_allowed_blended": total_allowed_blended,
            }
        )
    metrics = {
        "pure_spread": (pure_spread_wins, pure_spread_losses, pure_spread_pushes, pure_spread_bets, pure_spread_roi),
        "blended_spread": (
            blended_spread_wins,
            blended_spread_losses,
            blended_spread_pushes,
            blended_spread_bets,
            blended_spread_roi,
        ),
        "pure_total": (pure_total_wins, pure_total_losses, pure_total_pushes, pure_total_bets, pure_total_roi),
        "blended_total": (
            blended_total_wins,
            blended_total_losses,
            blended_total_pushes,
            blended_total_bets,
            blended_total_roi,
        ),
    }
    return rows, missing_closings, metrics


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for season, group in df.groupby("season"):
        summaries.append(
            {
                "season": season,
                "games": len(group),
                "spread_mae": group["spread_error"].abs().mean(),
                "spread_mae_blended": group["spread_error_blended"].abs().mean(),
                "total_mae": group["total_error"].abs().mean(),
                "total_mae_blended": group["total_error_blended"].abs().mean(),
                "brier": group["brier"].mean(),
                "brier_blended": group["brier_blended"].mean(),
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

    closing_lookup: Optional[Dict[Tuple[dt.date, str, str], Dict[str, object]]] = None
    odds_api_lookup = build_odds_api_lookup(
        args.odds_api_dir,
        seasons=range(start, end + 1),
        bookmaker=args.bookmaker,
    ) if args.odds_api_dir else {}
    if odds_api_lookup:
        closing_lookup = odds_api_lookup
    if oddslogic_lookup:
        if closing_lookup is None:
            closing_lookup = oddslogic_lookup
        else:
            merged = dict(closing_lookup)
            for key, value in oddslogic_lookup.items():
                merged.setdefault(key, value)
            closing_lookup = merged

    all_rows: List[Dict[str, object]] = []
    total_missing = 0
    metric_rows: List[Dict[str, tuple[int, int, int, int, float]]] = []
    for season in range(start, end + 1):
        rows, missing, metrics = evaluate_season(
            season,
            data_dir,
            api_key,
            args.season_type,
            max_week=args.max_week,
            closing_lookup=closing_lookup,
            spread_edge_min=spread_edge_min,
            total_edge_min=total_edge_min,
            min_provider_count=min_provider_count,
        )
        if not rows:
            print(f"No games processed for season {season}; check data availability.")
            continue
        all_rows.extend(rows)
        total_missing += missing
        metric_rows.append(metrics)
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

    if closing_lookup is not None and total_missing:
        print(f"Warning: {total_missing} games missing closing prices across requested span.")

    overall = summary.assign(games=summary["games"].astype(int))
    total_games = df.shape[0]
    print(f"Overall across {total_games} games:")
    print(f"  Spread MAE (pure): {df['spread_error'].abs().mean():.2f}")
    print(f"  Spread MAE (market-adjusted): {df['spread_error_blended'].abs().mean():.2f}")
    print(f"  Total MAE (pure): {df['total_error'].abs().mean():.2f}")
    print(f"  Total MAE (market-adjusted): {df['total_error_blended'].abs().mean():.2f}")
    print(f"  Brier (pure): {df['brier'].mean():.4f}")
    print(f"  Brier (market-adjusted): {df['brier_blended'].mean():.4f}")

    if metric_rows:
        def _aggregate(key: str) -> tuple[int, int, int, int, float]:
            wins = sum(entry[key][0] for entry in metric_rows)
            losses = sum(entry[key][1] for entry in metric_rows)
            pushes = sum(entry[key][2] for entry in metric_rows)
            bets = sum(entry[key][3] for entry in metric_rows)
            roi_sum = sum(entry[key][4] for entry in metric_rows)
            return wins, losses, pushes, bets, roi_sum

        ps_w, ps_l, ps_p, ps_b, ps_roi = _aggregate("pure_spread")
        bs_w, bs_l, bs_p, bs_b, bs_roi = _aggregate("blended_spread")
        pt_w, pt_l, pt_p, pt_b, pt_roi = _aggregate("pure_total")
        bt_w, bt_l, bt_p, bt_b, bt_roi = _aggregate("blended_total")

        def _roi(roi_sum: float, bets: int) -> float:
            return (roi_sum / bets) if bets else 0.0

        print(
            f"  Spread record (pure): {ps_w}-{ps_l}-{ps_p} "
            f"(edge ≥ {spread_edge_min}, providers ≥ {min_provider_count})"
        )
        print(f"  Spread ROI (pure, per bet, -110 assumed): {_roi(ps_roi, ps_b)*100:0.2f}%")
        print(f"  Spread bets (pure): {ps_b}")
        print(
            f"  Spread record (market-adjusted): {bs_w}-{bs_l}-{bs_p} "
            f"(edge ≥ {spread_edge_min}, providers ≥ {min_provider_count})"
        )
        print(f"  Spread ROI (market-adjusted, per bet, -110 assumed): {_roi(bs_roi, bs_b)*100:0.2f}%")
        print(f"  Spread bets (market-adjusted): {bs_b}")
        print(
            f"  Totals record (pure): {pt_w}-{pt_l}-{pt_p} "
            f"(edge ≥ {total_edge_min}, providers ≥ {min_provider_count})"
        )
        print(f"  Totals ROI (pure, per bet, -110 assumed): {_roi(pt_roi, pt_b)*100:0.2f}%")
        print(f"  Totals bets (pure): {pt_b}")
        print(
            f"  Totals record (market-adjusted): {bt_w}-{bt_l}-{bt_p} "
            f"(edge ≥ {total_edge_min}, providers ≥ {min_provider_count})"
        )
        print(f"  Totals ROI (market-adjusted, per bet, -110 assumed): {_roi(bt_roi, bt_b)*100:0.2f}%")
        print(f"  Totals bets (market-adjusted): {bt_b}")


if __name__ == "__main__":
    main()
