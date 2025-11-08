#!/usr/bin/env python3
"""
Archive FanDuel (or other bookmaker) open/close odds from The Odds API.

Example:
    export CFBD_API_KEY="..."
    export THE_ODDS_API_KEY="..."
    python scripts/fetch_the_odds_history.py --sport fbs --season 2025 --weeks 10 \
        --bookmaker fanduel --output data/lines/the_odds_api
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from cfb.io import the_odds_api
from cfb.names import normalize_team

try:
    from cfb.fcs_aliases import map_team as fcs_alias_map
except ImportError:  # pragma: no cover - alias helpers optional
    def fcs_alias_map(_: str) -> Optional[str]:
        return None


CFBD_API = "https://api.collegefootballdata.com"
DEFAULT_OPEN_WINDOW = timedelta(hours=12)
DEFAULT_CLOSE_WINDOW = timedelta(hours=2)


def _cfbd_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": "the-odds-history-fetcher/1.0 (+support@yourdomain.com)",
    }


def _call_cfbd(path: str, *, api_key: str, params: Optional[Dict[str, str]] = None) -> list[dict]:
    response = requests.get(
        CFBD_API + path,
        headers=_cfbd_headers(api_key),
        params=params or {},
        timeout=60,
    )
    if response.status_code == 401:
        raise RuntimeError("CFBD API rejected the key (401). Set CFBD_API_KEY.")
    response.raise_for_status()
    return response.json()


def _parse_weeks(token: str) -> List[int]:
    weeks: set[int] = set()
    for part in token.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start_i = int(start)
            end_i = int(end)
            weeks.update(range(min(start_i, end_i), max(start_i, end_i) + 1))
        else:
            weeks.add(int(part))
    return sorted(weeks)


def _normalize_label(label: Optional[str]) -> str:
    if not label:
        return ""
    text = normalize_team(label)
    text = re.sub(r"\(.*?\)", " ", text)
    text = text.replace("-", " ")
    text = text.replace("/", " ")
    text = text.replace(".", " ")
    text = text.replace("&", " AND ")
    text = text.replace("'", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize_label(label: Optional[str], classification: str) -> Tuple[str, ...]:
    cleaned = _normalize_label(label)
    tokens: list[str] = []
    seen: set[str] = set()
    for token in cleaned.split():
        upper = token.upper()
        if not upper or upper in seen:
            continue
        tokens.append(upper)
        seen.add(upper)
    if classification == "fcs":
        alias = fcs_alias_map(label or "")
        if alias:
            alias_clean = _normalize_label(alias)
            for token in alias_clean.split():
                upper = token.upper()
                if upper and upper not in seen:
                    tokens.append(upper)
                    seen.add(upper)
    return tuple(tokens)


def _tokens_match(candidate: Tuple[str, ...], target: Tuple[str, ...]) -> bool:
    if not candidate or not target:
        return False
    candidate_set = set(candidate)
    target_set = set(target)
    if candidate_set <= target_set or target_set <= candidate_set:
        return True
    candidate_str = " ".join(candidate)
    target_str = " ".join(target)
    if candidate_str and target_str:
        if candidate_str in target_str or target_str in candidate_str:
            return True
        ratio = difflib.SequenceMatcher(None, candidate_str, target_str).ratio()
        if ratio >= 0.78:
            return True
    return False


@dataclass
class MarketTracker:
    open_snapshot: Optional[dict] = None
    close_snapshot: Optional[dict] = None

    def update(self, row: dict, timestamp: datetime) -> None:
        snapshot = row.copy()
        snapshot["timestamp"] = timestamp
        if self.open_snapshot is None:
            self.open_snapshot = snapshot
        self.close_snapshot = snapshot


def _adjust_row(row: dict, invert: bool) -> dict:
    if not invert:
        return row
    updated = row.copy()
    market = updated.get("market")
    if market == "spread":
        point = updated.get("point")
        if point is not None:
            updated["point"] = -point
        price_home = updated.get("price_home")
        price_away = updated.get("price_away")
        updated["price_home"], updated["price_away"] = price_away, price_home
    elif market == "moneyline":
        price_home = updated.get("price_home")
        price_away = updated.get("price_away")
        updated["price_home"], updated["price_away"] = price_away, price_home
    return updated


def _build_game_index(games: list[dict], classification: str) -> Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], dict]:
    index: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], dict] = {}
    for game in games:
        home_tokens = _tokenize_label(game.get("homeTeam"), classification)
        away_tokens = _tokenize_label(game.get("awayTeam"), classification)
        key = (home_tokens, away_tokens)
        index[key] = game
    return index


def _find_game(
    home_tokens: Tuple[str, ...],
    away_tokens: Tuple[str, ...],
    index: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], dict],
) -> Tuple[Optional[dict], bool]:
    direct = index.get((home_tokens, away_tokens))
    if direct:
        return direct, False
    swapped = index.get((away_tokens, home_tokens))
    if swapped:
        return swapped, True
    # Fuzzy fallback
    for (cand_home, cand_away), game in index.items():
        if _tokens_match(cand_home, home_tokens) and _tokens_match(cand_away, away_tokens):
            return game, False
        if _tokens_match(cand_home, away_tokens) and _tokens_match(cand_away, home_tokens):
            return game, True
    return None, False


def _collect_snapshots(
    sport_key: str,
    *,
    start: datetime,
    end: datetime,
    regions: Iterable[str],
    markets: Iterable[str],
    bookmakers: Iterable[str],
) -> List[dict]:
    snapshots: List[dict] = []
    for payload in the_odds_api.iter_history_snapshots(
        sport_key,
        start=start,
        end=end,
        regions=regions,
        markets=markets,
        bookmakers=bookmakers,
    ):
        snapshots.append(payload)
    return snapshots


def _collect_week_history(
    *,
    games: list[dict],
    sport_key: str,
    bookmaker: str,
    regions: Iterable[str],
    markets: Iterable[str],
    open_window: timedelta,
    close_window: timedelta,
) -> pd.DataFrame:
    if not games:
        return pd.DataFrame()

    classification = games[0].get("homeClassification", "fbs") or "fbs"
    index = _build_game_index(games, classification.lower())

    trackers: Dict[int, Dict[str, MarketTracker]] = defaultdict(lambda: defaultdict(MarketTracker))
    time_bounds: Dict[int, Tuple[datetime, datetime]] = {}
    for game in games:
        commence = the_odds_api._parse_datetime(game.get("startDate"))
        if not commence:
            continue
        time_bounds[game["id"]] = (
            commence - open_window,
            commence + close_window,
        )

    if not time_bounds:
        return pd.DataFrame()

    games_by_date: Dict[datetime.date, list[int]] = defaultdict(list)
    kickoff_map: Dict[int, datetime] = {}
    for game in games:
        game_id = game.get("id")
        commence = the_odds_api._parse_datetime(game.get("startDate"))
        if not commence or game_id is None:
            continue
        kickoff_map[game_id] = commence
        games_by_date[commence.date()].append(game_id)

    bookmaker_key = bookmaker.lower()
    for date_key, game_ids in games_by_date.items():
        if not game_ids:
            continue
        start_window = min(time_bounds[gid][0] for gid in game_ids)
        end_window = max(time_bounds[gid][1] for gid in game_ids)
        snapshots = _collect_snapshots(
            sport_key,
            start=start_window,
            end=end_window,
            regions=regions,
            markets=markets,
            bookmakers=[bookmaker],
        )
        active_ids = set(game_ids)
        for payload in snapshots:
            snapshot_ts = the_odds_api._parse_datetime(payload.get("timestamp"))
            if not snapshot_ts or not (start_window <= snapshot_ts <= end_window):
                continue
            data = payload.get("data") or []
            for event in data:
                home_tokens = _tokenize_label(event.get("home_team"), classification)
                away_tokens = _tokenize_label(event.get("away_team"), classification)
                if not home_tokens or not away_tokens:
                    continue
                game, invert = _find_game(home_tokens, away_tokens, index)
                if not game:
                    continue
                game_id = game.get("id")
                if game_id not in active_ids:
                    continue
                bounds = time_bounds.get(game_id)
                if not bounds:
                    continue
                start_bound, end_bound = bounds
                if not (start_bound <= snapshot_ts <= end_bound):
                    continue
                rows = the_odds_api.normalise_prices(event)
                for row in rows:
                    row_bookmaker = str(row.get("bookmaker") or "").lower()
                    if row_bookmaker != bookmaker_key:
                        continue
                    adjusted = _adjust_row(row, invert)
                    market = adjusted.get("market")
                    if not market:
                        continue
                    tracker = trackers[game_id][market]
                    tracker.update(adjusted, snapshot_ts)

    records: List[dict] = []
    for game in games:
        game_id = game.get("id")
        commence = the_odds_api._parse_datetime(game.get("startDate"))
        market_map = trackers.get(game_id, {})
        for market, tracker in market_map.items():
            record = {
                "season": game.get("season"),
                "week": game.get("week"),
                "season_type": game.get("seasonType"),
                "classification": game.get("homeClassification"),
                "game_id": game_id,
                "home_team": game.get("homeTeam"),
                "away_team": game.get("awayTeam"),
                "commence_time": commence.isoformat() if commence else None,
                "bookmaker": bookmaker,
                "market": market,
            }
            open_snapshot = tracker.open_snapshot or {}
            close_snapshot = tracker.close_snapshot or {}
            record.update(
                {
                    "open_timestamp": open_snapshot.get("timestamp").isoformat() if open_snapshot else None,
                    "close_timestamp": close_snapshot.get("timestamp").isoformat() if close_snapshot else None,
                }
            )
            if market == "spread":
                record.update(
                    {
                        "open_point": open_snapshot.get("point"),
                        "open_price_home": open_snapshot.get("price_home"),
                        "open_price_away": open_snapshot.get("price_away"),
                        "close_point": close_snapshot.get("point"),
                        "close_price_home": close_snapshot.get("price_home"),
                        "close_price_away": close_snapshot.get("price_away"),
                    }
                )
            elif market == "total":
                record.update(
                    {
                        "open_point": open_snapshot.get("point"),
                        "open_price_over": open_snapshot.get("price_over"),
                        "open_price_under": open_snapshot.get("price_under"),
                        "close_point": close_snapshot.get("point"),
                        "close_price_over": close_snapshot.get("price_over"),
                        "close_price_under": close_snapshot.get("price_under"),
                    }
                )
            elif market == "moneyline":
                record.update(
                    {
                        "open_price_home": open_snapshot.get("price_home"),
                        "open_price_away": open_snapshot.get("price_away"),
                        "open_price_draw": open_snapshot.get("price_draw"),
                        "close_price_home": close_snapshot.get("price_home"),
                        "close_price_away": close_snapshot.get("price_away"),
                        "close_price_draw": close_snapshot.get("price_draw"),
                    }
                )
            records.append(record)
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch The Odds API odds-history snapshots.")
    parser.add_argument("--sport", choices=["fbs", "fcs"], required=True, help="Subdivision to archive.")
    parser.add_argument("--season", type=int, required=True, help="Target season year.")
    parser.add_argument("--weeks", required=True, help="Comma-separated weeks or ranges (e.g. 1-5,7,8).")
    parser.add_argument(
        "--bookmaker",
        default="fanduel",
        help="Comma-separated bookmaker keys (default: fanduel).",
    )
    parser.add_argument(
        "--markets",
        default="spreads,totals,h2h",
        help="Comma-separated markets to request (default: spreads,totals,h2h).",
    )
    parser.add_argument("--regions", default="us,us2", help="Comma-separated regions (default: us,us2).")
    parser.add_argument(
        "--open-window",
        type=float,
        default=12.0,
        help="Number of hours before kickoff to start sampling (default: 12).",
    )
    parser.add_argument(
        "--close-window",
        type=float,
        default=2.0,
        help="Number of hours after kickoff to continue sampling (default: 2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/lines/the_odds_api"),
        help="Output directory for archived CSVs (default: data/lines/the_odds_api).",
    )
    args = parser.parse_args()

    cfbd_key = os.environ.get("CFBD_API_KEY")
    if not cfbd_key:
        raise SystemExit("CFBD_API_KEY must be set to fetch schedules for alignment.")
    if not os.environ.get("THE_ODDS_API_KEY"):
        raise SystemExit("THE_ODDS_API_KEY must be set for The Odds API requests.")

    sport_key = "americanfootball_ncaaf"
    classification = "fbs" if args.sport == "fbs" else "fcs"
    weeks = _parse_weeks(args.weeks)
    regions = [token.strip() for token in args.regions.split(",") if token.strip()]
    markets = [token.strip() for token in args.markets.split(",") if token.strip()]
    bookmakers = [token.strip() for token in args.bookmaker.split(",") if token.strip()]
    if not bookmakers:
        raise SystemExit("At least one bookmaker key must be provided.")

    open_window = timedelta(hours=args.open_window)
    close_window = timedelta(hours=args.close_window)

    archive_rows: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    for week in weeks:
        games = _call_cfbd(
            "/games",
            api_key=cfbd_key,
            params={
                "year": args.season,
                "week": week,
                "seasonType": "regular",
                "classification": classification,
            },
        )
        if not games:
            print(f"[warn] No CFBD games for week {week}, skipping.")
            continue
        for bookmaker in bookmakers:
            try:
                df_week = _collect_week_history(
                    games=games,
                    sport_key=sport_key,
                    bookmaker=bookmaker,
                    regions=regions,
                    markets=markets,
                    open_window=open_window,
                    close_window=close_window,
                )
            except the_odds_api.TheOddsAPIError as exc:
                message = str(exc)
                if "401" in message:
                    raise SystemExit(
                        f"The Odds API returned 401 for {args.sport.upper()} odds-history (bookmaker '{bookmaker}'). "
                        "Verify that historical access is enabled for this sport on your account."
                    ) from exc
                raise
            if df_week.empty:
                print(f"[warn] No odds history captured for week {week} ({bookmaker}).")
                continue
            output_dir = args.output / f"season={args.season}" / f"week={week:02d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{bookmaker}_{classification}.csv"
            df_week.to_csv(output_path, index=False)
            print(f"[info] Stored {len(df_week)} rows to {output_path}")
            usage = the_odds_api.get_last_usage()
            if usage:
                print(
                    f"[info] Usage after week {week} ({bookmaker}): requests remaining={usage.get('requests_remaining', '?')}, "
                    f"token balance={usage.get('token_balance', '?')}"
                )
            archive_rows[bookmaker].append(df_week)

    for bookmaker, frames in archive_rows.items():
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        summary_path = args.output / f"season={args.season}" / f"{bookmaker}_{classification}_summary.csv"
        combined.to_csv(summary_path, index=False)
        print(f"[info] Wrote combined summary ({len(combined)} rows) to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
