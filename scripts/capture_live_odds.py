#!/usr/bin/env python3
"""Capture live odds snapshots from The Odds API for use in cron jobs."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

import fbs
import fcs
from cfb.io import the_odds_api

# The Odds API groups both subdivisions under the same sport key when the account
# only has access to `americanfootball_ncaaf`. Use that key for both and let the
# CFBD schedule filter drop games from the other classification.
SPORT_KEYS = {
    "fbs": "americanfootball_ncaaf",
    "fcs": "americanfootball_ncaaf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grab current odds snapshots for a given week.")
    parser.add_argument("--sport", choices=["fbs", "fcs"], required=True, help="Subdivision to capture.")
    parser.add_argument("--year", type=int, required=True, help="Season year.")
    parser.add_argument("--week", type=int, required=True, help="CFBD week to match.")
    parser.add_argument("--season-type", default="regular", help="Season type (default: regular).")
    parser.add_argument(
        "--regions",
        nargs="*",
        default=None,
        help="Regions to request (default: use config defaults).",
    )
    parser.add_argument(
        "--markets",
        nargs="*",
        default=None,
        help="Markets to request (default: spreads totals h2h).",
    )
    parser.add_argument(
        "--bookmakers",
        nargs="*",
        default=None,
        help="Optional explicit bookmaker filter sent to the API (default: all).",
    )
    parser.add_argument("--days-from", type=int, help="Limit odds to events within this many days.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/lines/live"),
        help="Destination directory for snapshots (default: data/lines/live).",
    )
    parser.add_argument(
        "--all-providers",
        action="store_true",
        help="Do not filter bookmakers (capture every provider returned by The Odds API).",
    )
    return parser.parse_args()


def _provider_filter(sport: str, *, disable: bool) -> Optional[set[str]]:
    if disable:
        return None
    if sport == "fbs":
        return {_normalize(name) for name in fbs.FBS_MARKET_PROVIDERS}
    config = fcs.CONFIG.get("market", {}) if isinstance(fcs.CONFIG, dict) else {}
    providers = config.get("providers") if isinstance(config.get("providers"), list) else []
    return {_normalize(name) for name in providers}


def _normalize(value: Optional[str]) -> str:
    return str(value or "").strip().lower()


def _build_cfbd_index(
    games: Iterable[dict],
    sport: str,
) -> tuple[Iterable[dict] | dict[int, dict], dict[int, dict]]:
    if sport == "fcs":
        tokenizer = fcs._tokenize_match_label
        cfbd_entries: list[dict] = []
    else:
        tokenizer = fbs._tokenize_match_label
        cfbd_entries = {}
    lookup: dict[int, dict] = {}
    for game in games:
        game_id = game.get("id")
        if not game_id:
            continue
        kickoff = game.get("startDate")
        kickoff_ts = None
        if kickoff:
            kickoff_ts = pd.to_datetime(kickoff, utc=True).to_pydatetime()
        meta = {
            "home_tokens": tokenizer(game.get("homeTeam")),
            "away_tokens": tokenizer(game.get("awayTeam")),
            "kickoff": kickoff_ts,
        }
        if sport == "fcs":
            meta["record"] = game
            cfbd_entries.append(meta)
        else:
            meta["record"] = game
            cfbd_entries[game_id] = meta
        lookup[game_id] = {
            "home_team": game.get("homeTeam"),
            "away_team": game.get("awayTeam"),
        }
    return cfbd_entries, lookup


def _coerce_float(value: object) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _rows_from_cfbd_lines(
    args: argparse.Namespace,
    client: fbs.CFBDClient,
) -> list[dict]:
    params: Dict[str, object] = {"year": args.year, "classification": "fcs"}
    if args.season_type:
        params["seasonType"] = args.season_type
    if args.week is not None:
        params["week"] = args.week
    entries = client.get("/lines", **params)
    timestamp = dt.datetime.utcnow().replace(microsecond=0, tzinfo=dt.timezone.utc)
    iso_ts = timestamp.isoformat().replace("+00:00", "Z")
    rows: list[dict] = []
    for entry in entries:
        if entry.get("homeClassification") != "fcs" or entry.get("awayClassification") != "fcs":
            continue
        game_id = entry.get("id")
        home_team = entry.get("homeTeam")
        away_team = entry.get("awayTeam")
        commence = entry.get("startDate")
        for line in entry.get("lines") or []:
            provider = str(line.get("provider") or "").strip()
            if not provider:
                continue
            spread = _coerce_float(line.get("spread"))
            total = _coerce_float(line.get("overUnder"))
            last_updated = line.get("lastUpdated") or line.get("updated")
            if spread is not None:
                rows.append(
                    {
                        "captured_at": iso_ts,
                        "game_id": game_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": commence,
                        "bookmaker": provider,
                        "market": "spread",
                        "point_home": -spread,
                        "point_away": spread,
                        "price_home": None,
                        "price_away": None,
                        "price_draw": None,
                        "last_update": last_updated,
                        "source": "CFBD",
                    }
                )
            if total is not None:
                rows.append(
                    {
                        "captured_at": iso_ts,
                        "game_id": game_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": commence,
                        "bookmaker": provider,
                        "market": "total",
                        "point_home": total,
                        "point_away": total,
                        "price_home": None,
                        "price_away": None,
                        "price_draw": None,
                        "last_update": last_updated,
                        "source": "CFBD",
                    }
                )
    return rows


def capture_snapshot(args: argparse.Namespace) -> Path:
    cfbd_api_key = os.environ.get("CFBD_API_KEY")
    if not cfbd_api_key:
        raise RuntimeError("CFBD_API_KEY must be set in the environment.")
    cfbd_client = fbs.CFBDClient(cfbd_api_key)
    params = {
        "year": args.year,
        "week": args.week,
        "seasonType": args.season_type,
        "classification": args.sport,
    }
    games = [game for game in cfbd_client.get("/games", **params) if game.get("week") == args.week]
    if not games:
        raise RuntimeError(f"No {args.sport.upper()} games found for week {args.week}.")
    if args.sport == "fcs":
        rows = _rows_from_cfbd_lines(args, cfbd_client)
    else:
        cfbd_index, lookup = _build_cfbd_index(games, args.sport)
        regions = args.regions or tuple(fbs.THE_ODDS_API_REGIONS)
        markets = args.markets or ("spreads", "totals", "h2h")
        bookmaker_filter = args.bookmakers or None
        sport_key = SPORT_KEYS[args.sport]
        odds_events = the_odds_api.fetch_current_odds(
            sport_key,
            regions=regions,
            markets=markets,
            bookmakers=bookmaker_filter,
            odds_format="american",
            days_from=args.days_from,
        )
        provider_filter = None
        rows = []
        timestamp = dt.datetime.utcnow().replace(microsecond=0, tzinfo=dt.timezone.utc)
        iso_ts = timestamp.isoformat().replace("+00:00", "Z")
        for event in odds_events:
            normalized_rows = the_odds_api.normalise_prices(event)
            if not normalized_rows:
                continue
            home_tokens = fbs._tokenize_match_label(event.get("home_team"))
            away_tokens = fbs._tokenize_match_label(event.get("away_team"))
            commence = normalized_rows[0].get("commence_time")
            match = fbs._resolve_odds_api_match(cfbd_index, home_tokens, away_tokens, commence)
            if not match:
                continue
            game_id, invert = match
            for row in normalized_rows:
                provider_key = fbs._format_bookmaker_name(row.get("bookmaker"))
                normalized_provider = _normalize(provider_key)
                if provider_filter and normalized_provider not in provider_filter:
                    continue
                market = row.get("market")
                entry = {
                    "captured_at": iso_ts,
                    "game_id": game_id,
                    "home_team": lookup[game_id]["home_team"],
                    "away_team": lookup[game_id]["away_team"],
                    "commence_time": row.get("commence_time"),
                    "bookmaker": provider_key,
                    "market": market,
                    "point_home": row.get("point_home"),
                    "point_away": row.get("point_away"),
                    "price_home": row.get("price_home"),
                    "price_away": row.get("price_away"),
                    "price_draw": row.get("price_draw"),
                    "last_update": row.get("last_update"),
                    "source": "TheOddsAPI",
                }
                if invert and market == "spread":
                    entry["point_home"], entry["point_away"] = (
                        -entry["point_home"] if entry["point_home"] is not None else None,
                        -entry["point_away"] if entry["point_away"] is not None else None,
                    )
                    entry["price_home"], entry["price_away"] = entry["price_away"], entry["price_home"]
                rows.append(entry)
    if not rows:
        raise RuntimeError("No odds matched the requested games.")
    df = pd.DataFrame(rows)
    out_dir = (
        Path(args.output)
        / f"sport={args.sport}"
        / f"season={args.year}"
        / f"week={args.week:02d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"snapshot_{timestamp.strftime('%Y%m%dT%H%M%S')}Z.csv"
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    args = parse_args()
    path = capture_snapshot(args)
    print(f"[ok] Stored snapshot to {path}")


if __name__ == "__main__":
    main()
