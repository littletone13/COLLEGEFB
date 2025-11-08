#!/usr/bin/env python3
"""Fetch football player prop markets from The Odds API."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from cfb.io import the_odds_api


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download football player props from The Odds API.")
    parser.add_argument("--sport", default="americanfootball_ncaaf", help="Sport key (default: americanfootball_ncaaf).")
    parser.add_argument(
        "--markets",
        nargs="+",
        default=["player_passing_yards", "player_rushing_yards", "player_receiving_yards"],
        help="Player prop markets to fetch (default: passing/rushing/receiving yards).",
    )
    parser.add_argument(
        "--bookmakers",
        nargs="+",
        default=["fanduel", "betonlineag"],
        help="Bookmakers to request (default: FanDuel + BetOnline).",
    )
    parser.add_argument("--regions", nargs="+", default=["us"], help="Regions to request (default: us).")
    parser.add_argument("--odds-format", default="american", choices=["american", "decimal"], help="Odds format.")
    parser.add_argument("--out", type=Path, default=Path("data/player_props/latest_player_props.ndjson"))
    parser.add_argument("--tags", nargs="*", help="Optional tags to embed in the metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "THE_ODDS_API_KEY" not in os.environ:
        raise SystemExit("Set THE_ODDS_API_KEY to call The Odds API.")

    payload = the_odds_api.fetch_current_odds(
        args.sport,
        regions=args.regions,
        markets=args.markets,
        bookmakers=args.bookmakers,
        odds_format=args.odds_format,
    )
    if not payload:
        print("No events returned by The Odds API.")
        return

    rows: list[dict] = []
    for event in payload:
        props = the_odds_api.normalise_player_props(event, markets=args.markets)
        if not props:
            continue
        rows.extend(props)

    if not rows:
        print("No player props matched the requested markets.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for entry in rows:
            # Serialize datetimes to ISO strings
            record = entry.copy()
            for key in ("commence_time", "last_update"):
                value = record.get(key)
                if isinstance(value, datetime):
                    record[key] = value.isoformat()
            record["_tags"] = args.tags or []
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(f"Saved {len(rows)} player props to {args.out}")


if __name__ == "__main__":
    main()
