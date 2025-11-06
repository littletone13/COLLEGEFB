#!/usr/bin/env python3
"""Pull FBS data from the CFBD API for backtesting pipelines.

The script downloads, stores, and validates:
* Game summaries
* Betting lines (split by provider)
* Weather reports
* Advanced team metrics (EPA/PPA/Win Probability)

Example:
    export CFBD_API_KEY=...
    python scripts/fetch_cfbd_fbs_data.py --start-year 2022 --end-year 2025 \
        --out-dir data/cfbd_fbs --providers draftkings circa caesars

Files are written as JSON under --out-dir with subdirectories for each dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import os

import requests

CFBD_BASE_URL = "https://api.collegefootballdata.com"
DEFAULT_SEASON_TYPES = ("regular", "postseason")
DEFAULT_SLEEP = 0.25  # seconds between requests to respect rate limits
METRIC_ENDPOINTS: Mapping[str, str] = {
    "ppa_teams": "/metrics/ppa/teams",
    "epa_teams": "/metrics/epa/teams",
    "wp_teams": "/metrics/wp/teams",
}


class FetchError(RuntimeError):
    """Raised when the CFBD API request fails."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, required=True, help="First season (inclusive).")
    parser.add_argument("--end-year", type=int, required=True, help="Last season (inclusive).")
    parser.add_argument(
        "--season-types",
        nargs="+",
        default=list(DEFAULT_SEASON_TYPES),
        help="Season types to pull (default: regular postseason).",
    )
    parser.add_argument(
        "--division",
        default="fbs",
        help="CFBD division filter. Defaults to fbs.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=[],
        help="Optional list of provider names to ensure are present in line data.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/cfbd"),
        help="Output directory for JSON artifacts.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help="Pause (seconds) between API calls to avoid throttling.",
    )
    parser.add_argument(
        "--api-key-env",
        default="CFBD_API_KEY",
        help="Environment variable holding the CFBD API key (default: CFBD_API_KEY).",
    )
    return parser.parse_args(argv)


def ensure_out_dirs(base: Path) -> Dict[str, Path]:
    subdirs = {
        "games": base / "games",
        "lines": base / "lines",
        "lines_providers": base / "lines_providers",
        "weather": base / "weather",
        "metrics": base / "metrics",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def get_api_key(env_var: str) -> str:
    api_key = os.environ.get(env_var)
    if not api_key:
        raise FetchError(f"Missing API key in environment variable {env_var}.")
    return api_key


def fetch_cfbd(path: str, *, api_key: str, params: Optional[Mapping[str, object]] = None) -> List[dict]:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{CFBD_BASE_URL}{path}", headers=headers, params=params or {}, timeout=60)
    if response.status_code == 401:
        raise FetchError("CFBD API rejected the key (401 Unauthorized).")
    if response.status_code == 403:
        raise FetchError("CFBD API returned 403 Forbidden. Check subscription tier.")
    if response.status_code == 429:
        raise FetchError("CFBD API rate-limited the request (429). Increase --sleep or retry later.")
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise FetchError(f"CFBD request failed: {exc} ({response.text})") from exc
    data = response.json()
    if isinstance(data, dict):
        return [data]
    return data


def normalise_provider(name: Optional[str]) -> str:
    if not name:
        return "unknown"
    return name.strip().lower().replace(" ", "_")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def collect_lines(
    raw_lines: Iterable[dict], *, year: int, season_type: str
) -> Tuple[List[dict], Dict[str, List[dict]]]:
    combined: List[dict] = []
    split: Dict[str, List[dict]] = defaultdict(list)

    for entry in raw_lines:
        game_id = entry.get("id")
        for market in entry.get("lines", []):
            provider = normalise_provider(market.get("provider"))
            record = {
                "season": year,
                "season_type": season_type,
                "game_id": game_id,
                "provider": provider,
                "spread": market.get("spread"),
                "over_under": market.get("overUnder"),
                "home_moneyline": market.get("homeMoneyline"),
                "away_moneyline": market.get("awayMoneyline"),
                "formatted_spread": market.get("formattedSpread"),
                "formatted_over_under": market.get("formattedOverUnder"),
                "last_updated": market.get("lastUpdated"),
            }
            combined.append(record)
            split[provider].append(record)
    return combined, split


def collect_summary(
    games: List[dict], lines_by_provider: Mapping[str, List[dict]], weather: List[dict]
) -> dict:
    fbs_games = [
        g
        for g in games
        if g.get("homeClassification") == "fbs" and g.get("awayClassification") == "fbs"
    ]
    unique_line_games = {
        provider: {entry["game_id"] for entry in records if entry.get("game_id") is not None}
        for provider, records in lines_by_provider.items()
    }
    all_line_games = set().union(*unique_line_games.values()) if unique_line_games else set()
    weather_games = {
        entry.get("gameId") or entry.get("id")
        for entry in weather
        if entry.get("gameId") or entry.get("id")
    }
    return {
        "total_games": len(games),
        "fbs_games": len(fbs_games),
        "lines_total_entries": sum(len(records) for records in lines_by_provider.values()),
        "lines_unique_games": len(all_line_games),
        "lines_by_provider": {provider: len(ids) for provider, ids in unique_line_games.items()},
        "weather_entries": len(weather),
        "weather_unique_games": len([gid for gid in weather_games if gid is not None]),
    }


def validate(
    season_key: str,
    summary: dict,
    *,
    expected_providers: Sequence[str],
) -> list[str]:
    issues: list[str] = []
    if summary["fbs_games"] == 0:
        issues.append(f"{season_key}: no FBS vs FBS games detected.")
    if summary["lines_unique_games"] == 0:
        issues.append(f"{season_key}: no betting lines found.")
    for provider in expected_providers:
        count = summary["lines_by_provider"].get(normalise_provider(provider), 0)
        if count == 0:
            issues.append(f"{season_key}: provider '{provider}' missing from lines dataset.")
    if summary["weather_unique_games"] and summary["fbs_games"]:
        ratio = summary["weather_unique_games"] / summary["fbs_games"]
        if ratio < 0.75:
            issues.append(
                f"{season_key}: weather coverage low ({summary['weather_unique_games']}/{summary['fbs_games']} games)."
            )
    return issues


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    api_key = get_api_key(args.api_key_env)
    out_dirs = ensure_out_dirs(args.out_dir)

    all_summaries: MutableMapping[str, dict] = {}
    issues: List[str] = []

    years = range(args.start_year, args.end_year + 1)
    for year in years:
        for season_type in args.season_types:
            params = {"year": year, "seasonType": season_type, "division": args.division}
            season_key = f"{year}_{season_type}"

            games = fetch_cfbd("/games", api_key=api_key, params=params)
            write_json(out_dirs["games"] / f"cfbd_games_{season_key}.json", games)
            time.sleep(args.sleep)

            lines = fetch_cfbd("/lines", api_key=api_key, params=params)
            combined_lines, lines_by_provider = collect_lines(lines, year=year, season_type=season_type)
            write_json(out_dirs["lines"] / f"cfbd_lines_{season_key}.json", combined_lines)
            provider_dir = out_dirs["lines_providers"] / season_key
            provider_dir.mkdir(parents=True, exist_ok=True)
            for provider, records in lines_by_provider.items():
                write_json(provider_dir / f"{provider}.json", records)
            time.sleep(args.sleep)

            weather = fetch_cfbd("/weather", api_key=api_key, params=params)
            write_json(out_dirs["weather"] / f"cfbd_weather_{season_key}.json", weather)
            time.sleep(args.sleep)

            for metric_name, endpoint in METRIC_ENDPOINTS.items():
                try:
                    metric_payload = fetch_cfbd(endpoint, api_key=api_key, params=params)
                except FetchError as exc:
                    issues.append(f"{season_key}: failed to fetch {metric_name}: {exc}")
                    continue
                write_json(
                    out_dirs["metrics"] / f"cfbd_{metric_name}_{season_key}.json",
                    metric_payload,
                )
                time.sleep(args.sleep)

            summary = collect_summary(games, lines_by_provider, weather)
            all_summaries[season_key] = summary
            issues.extend(validate(season_key, summary, expected_providers=args.providers))

    summary_path = args.out_dir / "cfbd_summary.json"
    write_json(summary_path, all_summaries)

    print(f"Wrote CFBD datasets to {args.out_dir}")
    print(f"Season summaries stored in {summary_path}")

    if issues:
        print("Validation issues detected:", file=sys.stderr)
        for msg in issues:
            print(f" - {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
