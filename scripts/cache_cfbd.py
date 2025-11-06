#!/usr/bin/env python3
"""Cache CFBD endpoints locally so the modelling pipeline can run offline."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import requests

BASE_URL = "https://api.collegefootballdata.com"
DEFAULT_OUT_DIR = Path("data/cache/cfbd")
USER_AGENT = "cfbd-cache/1.0 (+support@yourdomain.com)"


def _is_challenge(resp: requests.Response) -> bool:
    content_type = resp.headers.get("Content-Type", "").lower()
    if "text/html" in content_type and "cloudflare" in resp.text.lower():
        return True
    if resp.status_code in (403, 503):
        return True
    if resp.headers.get("cf-mitigated") == "challenge":
        return True
    return False


def _create_session(api_key: str, user_agent: str = USER_AGENT) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": user_agent,
        }
    )
    return session


def _safe_get(
    session: requests.Session,
    endpoint: str,
    *,
    params: Optional[dict] = None,
    retries: int = 6,
    base_backoff: float = 1.5,
) -> dict | list:
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=60)
        except requests.RequestException as exc:
            sleep = base_backoff * (2 ** attempt) + random.random()
            time.sleep(sleep)
            if attempt == retries - 1:
                raise RuntimeError(f"Request failed for {url}: {exc}") from exc
            continue
        if _is_challenge(resp):
            sleep = base_backoff * (2 ** attempt) + random.random()
            time.sleep(sleep)
            continue
        if resp.status_code >= 500:
            sleep = base_backoff * (2 ** attempt) + random.random()
            time.sleep(sleep)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Cloudflare challenge persisted for {url}")


def _save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    tmp.replace(path)


def cache_season(
    session: requests.Session,
    out_dir: Path,
    year: int,
    *,
    season_type: str = "regular",
    max_week: Optional[int] = None,
    include_plays: bool = True,
    include_game_adv: bool = True,
    include_team_adv: bool = True,
    classification: str = "fbs",
    force: bool = False,
) -> None:
    season_dir = out_dir / f"season_{year}"
    season_dir.mkdir(parents=True, exist_ok=True)

    # Team-level advanced stats
    if include_team_adv:
        team_path = season_dir / f"advanced_team_{season_type}.json"
        if force or not team_path.exists():
            data = _safe_get(
                session,
                "/stats/season/advanced",
                params={"year": year, "seasonType": season_type, "classification": classification},
            )
            _save_json(team_path, data)

    # Game-level advanced stats (per week)
    if include_game_adv:
        week = 1
        while True:
            if max_week is not None and week > max_week:
                break
            game_path = season_dir / f"advanced_game_{season_type}_wk{week:02d}.json"
            if not force and game_path.exists():
                week += 1
                continue
            params = {
                "year": year,
                "seasonType": season_type,
                "week": week,
                "classification": classification,
            }
            try:
                payload = _safe_get(session, "/stats/game/advanced", params=params)
            except RuntimeError:
                # If the API challenges here we skip the rest to avoid hammering.
                break
            if not payload:
                if max_week is None:
                    break
                week += 1
                continue
            _save_json(game_path, payload)
            week += 1

    # Play-by-play (per week)
    if include_plays:
        week = 1
        while True:
            if max_week is not None and week > max_week:
                break
            plays_path = season_dir / f"plays_{season_type}_wk{week:02d}.json"
            if not force and plays_path.exists():
                week += 1
                continue
            params = {
                "year": year,
                "seasonType": season_type,
                "week": week,
                "classification": classification,
            }
            try:
                payload = _safe_get(session, "/plays", params=params)
            except RuntimeError:
                break
            if not payload:
                if max_week is None:
                    break
                week += 1
                continue
            _save_json(plays_path, payload)
            week += 1


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache CFBD data locally.")
    parser.add_argument("--api-key", help="CFBD API key (defaults to CFBD_API_KEY env).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for cached files.")
    parser.add_argument("--start-year", type=int, default=2025, help="Most recent season to cache (default 2025).")
    parser.add_argument("--seasons", type=int, default=5, help="Number of seasons to cache counting backwards (default 5).")
    parser.add_argument("--season-type", default="regular", help="Season type to cache (default regular).")
    parser.add_argument("--max-week", type=int, help="Optional maximum week to fetch.")
    parser.add_argument("--skip-plays", action="store_true", help="Skip caching play-by-play.")
    parser.add_argument("--skip-game-advanced", action="store_true", help="Skip caching game-level advanced stats.")
    parser.add_argument("--skip-team-advanced", action="store_true", help="Skip caching team-level advanced stats.")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    api_key = args.api_key or getenv("CFBD_API_KEY", "")
    if not api_key:
        raise RuntimeError("CFBD API key required (pass --api-key or set CFBD_API_KEY).")

    session = _create_session(api_key)
    for offset in range(args.seasons):
        year = args.start_year - offset
        cache_season(
            session,
            args.out,
            year,
            season_type=args.season_type,
            max_week=args.max_week,
            include_plays=not args.skip_plays,
            include_game_adv=not args.skip_game_advanced,
            include_team_adv=not args.skip_team_advanced,
            force=args.force,
        )
        # brief pause between seasons to be gentle on the API
        time.sleep(1.0 + random.random())


if __name__ == "__main__":
    from os import getenv

    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
