"""Helpers for loading CFBD data from the local cache (and fetching when missing)."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

CACHE_ROOT = Path("data/cache/cfbd")
BASE_URL = "https://api.collegefootballdata.com"
USER_AGENT = "cfbd-cache-loader/1.0 (+support@yourdomain.com)"


def _season_dir(year: int) -> Path:
    return CACHE_ROOT / f"season_{year}"


def _load_json(path: Path) -> list | dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    tmp.replace(path)


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


def _is_challenge(resp: requests.Response) -> bool:
    content_type = resp.headers.get("Content-Type", "").lower()
    if "text/html" in content_type and "cloudflare" in resp.text.lower():
        return True
    if resp.status_code in (403, 503):
        return True
    if resp.headers.get("cf-mitigated") == "challenge":
        return True
    return False


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


def _ensure_cache(
    year: int,
    filename: str,
    *,
    endpoint: str,
    params: Optional[dict],
    api_key: Optional[str],
) -> Path:
    target = _season_dir(year) / filename
    if target.exists():
        return target
    if not api_key:
        raise FileNotFoundError(f"{target} missing and no API key supplied to fetch it.")
    session = _create_session(api_key)
    payload = _safe_get(session, endpoint, params=params)
    _save_json(target, payload)
    return target


def load_advanced_team(
    year: int,
    season_type: str = "regular",
    *,
    fetch_if_missing: bool = True,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    filename = f"advanced_team_{season_type}.json"
    path = _season_dir(year) / filename
    if not path.exists():
        if not fetch_if_missing:
            raise FileNotFoundError(path)
        path = _ensure_cache(
            year,
            filename,
            endpoint="/stats/season/advanced",
            params={"year": year, "seasonType": season_type, "classification": "fbs"},
            api_key=api_key or _env_key(),
        )
    data = _load_json(path)
    return pd.DataFrame(data)


def load_game_advanced(
    year: int,
    week: int,
    season_type: str = "regular",
    *,
    fetch_if_missing: bool = True,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    filename = f"advanced_game_{season_type}_wk{week:02d}.json"
    path = _season_dir(year) / filename
    if not path.exists():
        if not fetch_if_missing:
            raise FileNotFoundError(path)
        path = _ensure_cache(
            year,
            filename,
            endpoint="/stats/game/advanced",
            params={
                "year": year,
                "seasonType": season_type,
                "week": week,
                "classification": "fbs",
            },
            api_key=api_key or _env_key(),
        )
    data = _load_json(path)
    return pd.DataFrame(data)


def load_plays(
    year: int,
    week: int,
    *,
    season_type: str = "regular",
    classification: str = "fbs",
    fetch_if_missing: bool = True,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    filename = f"plays_{season_type}_wk{week:02d}.json"
    path = _season_dir(year) / filename
    if not path.exists():
        if not fetch_if_missing:
            raise FileNotFoundError(path)
        path = _ensure_cache(
            year,
            filename,
            endpoint="/plays",
            params={
                "year": year,
                "seasonType": season_type,
                "week": week,
                "classification": classification,
            },
            api_key=api_key or _env_key(),
        )
    data = _load_json(path)
    return pd.DataFrame(data)


def available_weeks(year: int, season_type: str = "regular") -> list[int]:
    season_dir = _season_dir(year)
    if not season_dir.exists():
        return []
    prefix = f"plays_{season_type}_wk"
    weeks: list[int] = []
    for path in season_dir.glob(f"{prefix}*.json"):
        token = path.stem.split("_wk")[-1]
        try:
            weeks.append(int(token))
        except ValueError:
            continue
    return sorted(set(weeks))


def _env_key() -> str:
    key = getenv("CFBD_API_KEY", "")
    if not key:
        raise RuntimeError("CFBD_API_KEY not set; required to fetch missing cache files.")
    return key


def configure_cache_root(path: Path) -> None:
    global CACHE_ROOT
    CACHE_ROOT = Path(path)


from os import getenv  # noqa: E402  (delayed import to avoid polluting module namespace)
