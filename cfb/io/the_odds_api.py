"""Client helpers for The Odds API v4."""

from __future__ import annotations

import logging
import time
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"

_LAST_USAGE: Dict[str, str] = {}


class TheOddsAPIError(RuntimeError):
    """Raised when a request to The Odds API fails."""


def american_to_decimal(price: Optional[float]) -> Optional[float]:
    """Convert an American price (e.g., -110) to decimal odds."""

    if price is None:
        return None
    try:
        american = float(price)
    except (TypeError, ValueError):
        return None
    if american == 0:
        return None
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(price: Optional[float]) -> Optional[float]:
    """Convert decimal odds back into the American format."""

    if price is None:
        return None
    try:
        decimal = float(price)
    except (TypeError, ValueError):
        return None
    if decimal <= 1.0:
        return None
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    return -100.0 / (decimal - 1.0)


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _datetime_to_iso(value: datetime) -> str:
    normalized = _normalize_datetime(value).replace(microsecond=0)
    return normalized.isoformat().replace("+00:00", "Z")


def get_last_usage() -> Dict[str, str]:
    """Return the most recent usage headers returned by The Odds API."""

    return _LAST_USAGE.copy()


def _api_key() -> str:
    key = os.environ.get("THE_ODDS_API_KEY")
    if not key:
        raise TheOddsAPIError(
            "THE_ODDS_API_KEY is not set. Provide your The Odds API credential via environment variable."
        )
    return key


def _request(endpoint: str, *, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Any:
    key = _api_key()
    url = f"{BASE_URL}{endpoint}"
    query = {"apiKey": key}
    if params:
        query.update(params)

    try:
        response = requests.get(url, params=query, timeout=timeout)
    except requests.RequestException as exc:  # pragma: no cover
        raise TheOddsAPIError(f"Request to {url} failed: {exc}") from exc

    usage_headers = {
        "requests_remaining": response.headers.get("x-requests-remaining"),
        "requests_used": response.headers.get("x-requests-used"),
        "requests_limit": response.headers.get("x-requests-limit"),
        "token_balance": response.headers.get("x-odds-token-balance"),
    }
    global _LAST_USAGE
    _LAST_USAGE = {key: value for key, value in usage_headers.items() if value is not None}
    if _LAST_USAGE and "odds-history" in endpoint:
        logger.info(
            "The Odds API usage after %s: requests remaining=%s, token balance=%s",
            endpoint,
            _LAST_USAGE.get("requests_remaining", "?"),
            _LAST_USAGE.get("token_balance", "?"),
        )

    if response.status_code == 401:
        raise TheOddsAPIError("The Odds API rejected the key (401). Check THE_ODDS_API_KEY.")
    if response.status_code == 429:
        raise TheOddsAPIError("The Odds API rate limit was exceeded (429).")
    if response.status_code >= 500:
        raise TheOddsAPIError(f"The Odds API server error {response.status_code}: {response.text[:200]}")

    try:
        return response.json()
    except ValueError as exc:
        raise TheOddsAPIError(f"Invalid JSON returned by The Odds API: {response.text[:200]}") from exc


@dataclass(frozen=True)
class OddsSelection:
    bookmaker: str
    market: str
    outcome: str
    outcome_key: str
    price: Optional[float]
    point: Optional[float]
    last_update: Optional[datetime]
    description: Optional[str] = None
    player_name: Optional[str] = None
    player_id: Optional[str] = None
    team: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def fetch_current_odds(
    sport_key: str,
    *,
    regions: Iterable[str] = ("us",),
    markets: Iterable[str] = ("spreads", "totals", "h2h"),
    bookmakers: Optional[Iterable[str]] = None,
    odds_format: str = "american",
    days_from: Optional[int] = None,
    timeout: int = 30,
) -> list[dict]:
    """Fetch live odds for the requested sport."""

    params: Dict[str, Any] = {
        "regions": ",".join(regions),
        "markets": ",".join(markets),
        "oddsFormat": odds_format,
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)
    if days_from is not None:
        params["daysFrom"] = int(days_from)

    payload = _request(f"/sports/{sport_key}/odds", params=params, timeout=timeout)
    if not isinstance(payload, list):
        raise TheOddsAPIError("Unexpected response payload from The Odds API.")
    return payload


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _team_key(label: Optional[str]) -> str:
    """Return a normalised The Odds API team key (lower + alnum)."""

    return "".join(ch for ch in (label or "").lower() if ch.isalnum())


def _parse_event(event: dict) -> dict:
    home_team = event.get("home_team")
    away_team = event.get("away_team")
    home_key = _team_key(home_team)
    away_key = _team_key(away_team)
    selections: list[OddsSelection] = []
    for bookmaker in event.get("bookmakers", []) or []:
        book_key = bookmaker.get("key")
        last_update = _parse_datetime(bookmaker.get("last_update"))
        for market in bookmaker.get("markets", []) or []:
            market_key = market.get("key")
            for outcome in market.get("outcomes", []) or []:
                outcome_name = (outcome.get("name") or "").strip()
                player_id = outcome.get("player_id") or outcome.get("playerId") or outcome.get("id")
                player_id = str(player_id) if player_id is not None else None
                description = outcome.get("description") or outcome.get("participant") or outcome.get("player")
                player_name = outcome.get("player_name") or description
                selections.append(
                    OddsSelection(
                        bookmaker=book_key,
                        market=market_key,
                        outcome=outcome_name.lower(),
                        outcome_key=_team_key(outcome_name),
                        price=outcome.get("price"),
                        point=outcome.get("point"),
                        last_update=last_update,
                        description=description.strip() if isinstance(description, str) else description,
                        player_name=player_name.strip() if isinstance(player_name, str) else player_name,
                        player_id=player_id,
                        team=outcome.get("team"),
                        extra={
                            "player": outcome.get("player"),
                            "participant": outcome.get("participant"),
                            "team": outcome.get("team"),
                            "description": outcome.get("description"),
                        },
                    )
                )

    return {
        "event_id": event.get("id"),
        "commence_time": _parse_datetime(event.get("commence_time")),
        "home_team": event.get("home_team"),
        "away_team": event.get("away_team"),
        "home_key": home_key,
        "away_key": away_key,
        "selections": selections,
    }


def normalise_prices(event: dict) -> list[dict]:
    """Denormalise spreads/totals/moneyline selections into tabular rows.

    Each bookmaker/market pair becomes its own row so multiple books can
    coexist for the same game.
    """

    parsed = _parse_event(event)
    rows: list[dict] = []
    by_book_market: Dict[tuple[str, str], Dict[str, OddsSelection]] = {}
    for selection in parsed["selections"]:
        if not selection.bookmaker:
            continue
        key = (selection.bookmaker, selection.market)
        by_book_market.setdefault(key, {})[selection.outcome] = selection

    for (bookmaker, market_key), selections in by_book_market.items():
        if market_key == "spreads":
            home_spread = None
            away_spread = None
            for selection in selections.values():
                if selection.outcome_key == parsed["home_key"]:
                    home_spread = selection
                elif selection.outcome_key == parsed["away_key"]:
                    away_spread = selection
            if home_spread or away_spread:
                point = None
                if home_spread and home_spread.point is not None:
                    point = home_spread.point
                elif away_spread and away_spread.point is not None:
                    point = -away_spread.point
                rows.append(
                    {
                        "event_id": parsed["event_id"],
                        "commence_time": parsed["commence_time"],
                        "market": "spread",
                        "home_team": parsed["home_team"],
                        "away_team": parsed["away_team"],
                        "bookmaker": bookmaker,
                        "point": point,
                        "price_home": home_spread.price if home_spread else None,
                        "price_away": away_spread.price if away_spread else None,
                        "last_update": home_spread.last_update
                        if home_spread
                        else away_spread.last_update if away_spread else None,
                    }
                )
        elif market_key == "totals":
            over_sel = selections.get("over")
            under_sel = selections.get("under")
            if over_sel or under_sel:
                point = (
                    over_sel.point
                    if over_sel and over_sel.point is not None
                    else under_sel.point
                    if under_sel
                    else None
                )
                rows.append(
                    {
                        "event_id": parsed["event_id"],
                        "commence_time": parsed["commence_time"],
                        "market": "total",
                        "home_team": parsed["home_team"],
                        "away_team": parsed["away_team"],
                        "bookmaker": bookmaker,
                        "point": point,
                        "price_over": over_sel.price if over_sel else None,
                        "price_under": under_sel.price if under_sel else None,
                        "last_update": over_sel.last_update
                        if over_sel
                        else under_sel.last_update if under_sel else None,
                    }
                )
        elif market_key in {"h2h", "moneyline"}:
            home_ml = None
            away_ml = None
            draw_ml = None
            for selection in selections.values():
                if selection.outcome_key == parsed["home_key"]:
                    home_ml = selection
                elif selection.outcome_key == parsed["away_key"]:
                    away_ml = selection
                elif selection.outcome in {"draw", "tie"}:
                    draw_ml = selection
            if home_ml or away_ml:
                rows.append(
                    {
                        "event_id": parsed["event_id"],
                        "commence_time": parsed["commence_time"],
                        "market": "moneyline",
                        "home_team": parsed["home_team"],
                        "away_team": parsed["away_team"],
                        "bookmaker": bookmaker,
                        "price_home": home_ml.price if home_ml else None,
                        "price_away": away_ml.price if away_ml else None,
                        "price_draw": draw_ml.price if draw_ml else None,
                        "last_update": home_ml.last_update
                        if home_ml
                        else away_ml.last_update if away_ml else None,
                    }
                )

    return rows


def normalise_player_props(event: dict, *, markets: Optional[Iterable[str]] = None) -> list[dict]:
    """Flatten player prop selections from a The Odds API event payload."""

    parsed = _parse_event(event)
    rows: list[dict] = []
    allowed_markets = {m.lower() for m in markets} if markets else None

    for selection in parsed["selections"]:
        market_key = selection.market
        if allowed_markets and market_key.lower() not in allowed_markets:
            continue

        # Heuristic: treat markets starting with player_ or with player metadata as props
        has_player_meta = any(
            [
                selection.player_name,
                selection.description,
                selection.extra.get("participant"),
                selection.extra.get("player"),
                market_key.lower().startswith("player_"),
            ]
        )
        if not has_player_meta:
            continue

        player_label = (
            selection.player_name
            or selection.description
            or selection.extra.get("participant")
            or selection.extra.get("player")
            or selection.outcome
        )

        rows.append(
            {
                "event_id": parsed["event_id"],
                "commence_time": parsed["commence_time"],
                "market": market_key,
                "bookmaker": selection.bookmaker,
                "player": player_label,
                "outcome": selection.outcome,
                "point": selection.point,
                "price": selection.price,
                "team": selection.team or selection.extra.get("team"),
                "last_update": selection.last_update,
                "player_id": selection.player_id,
                "home_team": parsed["home_team"],
                "away_team": parsed["away_team"],
            }
        )

    return rows


def fetch_history_snapshot(
    sport_key: str,
    timestamp: datetime,
    *,
    regions: Iterable[str] = ("us",),
    markets: Iterable[str] = ("spreads", "totals", "h2h"),
    bookmakers: Optional[Iterable[str]] = None,
    event_ids: Optional[Iterable[str]] = None,
    odds_format: str = "american",
    timeout: int = 30,
) -> dict:
    """Fetch a single odds-history snapshot."""

    params: Dict[str, Any] = {
        "date": _datetime_to_iso(timestamp),
        "regions": ",".join(regions),
        "markets": ",".join(markets),
        "oddsFormat": odds_format,
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)
    if event_ids:
        params["eventIds"] = ",".join(event_ids)
    payload = _request(f"/sports/{sport_key}/odds-history/", params=params, timeout=timeout)
    if not isinstance(payload, dict):
        raise TheOddsAPIError("Unexpected payload returned by The Odds API odds-history endpoint.")
    return payload


def iter_history_snapshots(
    sport_key: str,
    *,
    start: datetime,
    end: datetime,
    regions: Iterable[str] = ("us",),
    markets: Iterable[str] = ("spreads", "totals", "h2h"),
    bookmakers: Optional[Iterable[str]] = None,
    event_ids: Optional[Iterable[str]] = None,
    odds_format: str = "american",
    timeout: int = 30,
    max_iterations: int = 288,
    retry_attempts: int = 3,
    retry_backoff: float = 3.0,
) -> Iterator[dict]:
    """Yield odds-history snapshots between ``start`` and ``end`` inclusive."""

    cursor = _normalize_datetime(start)
    limit = _normalize_datetime(end)
    iterations = 0
    while cursor <= limit and iterations < max_iterations:
        attempt = 0
        while True:
            try:
                payload = fetch_history_snapshot(
                    sport_key,
                    cursor,
                    regions=regions,
                    markets=markets,
                    bookmakers=bookmakers,
                    event_ids=event_ids,
                    odds_format=odds_format,
                    timeout=timeout,
                )
                break
            except TheOddsAPIError as exc:
                attempt += 1
                if attempt >= retry_attempts:
                    raise
                sleep_for = retry_backoff * attempt
                logger.warning(
                    "The Odds API history request failed (%s). Retrying in %.1fs (attempt %d/%d).",
                    exc,
                    sleep_for,
                    attempt,
                    retry_attempts,
                )
                time.sleep(sleep_for)
        yield payload
        next_raw = payload.get("next_timestamp")
        if not next_raw:
            break
        next_dt = _parse_datetime(next_raw)
        if not next_dt:
            break
        next_dt = _normalize_datetime(next_dt)
        if next_dt <= cursor:
            break
        cursor = next_dt
        iterations += 1
