"""Client helpers for The Odds API v4."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"


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


def fetch_current_odds(
    sport_key: str,
    *,
    regions: Iterable[str] = ("us",),
    markets: Iterable[str] = ("spreads", "totals", "h2h"),
    bookmakers: Optional[Iterable[str]] = None,
    odds_format: str = "american",
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
                selections.append(
                    OddsSelection(
                        bookmaker=book_key,
                        market=market_key,
                        outcome=outcome_name.lower(),
                        outcome_key=_team_key(outcome_name),
                        price=outcome.get("price"),
                        point=outcome.get("point"),
                        last_update=last_update,
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
    """Denormalise spreads/totals/moneyline selections into tabular rows."""

    parsed = _parse_event(event)
    rows: list[dict] = []
    by_market: Dict[str, Dict[str, OddsSelection]] = {}
    for selection in parsed["selections"]:
        market_map = by_market.setdefault(selection.market, {})
        market_map[selection.outcome] = selection

    spreads = by_market.get("spreads", {})
    home_spread = None
    away_spread = None
    for selection in spreads.values():
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
                "bookmaker": (home_spread or away_spread).bookmaker if (home_spread or away_spread) else None,
                "point": point,
                "price_home": home_spread.price if home_spread else None,
                "price_away": away_spread.price if away_spread else None,
                "last_update": (home_spread or away_spread).last_update if (home_spread or away_spread) else None,
            }
        )

    totals = by_market.get("totals", {})
    over_sel = totals.get("over")
    under_sel = totals.get("under")
    if over_sel or under_sel:
        point = over_sel.point if over_sel and over_sel.point is not None else under_sel.point if under_sel else None
        rows.append(
            {
                "event_id": parsed["event_id"],
                "commence_time": parsed["commence_time"],
                "market": "total",
                "home_team": parsed["home_team"],
                "away_team": parsed["away_team"],
                "bookmaker": (over_sel or under_sel).bookmaker if (over_sel or under_sel) else None,
                "point": point,
                "price_over": over_sel.price if over_sel else None,
                "price_under": under_sel.price if under_sel else None,
                "last_update": (over_sel or under_sel).last_update if (over_sel or under_sel) else None,
            }
        )

    moneyline = by_market.get("h2h", {})
    home_ml = None
    away_ml = None
    draw_ml = None
    for selection in moneyline.values():
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
                "bookmaker": (home_ml or away_ml).bookmaker if (home_ml or away_ml) else None,
                "price_home": home_ml.price if home_ml else None,
                "price_away": away_ml.price if away_ml else None,
                "price_draw": draw_ml.price if draw_ml else None,
                "last_update": (home_ml or away_ml).last_update if (home_ml or away_ml) else None,
            }
        )

    return rows
