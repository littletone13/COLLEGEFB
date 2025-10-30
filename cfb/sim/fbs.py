"""Week-level FBS simulation helpers."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

import fbs


def simulate_week(
    year: int,
    week: int,
    *,
    api_key: Optional[str] = None,
    season_type: str = "regular",
    include_completed: bool = False,
    neutral_default: bool = False,
    providers: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return a DataFrame of simulated games for a given FBS week."""

    api_key = api_key or os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD API key required; set CFBD_API_KEY or pass api_key.")

    games = fbs.fetch_games(year, api_key, season_type=season_type)
    calibration_games = [g for g in games if g.get("completed") and (g.get("week") or 0) < week]
    ratings, book = fbs.build_rating_book(
        year,
        api_key=api_key,
        adjust_week=week,
        calibration_games=calibration_games,
    )

    weather_lookup = fbs.fetch_game_weather(
        year,
        api_key,
        week=week,
        season_type=season_type,
    )

    market_lookup = fbs.fetch_market_lines(
        year,
        api_key,
        week=week,
        season_type=season_type,
        classification="fbs",
        providers=providers,
    )

    rows: Dict[str, list] = {column: [] for column in (
        "game_id",
        "week",
        "start_date",
        "home_team",
        "away_team",
        "neutral",
        "spread_home_minus_away",
        "total_points",
        "home_points",
        "away_points",
        "home_win_prob",
        "home_moneyline",
        "away_moneyline",
        "market_spread",
        "market_total",
        "market_provider_lines",
        "market_providers",
        "market_provider_count",
        "spread_vs_market",
        "total_vs_market",
        "weather_condition",
        "weather_temp",
        "weather_wind",
        "weather_total_adj",
    )}

    for game in games:
        if game.get("week") != week:
            continue
        if game.get("homeClassification") != "fbs" or game.get("awayClassification") != "fbs":
            continue
        if game.get("completed") and not include_completed:
            continue
        neutral = game.get("neutralSite")
        if neutral is None:
            neutral = neutral_default
        try:
            result = book.predict(game["homeTeam"], game["awayTeam"], neutral_site=neutral)
        except KeyError:
            continue
        market = market_lookup.get(game.get("id"))
        result = fbs.apply_market_prior(result, market, prob_sigma=book.prob_sigma)
        weather = weather_lookup.get(game.get("id")) or game.get("weather") or {}
        result = fbs.apply_weather_adjustment(result, weather)

        rows["game_id"].append(game.get("id"))
        rows["week"].append(game.get("week"))
        rows["start_date"].append(game.get("startDate"))
        rows["home_team"].append(result["home_team"])
        rows["away_team"].append(result["away_team"])
        rows["neutral"].append(neutral)
        rows["spread_home_minus_away"].append(result["spread_home_minus_away"])
        rows["total_points"].append(result["total_points"])
        rows["home_points"].append(result["home_points"])
        rows["away_points"].append(result["away_points"])
        rows["home_win_prob"].append(result["home_win_prob"])
        rows["home_moneyline"].append(result["home_moneyline"])
        rows["away_moneyline"].append(result["away_moneyline"])
        rows["market_spread"].append(result.get("market_spread"))
        rows["market_total"].append(result.get("market_total"))
        provider_lines = result.get("market_provider_lines") or {}
        rows["market_provider_lines"].append(json.dumps(provider_lines, sort_keys=True) if provider_lines else None)
        providers_list = result.get("market_providers") or []
        rows["market_providers"].append(", ".join(providers_list))
        rows["market_provider_count"].append(len(providers_list))
        rows["spread_vs_market"].append(result.get("spread_vs_market"))
        rows["total_vs_market"].append(result.get("total_vs_market"))
        rows["weather_condition"].append(result.get("weather_condition"))
        rows["weather_temp"].append(result.get("weather_temp"))
        rows["weather_wind"].append(result.get("weather_wind"))
        rows["weather_total_adj"].append(result.get("weather_total_adj"))

    return pd.DataFrame(rows)
