"""Week-level FBS simulation helpers."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

import fbs
from cfb.market import edges as edge_utils


CALIBRATION_CURRENT_WEIGHT = float(os.environ.get("FBS_CAL_WEIGHT_CURRENT", "1.0"))
CALIBRATION_PRIOR_WEIGHT = float(os.environ.get("FBS_CAL_WEIGHT_PRIOR", "0.35"))
CALIBRATION_PRIOR_YEARS = int(os.environ.get("FBS_CAL_PRIOR_YEARS", "1"))

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

    calibration_games: list[dict] = []
    for game in games:
        if not game.get("completed"):
            continue
        if (game.get("week") or 0) >= week:
            continue
        entry = game.copy()
        entry["__cal_weight"] = CALIBRATION_CURRENT_WEIGHT
        calibration_games.append(entry)

    prior_weight = CALIBRATION_PRIOR_WEIGHT
    if prior_weight > 0.0 and CALIBRATION_PRIOR_YEARS > 0:
        for offset in range(1, CALIBRATION_PRIOR_YEARS + 1):
            target_year = year - offset
            if target_year < 2014:
                break
            try:
                prior_games = fbs.fetch_games(target_year, api_key, season_type=season_type)
            except Exception:
                continue
            weight = prior_weight / float(offset)
            if weight <= 0.0:
                continue
            for game in prior_games:
                if not game.get("completed"):
                    continue
                if game.get("homeClassification") != "fbs" or game.get("awayClassification") != "fbs":
                    continue
                entry = game.copy()
                entry["__cal_weight"] = weight
                calibration_games.append(entry)
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
        "model_spread",
        "model_total",
        "model_home_points",
        "model_away_points",
        "model_home_win_prob",
        "model_home_moneyline",
        "model_away_moneyline",
        "market_spread",
        "market_total",
        "market_primary_provider",
        "market_provider_lines",
        "market_providers",
        "market_provider_count",
        "prob_sigma",
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
        weather = weather_lookup.get(game.get("id")) or game.get("weather") or {}
        result_weather = fbs.apply_weather_adjustment(result, weather)
        primary_provider = fbs.primary_provider_from_market(market)
        pure_result = fbs.apply_bias_recenter(
            result_weather.copy(),
            provider_hint=primary_provider,
            prob_sigma=book.prob_sigma,
        )
        if market:
            weight_multiplier = fbs.market_weight_for_provider(primary_provider)
            result = fbs.apply_market_prior(
                pure_result.copy(),
                market,
                prob_sigma=book.prob_sigma,
                spread_weight=fbs.MARKET_SPREAD_WEIGHT * weight_multiplier,
                total_weight=fbs.MARKET_TOTAL_WEIGHT * weight_multiplier,
            )
        else:
            result = pure_result

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
        rows["model_spread"].append(pure_result["spread_home_minus_away"])
        rows["model_total"].append(pure_result["total_points"])
        rows["model_home_points"].append(pure_result["home_points"])
        rows["model_away_points"].append(pure_result["away_points"])
        rows["model_home_win_prob"].append(pure_result["home_win_prob"])
        rows["model_home_moneyline"].append(pure_result["home_moneyline"])
        rows["model_away_moneyline"].append(pure_result["away_moneyline"])
        rows["market_spread"].append(result.get("market_spread"))
        rows["market_total"].append(result.get("market_total"))
        provider_lines = result.get("market_provider_lines") or {}
        rows["market_provider_lines"].append(json.dumps(provider_lines, sort_keys=True) if provider_lines else None)
        providers_list = result.get("market_providers") or []
        rows["market_providers"].append(", ".join(providers_list))
        rows["market_provider_count"].append(len(providers_list))
        rows["market_primary_provider"].append(primary_provider)
        rows["prob_sigma"].append(book.prob_sigma)
        rows["spread_vs_market"].append(result.get("spread_vs_market"))
        rows["total_vs_market"].append(result.get("total_vs_market"))
        rows["weather_condition"].append(result.get("weather_condition"))
        rows["weather_temp"].append(result.get("weather_temp"))
        rows["weather_wind"].append(result.get("weather_wind"))
        rows["weather_total_adj"].append(result.get("weather_total_adj"))

    df = pd.DataFrame(rows)
    df = edge_utils.annotate_edges(
        df,
        model_spread_col="spread_home_minus_away",
        market_spread_col="market_spread",
        model_total_col="total_points",
        market_total_col="market_total",
        win_prob_col="home_win_prob",
        provider_count_col="market_provider_count",
    )
    return df
