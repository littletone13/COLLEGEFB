"""Weather adjustment helpers."""

from __future__ import annotations

from typing import Dict, Optional, Tuple


def weather_features(weather: Dict[str, object]) -> Dict[str, float]:
    temp = _to_float(weather.get("temperature"))
    wind = _to_float(weather.get("windSpeed"))
    humidity = _to_float(weather.get("humidity"))
    dew_point = _to_float(weather.get("dewPoint"))
    condition = _condition(weather)
    return {
        "wind_high": max((wind or 0.0) - 10.0, 0.0),
        "cold": max(50.0 - (temp or 50.0), 0.0),
        "heat": max((temp or 50.0) - 85.0, 0.0),
        "rain": 1.0 if ("rain" in condition or (_to_float(weather.get("precipitation")) or 0.0) > 0) else 0.0,
        "snow": 1.0 if ("snow" in condition or (_to_float(weather.get("snowfall")) or 0.0) > 0) else 0.0,
        "humid": max((humidity or 0.0) - 75.0, 0.0),
        "dewpos": max((dew_point or 0.0) - 65.0, 0.0),
    }


def total_adjustment(
    weather: Optional[Dict[str, object]],
    coeffs: Dict[str, float],
    clamp_bounds: Tuple[float, float],
) -> float:
    if not weather:
        return 0.0
    features = weather_features(weather)
    delta = sum(coeffs.get(name, 0.0) * value for name, value in features.items())
    delta = min(delta, 0.0)
    lo, hi = clamp_bounds
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, min(hi, delta))


def apply_weather_adjustment(
    result: Dict[str, float],
    weather: Optional[Dict[str, object]],
    *,
    coeffs: Dict[str, float],
    clamp_bounds: Tuple[float, float],
    max_total_adjustment: float = 10.0,
    spread_key: str = "spread_home_minus_away",
    total_key: str = "total_points",
    home_points_key: str = "home_points",
    away_points_key: str = "away_points",
    market_total_key: str = "market_total",
    total_vs_market_key: str = "total_vs_market",
) -> Dict[str, float]:
    if not weather:
        return result

    total_adj = total_adjustment(weather, coeffs, clamp_bounds)
    total_adj = max(min(total_adj, max_total_adjustment), -max_total_adjustment)
    if abs(total_adj) < 1e-6:
        return result

    spread = result.get(spread_key)
    total = result.get(total_key)
    if spread is None or total is None:
        return result

    base_total = total + total_adj
    total_points = max(20.0, base_total)
    home_points = (total_points + spread) / 2.0
    away_points = total_points - home_points

    adjusted = result.copy()
    adjusted[total_key] = total_points
    adjusted[home_points_key] = home_points
    adjusted[away_points_key] = away_points
    adjusted["weather_total_adj"] = total_adj
    adjusted["weather_condition"] = _condition(weather)
    adjusted["weather_temp"] = _to_float(weather.get("temperature"))
    adjusted["weather_wind"] = _to_float(weather.get("windSpeed"))

    if adjusted.get(market_total_key) is not None:
        adjusted[total_vs_market_key] = adjusted[total_key] - adjusted[market_total_key]
    return adjusted


def _to_float(value: object) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _condition(weather: Dict[str, object]) -> str:
    condition_raw = (
        weather.get("displayValue")
        or weather.get("condition")
        or weather.get("weatherCondition")
        or weather.get("summary")
        or ""
    )
    return str(condition_raw).lower()
