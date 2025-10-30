"""Market blending and edge utilities."""

from __future__ import annotations

import math
from typing import Dict, Optional

from .edges import (
    EdgeFilterConfig,
    allow_spread_bet,
    annotate_edges,
    edge_filter_mask,
    filter_edges,
)

__all__ = [
    "moneyline_from_prob",
    "apply_market_prior",
    "EdgeFilterConfig",
    "annotate_edges",
    "edge_filter_mask",
    "filter_edges",
    "allow_spread_bet",
]


def moneyline_from_prob(prob: float) -> Optional[float]:
    """Convert a win probability into American moneyline odds."""
    if prob <= 0.0 or prob >= 1.0:
        return None
    if prob >= 0.5:
        return -100.0 * prob / (1 - prob)
    return 100.0 * (1 - prob) / prob


def apply_market_prior(
    result: Dict[str, float],
    market: Optional[dict],
    *,
    prob_sigma: float,
    spread_weight: float,
    total_weight: float,
    spread_key: str = "spread_home_minus_away",
    total_key: str = "total_points",
    home_points_key: str = "home_points",
    away_points_key: str = "away_points",
    win_prob_key: str = "home_win_prob",
    away_win_prob_key: str = "away_win_prob",
    home_moneyline_key: str = "home_moneyline",
    away_moneyline_key: str = "away_moneyline",
) -> Dict[str, float]:
    """Blend model projections with market lines."""

    updated = result.copy()
    spread = updated.get(spread_key)
    total = updated.get(total_key)

    market_spread = None
    market_total = None
    provider_list: list[str] = []
    provider_map: Dict[str, dict] = {}

    if market:
        market_spread = market.get("spread")
        market_total = market.get("total")
        provider_map = market.get("provider_lines") or {}
        provider_list = sorted(provider_map) if provider_map else market.get("providers", [])

        if spread is not None and market_spread is not None:
            w = min(max(spread_weight, 0.0), 1.0)
            spread = (1.0 - w) * spread + w * market_spread
        if total is not None and market_total is not None:
            w = min(max(total_weight, 0.0), 1.0)
            total = (1.0 - w) * total + w * market_total

    if spread is None or total is None:
        updated.setdefault("market_spread", market_spread)
        updated.setdefault("market_total", market_total)
        updated.setdefault("market_provider_lines", provider_map)
        updated.setdefault("market_providers", provider_list)
        updated.setdefault("market_provider_count", len(provider_list))
        updated.setdefault("spread_vs_market", None)
        updated.setdefault("total_vs_market", None)
        return updated

    home_points = (total + spread) / 2.0
    away_points = total - home_points
    win_prob = 0.5 * (1.0 + math.erf(spread / (prob_sigma * math.sqrt(2))))

    updated[spread_key] = spread
    updated[total_key] = total
    updated[home_points_key] = home_points
    updated[away_points_key] = away_points
    updated[win_prob_key] = win_prob
    updated[away_win_prob_key] = 1.0 - win_prob
    updated[home_moneyline_key] = moneyline_from_prob(win_prob)
    updated[away_moneyline_key] = moneyline_from_prob(1.0 - win_prob)
    updated["market_spread"] = market_spread
    updated["market_total"] = market_total
    updated["market_provider_lines"] = provider_map
    updated["market_providers"] = provider_list
    updated["market_provider_count"] = len(provider_list)
    updated["spread_vs_market"] = (
        spread - market_spread if market_spread is not None else None
    )
    updated["total_vs_market"] = (
        total - market_total if market_total is not None else None
    )
    return updated
