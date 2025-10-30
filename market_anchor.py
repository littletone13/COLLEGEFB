"""Utilities for deriving market-based power adjustments from OddsLogic closing lines."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

import oddslogic_loader

SHARP_PROVIDERS = {"Circa", "Pinnacle", "South Point", "Superbook", "Bookmaker"}


@dataclass
class MarketAnchorConfig:
    archive_path: Path
    classification: str = "fbs"
    providers: Optional[Iterable[str]] = None
    sharp_only: bool = True
    min_provider_weight: float = 0.25


def _load_closing_lookup(config: MarketAnchorConfig) -> Dict[Tuple[pd.Timestamp, str, str], Dict[str, object]]:
    df_archive = oddslogic_loader.load_archive_dataframe(config.archive_path)
    providers = list(config.providers) if config.providers else None
    return oddslogic_loader.build_closing_lookup(df_archive, config.classification, providers=providers)


def _team_keys(home: str, away: str) -> Tuple[str, str]:
    return (
        oddslogic_loader.normalize_label(home),
        oddslogic_loader.normalize_label(away),
    )


def derive_power_adjustments(
    games: Iterable[dict],
    *,
    config: MarketAnchorConfig,
) -> Dict[str, float]:
    closing_lookup = _load_closing_lookup(config)

    totals: Dict[str, float] = defaultdict(float)
    weights: Dict[str, float] = defaultdict(float)

    for game in games:
        if game.get("id") is None:
            continue
        if game.get("startDate") is None:
            continue

        kickoff = pd.to_datetime(game.get("startDate"), errors="coerce")
        if pd.isna(kickoff):
            continue
        kickoff_date = kickoff.date()

        home_team = game.get("homeTeam")
        away_team = game.get("awayTeam")
        if not home_team or not away_team:
            continue

        home_key, away_key = _team_keys(home_team, away_team)
        lookup_key = (kickoff_date, home_key, away_key)
        closing = closing_lookup.get(lookup_key)
        invert = False
        if not closing:
            closing = closing_lookup.get((kickoff_date, away_key, home_key))
            if closing:
                invert = True
        if not closing:
            continue

        model_spread = game.get("model_spread")
        if model_spread is None:
            continue

        provider_entries = closing.get("providers") or {}

        def _provider_weight(name: str, sportsbook_id: Optional[int]) -> float:
            if not config.sharp_only:
                return 1.0
            if name in SHARP_PROVIDERS:
                return 1.0
            if sportsbook_id in oddslogic_loader.SHARP_PROVIDER_IDS:
                return 1.0
            return config.min_provider_weight

        if provider_entries:
            for payload in provider_entries.values():
                provider_name = payload.get("sportsbook_name") or ""
                spread_value = payload.get("spread_value")
                if spread_value is None or math.isnan(spread_value):
                    continue
                if invert:
                    spread_value = -spread_value
                weight = _provider_weight(provider_name, payload.get("sportsbook_id"))
                if weight <= 0.0:
                    continue
                diff = spread_value - model_spread
                adjustment = diff / 2.0
                totals[home_team] += adjustment * weight
                totals[away_team] -= adjustment * weight
                weights[home_team] += weight
                weights[away_team] += weight
        else:
            spread_value = closing.get("spread_value")
            if spread_value is None or math.isnan(spread_value):
                continue
            if invert:
                spread_value = -spread_value
            provider_name = closing.get("sportsbook_name") or ""
            weight = _provider_weight(provider_name, closing.get("sportsbook_id"))
            if weight <= 0.0:
                continue
            diff = spread_value - model_spread
            adjustment = diff / 2.0
            totals[home_team] += adjustment * weight
            totals[away_team] -= adjustment * weight
            weights[home_team] += weight
            weights[away_team] += weight

    adjustments: Dict[str, float] = {}
    for team, total in totals.items():
        if weights[team]:
            adjustments[team] = total / weights[team]
    return adjustments


def save_adjustments(adjustments: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(adjustments, f, indent=2, sort_keys=True)


def load_adjustments(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
