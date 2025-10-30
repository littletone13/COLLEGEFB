"""FCS simulation helpers."""

from __future__ import annotations

import difflib
import json
import os
import re
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
import requests

import fcs
import ncaa_stats
from cfb.fcs_aliases import DISPLAY_NAME_OVERRIDES, TEAM_NAME_ALIASES, normalize_label as _normalize_label
from cfb.market import edges as edge_utils


def _map_team(name: Optional[str], pff_set: set[str], pff_names: Sequence[str]) -> Optional[str]:
    if not name:
        return None
    raw = name.upper()
    if raw in pff_set:
        return raw
    normalized = _normalize_label(raw)
    alias = TEAM_NAME_ALIASES.get(normalized)
    if alias:
        return alias
    if normalized in pff_set:
        return normalized
    matches = difflib.get_close_matches(raw, pff_names, n=1, cutoff=0.75)
    return matches[0] if matches else None


def _resolve_entry(entry: dict, pff_set: set[str], pff_names: Sequence[str]) -> Optional[str]:
    team = entry.get("team") or {}
    labels = [
        team.get("location"),
        team.get("abbreviation"),
        team.get("displayName"),
        team.get("shortDisplayName"),
        team.get("name"),
    ]
    for label in labels:
        candidate = _map_team(label, pff_set, pff_names)
        if candidate:
            return candidate
    return None


def _fetch_espn_schedule(start: date, end: date, pff_set: set[str], pff_names: Sequence[str]) -> pd.DataFrame:
    records: list[dict] = []
    current = start
    while current <= end:
        datestr = current.strftime("%Y%m%d")
        url = (
            "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
            f"?groups=81&dates={datestr}"
        )
        try:
            data = requests.get(url, timeout=30).json()
        except Exception:
            current += timedelta(days=1)
            continue
        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            home_team = _resolve_entry(home, pff_set, pff_names)
            away_team = _resolve_entry(away, pff_set, pff_names)
            start_iso = comp.get("date")
            records.append(
                {
                    "start_date": start_iso,
                    "home_team": home_team,
                    "away_team": away_team,
                }
            )
        current += timedelta(days=1)
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["start_dt"] = pd.to_datetime(df["start_date"], errors="coerce")
    df = df[(df["start_dt"].dt.date >= start) & (df["start_dt"].dt.date <= end)]
    return df.dropna(subset=["home_team", "away_team"])


def _fetch_ncaa_schedule(start: date, end: date, season_year: int) -> pd.DataFrame:
    try:
        scoreboard = ncaa_stats.fetch_scoreboard_games(season_year)
    except Exception:
        return pd.DataFrame(columns=["start_date", "home_team", "away_team"])
    if scoreboard.empty:
        return scoreboard
    scoreboard["date"] = pd.to_datetime(scoreboard["date"], errors="coerce").dt.date
    window = scoreboard[(scoreboard["date"] >= start) & (scoreboard["date"] <= end)].copy()
    if window.empty:
        return pd.DataFrame(columns=["start_date", "home_team", "away_team"])
    window["home_team"] = window["home_slug"].map(ncaa_stats.SLUG_TO_PFF.get)
    window["away_team"] = window["away_slug"].map(ncaa_stats.SLUG_TO_PFF.get)
    window["start_date"] = window["start_date"].fillna(
        window["date"].apply(lambda d: datetime.combine(d, datetime.min.time()).isoformat())
    )
    return window.dropna(subset=["home_team", "away_team"])[["start_date", "home_team", "away_team"]]


def simulate_window(
    start_date: date,
    *,
    days: int = 3,
    week: Optional[int] = None,
    api_key: Optional[str] = None,
    data_dir: Path = fcs.DATA_DIR_DEFAULT,
    providers: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Simulate games in the given date window."""

    end_date = start_date + timedelta(days=days)
    season_year = start_date.year if start_date.month >= 7 else start_date.year - 1

    teams, book = fcs.build_rating_book(data_dir, season_year=season_year)

    pff_names = list(ncaa_stats.SLUG_TO_PFF.values())
    pff_set = set(pff_names)

    provider_filter = [p.strip() for p in providers if p.strip()] if providers else None
    market_lookup: Dict[Tuple[str, str], dict] = {}
    if api_key and week is not None:
        try:
            market_entries = fcs.fetch_market_lines(
                season_year,
                api_key,
                week=week,
                season_type="regular",
                providers=provider_filter,
            )
            for entry in market_entries:
                mapped_home = _map_team(entry.get("home_team"), pff_set, pff_names)
                mapped_away = _map_team(entry.get("away_team"), pff_set, pff_names)
                if not mapped_home or not mapped_away:
                    continue
                market_lookup[(mapped_home, mapped_away)] = entry
        except Exception as exc:  # pragma: no cover - network failures
            warnings.warn(f"Unable to fetch FCS market lines: {exc}")

    espn_slate = _fetch_espn_schedule(start_date, end_date, pff_set, pff_names)
    ncaa_slate = _fetch_ncaa_schedule(start_date, end_date, season_year)
    slate = pd.concat([espn_slate, ncaa_slate], ignore_index=True)
    if not slate.empty:
        slate = slate.drop_duplicates(subset=["start_date", "home_team", "away_team"])
    if slate.empty:
        return pd.DataFrame(columns=[
            "start_date",
            "home_team",
            "away_team",
            "spread",
            "total",
            "home_points",
            "away_points",
            "home_win_prob",
            "home_ml",
            "away_ml",
            "market_spread",
            "market_total",
            "market_provider_count",
            "market_providers",
            "market_provider_lines",
            "spread_vs_market",
            "total_vs_market",
        ])

    projections: Dict[str, list] = {key: [] for key in [
        "start_date",
        "home_team",
        "away_team",
        "spread",
        "total",
        "home_points",
        "away_points",
        "home_win_prob",
        "home_ml",
        "away_ml",
        "market_spread",
        "market_total",
        "market_provider_count",
        "market_providers",
        "market_provider_lines",
        "spread_vs_market",
        "total_vs_market",
    ]}

    for _, row in slate.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        try:
            result = book.predict(home_team, away_team, neutral_site=False)
        except KeyError:
            continue
        market_entry = market_lookup.get((home_team, away_team))
        result = fcs.apply_market_prior(result, market_entry)
        display_home = DISPLAY_NAME_OVERRIDES.get(result.get("team_one") or result.get("home_team"), result.get("team_one") or result.get("home_team"))
        display_away = DISPLAY_NAME_OVERRIDES.get(result.get("team_two") or result.get("away_team"), result.get("team_two") or result.get("away_team"))
        projections["start_date"].append(row.get("start_date"))
        projections["home_team"].append(display_home)
        projections["away_team"].append(display_away)
        projections["spread"].append(result.get("spread_home_minus_away") or result.get("spread_team_one_minus_team_two"))
        projections["total"].append(result["total_points"])
        projections["home_points"].append(result.get("home_points") or result.get("team_one_points"))
        projections["away_points"].append(result.get("away_points") or result.get("team_two_points"))
        projections["home_win_prob"].append(result.get("home_win_prob") or result.get("team_one_win_prob"))
        projections["home_ml"].append(result.get("home_moneyline") or result.get("team_one_moneyline"))
        projections["away_ml"].append(result.get("away_moneyline") or result.get("team_two_moneyline"))
        projections["market_spread"].append(result.get("market_spread"))
        projections["market_total"].append(result.get("market_total"))
        providers_list = result.get("market_providers") or []
        projections["market_provider_count"].append(len(providers_list))
        projections["market_providers"].append(", ".join(providers_list) if providers_list else None)
        provider_lines = result.get("market_provider_lines") or {}
        projections["market_provider_lines"].append(
            json.dumps(provider_lines, sort_keys=True) if provider_lines else None
        )
        projections["spread_vs_market"].append(result.get("spread_vs_market"))
        projections["total_vs_market"].append(result.get("total_vs_market"))

    df = pd.DataFrame(projections)
    df = edge_utils.annotate_edges(
        df,
        model_spread_col="spread",
        market_spread_col="market_spread",
        model_total_col="total",
        market_total_col="market_total",
        win_prob_col="home_win_prob",
        provider_count_col="market_provider_count",
    )
    return df.sort_values("start_date")
