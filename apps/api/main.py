"""FastAPI interface for simulations."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

from fastapi import FastAPI, HTTPException, Query

from cfb.sim import fbs as fbs_sim
from cfb.sim import fcs as fcs_sim
from cfb.io import oddslogic as oddslogic_io
from cfb.market import edges as edge_utils

ARCHIVE_DIR = Path(os.environ.get("ODDSLOGIC_ARCHIVE_DIR", "oddslogic_ncaa_all")).expanduser()

app = FastAPI(title="College Football Sims API")


def _require_api_key(provided: Optional[str]) -> None:
    expected = os.environ.get("SIMS_API_KEY")
    if expected and provided != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/fbs/week")
async def simulate_fbs_week(
    year: int,
    week: int,
    include_completed: bool = Query(False),
    neutral_default: bool = Query(False),
    providers: Optional[str] = Query(None, description="Comma separated provider filter"),
    min_spread_edge: float = Query(0.0, ge=0.0),
    min_total_edge: float = Query(0.0, ge=0.0),
    min_provider_count: int = Query(0, ge=0),
    api_key: Optional[str] = Query(None, alias="auth"),
) -> dict:
    _require_api_key(api_key)
    provider_list: Optional[Sequence[str]] = (
        [p.strip() for p in providers.split(",") if p.strip()]
        if providers
        else None
    )
    df = fbs_sim.simulate_week(
        year,
        week,
        include_completed=include_completed,
        neutral_default=neutral_default,
        providers=provider_list,
    )
    total_games = int(df.shape[0])
    config = edge_utils.EdgeFilterConfig(
        spread_edge_min=min_spread_edge,
        total_edge_min=min_total_edge,
        min_provider_count=min_provider_count,
    )
    mask = edge_utils.edge_filter_mask(
        df,
        config,
        provider_count_col="provider_count",
    )
    if config.spread_edge_min > 0 or config.total_edge_min > 0 or config.min_provider_count > 0:
        df = df.loc[mask]
    return {
        "games": df.to_dict(orient="records"),
        "meta": {
            "total_games": total_games,
            "returned_games": int(df.shape[0]),
            "filters": {
                "spread_edge_min": min_spread_edge,
                "total_edge_min": min_total_edge,
                "min_provider_count": min_provider_count,
            },
        },
    }


@app.get("/fcs/window")
async def simulate_fcs_window(
    start_date: date,
    days: int = Query(3, ge=1, le=7),
    week: Optional[int] = Query(None),
    providers: Optional[str] = Query(None),
    min_spread_edge: float = Query(0.0, ge=0.0),
    min_total_edge: float = Query(0.0, ge=0.0),
    min_provider_count: int = Query(0, ge=0),
    api_key: Optional[str] = Query(None, alias="auth"),
) -> dict:
    _require_api_key(api_key)
    provider_list: Optional[Sequence[str]] = (
        [p.strip() for p in providers.split(",") if p.strip()]
        if providers
        else None
    )
    df = fcs_sim.simulate_window(
        start_date,
        days=days,
        week=week,
        providers=provider_list,
    )
    total_games = int(df.shape[0])
    config = edge_utils.EdgeFilterConfig(
        spread_edge_min=min_spread_edge,
        total_edge_min=min_total_edge,
        min_provider_count=min_provider_count,
    )
    mask = edge_utils.edge_filter_mask(
        df,
        config,
        provider_count_col="provider_count",
    )
    if config.spread_edge_min > 0 or config.total_edge_min > 0 or config.min_provider_count > 0:
        df = df.loc[mask]

    return {
        "games": df.to_dict(orient="records"),
        "meta": {
            "total_games": total_games,
            "returned_games": int(df.shape[0]),
            "filters": {
                "spread_edge_min": min_spread_edge,
                "total_edge_min": min_total_edge,
                "min_provider_count": min_provider_count,
            },
        },
    }


@app.get("/oddslogic/coverage")
async def oddslogic_coverage(
    classification: Optional[str] = Query(None, description="Filter by classification: fbs/fcs"),
    api_key: Optional[str] = Query(None, alias="auth"),
) -> dict:
    _require_api_key(api_key)
    if not ARCHIVE_DIR.exists():
        raise HTTPException(status_code=404, detail="Archive directory not found")
    df_archive = oddslogic_io.load_archive(ARCHIVE_DIR)
    coverage, providers = oddslogic_io.summarize_coverage(df_archive, classification=classification)
    return {
        "coverage": coverage.to_dict(orient="records"),
        "provider_coverage": providers.to_dict(orient="records"),
    }


@app.get("/oddslogic/injuries")
async def oddslogic_injuries(
    league: str = Query("ncaaf_fbs", description="OddsLogic league identifier (e.g., ncaaf_fbs, nfl)"),
    team_id: int = Query(0, ge=0),
    player_name: Optional[str] = Query(None),
    api_key: Optional[str] = Query(None, alias="auth"),
) -> dict:
    _require_api_key(api_key)
    payload = oddslogic_io.fetch_injuries(
        league=league,
        team_id=team_id,
        player_name=player_name or "",
    )
    injuries = []
    for name, info in payload.items():
        entry = {"player_key": name}
        entry.update(info)
        injuries.append(entry)
    return {"league": league, "injuries": injuries}
