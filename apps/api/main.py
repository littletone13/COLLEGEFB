"""FastAPI interface for simulations."""

from __future__ import annotations

import os
from datetime import date
from typing import Optional, Sequence

from fastapi import FastAPI, HTTPException, Query

from cfb.sim import fbs as fbs_sim
from cfb.sim import fcs as fcs_sim

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
    return {"games": df.to_dict(orient="records")}


@app.get("/fcs/window")
async def simulate_fcs_window(
    start_date: date,
    days: int = Query(3, ge=1, le=7),
    week: Optional[int] = Query(None),
    providers: Optional[str] = Query(None),
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
    return {"games": df.to_dict(orient="records")}
