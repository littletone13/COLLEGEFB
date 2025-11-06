"""Lightweight FastAPI wrapper to expose CFBD sportsbook lines.

Start the API with:

    uvicorn lines_api:app --reload --port 8000

Ensure the environment variable ``CFBD_API_KEY`` is set (Bearer token
without the prefix) or pass ``api_key`` as a query parameter.
"""
from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query

import fbs
import fcs


app = FastAPI(title="College Football Lines API", version="1.0.0")


def _resolve_api_key(explicit: Optional[str]) -> str:
    key = explicit or os.environ.get("CFBD_API_KEY")
    if not key:
        raise HTTPException(
            status_code=400,
            detail="CFBD API key missing. Set CFBD_API_KEY or supply the api_key query parameter.",
        )
    return key


def _parse_providers(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    providers = [part.strip() for part in value.split(",") if part.strip()]
    return providers or None


@app.get("/getlines")
def get_lines(
    classification: str = Query("fbs", description="Team classification: fbs or fcs."),
    year: int = Query(..., ge=2000, description="Season year."),
    week: Optional[int] = Query(None, ge=1, description="Week number (optional)."),
    season_type: str = Query(
        "regular",
        pattern="^(regular|postseason)$",
        description="Season type selector.",
    ),
    providers: Optional[str] = Query(
        None,
        description="Comma-separated sportsbook providers to include (default: all).",
    ),
    api_key: Optional[str] = Query(None, description="Override CFBD API key for this request."),
):
    key = _resolve_api_key(api_key)
    provider_list = _parse_providers(providers)
    classification_norm = classification.lower()
    if classification_norm not in {"fbs", "fcs"}:
        raise HTTPException(status_code=400, detail="classification must be 'fbs' or 'fcs'.")

    if classification_norm == "fbs":
        raw = fbs.fetch_market_lines(
            year,
            key,
            week=week,
            season_type=season_type,
            classification="fbs",
            providers=provider_list,
        )
        games = [
            {
                "game_id": game_id,
                "home_team": info.get("home_team"),
                "away_team": info.get("away_team"),
                "start_date": info.get("start_date"),
                "neutral_site": info.get("neutral_site"),
                "consensus_spread": info.get("spread"),
                "consensus_total": info.get("total"),
                "providers": info.get("providers"),
                "provider_lines": info.get("provider_lines"),
            }
            for game_id, info in raw.items()
        ]
    else:
        raw = fcs.fetch_market_lines(
            year,
            key,
            week=week,
            season_type=season_type,
            classification="fcs",
            providers=provider_list,
        )
        games = [
            {
                "game_id": entry.get("game_id"),
                "home_team": entry.get("home_team"),
                "away_team": entry.get("away_team"),
                "start_date": entry.get("start_date"),
                "consensus_spread": entry.get("spread"),
                "consensus_total": entry.get("total"),
                "providers": entry.get("providers"),
                "provider_lines": entry.get("provider_lines"),
            }
            for entry in raw
        ]

    return {
        "classification": classification_norm,
        "year": year,
        "week": week,
        "season_type": season_type,
        "providers": provider_list,
        "games": games,
    }
