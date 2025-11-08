#!/usr/bin/env python3
"""Fit FBS offense/defense/power rating weights from historical seasons."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import fbs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn FBS rating weights via ridge regression on historical seasons."
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        required=True,
        help="Season years to include (e.g., 2022 2023 2024).",
    )
    parser.add_argument(
        "--season-type",
        default="regular",
        help="Season type passed to CFBD (default: regular).",
    )
    parser.add_argument(
        "--api-key",
        help="CFBD API key (defaults to CFBD_API_KEY env var).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=fbs.RATING_WEIGHTS_PATH,
        help=f"Destination for the learned weights (default: {fbs.RATING_WEIGHTS_PATH}).",
    )
    return parser.parse_args()


def _load_weights(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _filter_games(games: Iterable[dict]) -> List[dict]:
    filtered: List[dict] = []
    for game in games:
        if not game.get("completed"):
            continue
        if game.get("homeClassification") != "fbs" or game.get("awayClassification") != "fbs":
            continue
        entry = game.copy()
        entry["__cal_weight"] = float(entry.get("__cal_weight", 1.0))
        filtered.append(entry)
    return filtered


def _accumulate(section: Dict[str, float], weight: float, totals: Dict[str, float]) -> None:
    if weight <= 0:
        return
    for key, value in section.items():
        totals[key] = totals.get(key, 0.0) + float(value) * weight


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD API key is required via --api-key or CFBD_API_KEY env var.")

    seasons = sorted(set(args.seasons))
    if not seasons:
        raise RuntimeError("At least one season must be provided.")

    offense_totals: Dict[str, float] = {}
    defense_totals: Dict[str, float] = {}
    power_totals: Dict[str, float] = {}
    offense_rows_total = 0.0
    power_rows_total = 0.0

    for season in seasons:
        games = fbs.fetch_games(season, api_key, season_type=args.season_type)
        calibration_games = _filter_games(games)
        if not calibration_games:
            print(f"[warn] No completed FBS games for {season}; skipping.")
            continue
        print(f"[info] Fitting weights for {season} on {len(calibration_games)} games...")
        fbs.build_rating_book(
            season,
            api_key=api_key,
            adjust_week=None,
            calibration_games=calibration_games,
        )
        if not fbs.RATING_WEIGHTS_PATH.exists():
            raise RuntimeError(f"Expected weights file {fbs.RATING_WEIGHTS_PATH} was not generated.")
        payload = _load_weights(fbs.RATING_WEIGHTS_PATH)
        meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
        offense_rows = float(meta.get("offense_rows") or len(calibration_games))
        power_rows = float(meta.get("power_rows") or len(calibration_games))
        _accumulate(payload.get("offense", {}), offense_rows, offense_totals)
        _accumulate(payload.get("defense", {}), offense_rows, defense_totals)
        _accumulate(payload.get("power", {}), power_rows, power_totals)
        offense_rows_total += offense_rows
        power_rows_total += power_rows

    if offense_rows_total <= 0 or power_rows_total <= 0:
        raise RuntimeError("No seasons produced usable weights.")

    def _normalize(totals: Dict[str, float], denom: float) -> Dict[str, float]:
        return {key: value / denom for key, value in totals.items()}

    final_payload = {
        "generated": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "offense": _normalize(offense_totals, offense_rows_total),
        "defense": _normalize(defense_totals, offense_rows_total),
        "power": _normalize(power_totals, power_rows_total),
        "meta": {
            "offense_rows": offense_rows_total,
            "defense_rows": offense_rows_total,
            "power_rows": power_rows_total,
            "seasons": seasons,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(final_payload, handle, indent=2, sort_keys=True)
    print(f"[ok] Wrote aggregated weights for seasons {seasons} to {args.output}")


if __name__ == "__main__":
    main()
