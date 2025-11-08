"""Helpers for working with The Odds API historical exports."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_closing_prices(
    base_dir: Path | str,
    *,
    year: int,
    provider: str,
    classification: str = "fbs",
) -> pd.DataFrame:
    """
    Load closing spreads/totals/moneylines from the odds-history archive.

    Parameters
    ----------
    base_dir:
        Directory containing ``season=YYYY/week=WW`` CSV files produced by
        ``scripts/fetch_the_odds_history.py``.
    year:
        Target season year.
    provider:
        Bookmaker key (e.g., ``fanduel``).
    classification:
        ``"fbs"`` or ``"fcs"``.
    """

    base = Path(base_dir)
    season_dir = base / f"season={year}"
    if not season_dir.exists():
        return pd.DataFrame(columns=["Spread", "SpreadPriceHome", "SpreadPriceAway", "Total", "TotalPriceOver", "TotalPriceUnder", "HomeMoneyline", "AwayMoneyline"])

    pattern = f"{provider.lower()}_{classification.lower()}.csv"
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(season_dir.glob("week=*/*")):
        if csv_path.name.lower() != pattern:
            continue
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            continue
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["Spread", "SpreadPriceHome", "SpreadPriceAway", "Total", "TotalPriceOver", "TotalPriceUnder", "HomeMoneyline", "AwayMoneyline"])

    combined = pd.concat(frames, ignore_index=True)

    def _ensure_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    spreads = (
        combined[combined["market"] == "spread"]
        .rename(
            columns={
                "game_id": "Id",
                "close_point": "Spread",
                "close_price_home": "SpreadPriceHome",
                "close_price_away": "SpreadPriceAway",
            }
        )[["Id", "Spread", "SpreadPriceHome", "SpreadPriceAway"]]
    )
    totals = (
        combined[combined["market"] == "total"]
        .rename(
            columns={
                "game_id": "Id",
                "close_point": "Total",
                "close_price_over": "TotalPriceOver",
                "close_price_under": "TotalPriceUnder",
            }
        )[["Id", "Total", "TotalPriceOver", "TotalPriceUnder"]]
    )
    money = (
        combined[combined["market"] == "moneyline"]
        .rename(
            columns={
                "game_id": "Id",
                "close_price_home": "HomeMoneyline",
                "close_price_away": "AwayMoneyline",
            }
        )[["Id", "HomeMoneyline", "AwayMoneyline"]]
    )

    merged = (
        spreads.merge(totals, on="Id", how="outer")
        .merge(money, on="Id", how="outer")
        .drop_duplicates("Id", keep="last")
    )
    for column in ["Spread", "SpreadPriceHome", "SpreadPriceAway", "Total", "TotalPriceOver", "TotalPriceUnder", "HomeMoneyline", "AwayMoneyline"]:
        if column in merged.columns:
            merged[column] = _ensure_numeric(merged[column])

    merged["Spread"] = -merged["Spread"]  # convert to home-minus-away margin
    return merged.set_index("Id")
