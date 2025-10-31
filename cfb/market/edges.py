"""Edge and closing-line utilities shared across simulations and backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class EdgeFilterConfig:
    """Threshold values for filtering model edges."""

    spread_edge_min: float = 0.0
    total_edge_min: float = 0.0
    win_prob_edge_min: float = 0.0
    min_provider_count: int = 0


def _resolve_column(df: pd.DataFrame, column: Optional[str]) -> Optional[str]:
    if not column:
        return None
    return column if column in df.columns else None


def annotate_edges(
    df: pd.DataFrame,
    *,
    model_spread_col: str,
    market_spread_col: Optional[str],
    model_total_col: str,
    market_total_col: Optional[str],
    win_prob_col: Optional[str] = None,
    provider_count_col: Optional[str] = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with standardised edge columns.

    Parameters
    ----------
    df :
        Dataframe containing model projections and (optionally) market lines.
    model_spread_col :
        Column holding the model spread (home minus away).
    market_spread_col :
        Column holding the market spread (home minus away). When missing,
        spread edge columns are filled with ``NaN``.
    model_total_col :
        Column with the model total points projection.
    market_total_col :
        Column with the market total points. Required for total edges.
    win_prob_col :
        Optional column with the model home win probability, used to
        derive probability edges if present.
    provider_count_col :
        Optional column with the number of contributing sportsbooks.
    """

    if df.empty:
        return df.copy()

    frame = df.copy()
    market_spread_col = _resolve_column(frame, market_spread_col)
    market_total_col = _resolve_column(frame, market_total_col)
    provider_count_col = _resolve_column(frame, provider_count_col)
    win_prob_col = _resolve_column(frame, win_prob_col)

    spread_series = frame[model_spread_col]
    market_spread_series = frame[market_spread_col] if market_spread_col else pd.Series(float("nan"), index=frame.index)
    frame["spread_edge"] = spread_series - market_spread_series
    frame["spread_edge_abs"] = frame["spread_edge"].abs()
    frame["spread_edge_direction"] = frame["spread_edge"].apply(
        lambda value: "home" if pd.notna(value) and value >= 0 else ("away" if pd.notna(value) else None)
    )

    total_series = frame[model_total_col]
    market_total_series = frame[market_total_col] if market_total_col else pd.Series(float("nan"), index=frame.index)
    frame["total_edge"] = total_series - market_total_series
    frame["total_edge_abs"] = frame["total_edge"].abs()

    if win_prob_col:
        if "market_home_implied_prob" in frame.columns:
            market_prob_series = frame["market_home_implied_prob"]
        else:
            market_prob_series = pd.Series(float("nan"), index=frame.index)
        frame["win_prob_edge"] = frame[win_prob_col] - market_prob_series
        frame["win_prob_edge_abs"] = frame["win_prob_edge"].abs()
    else:
        frame["win_prob_edge"] = float("nan")
        frame["win_prob_edge_abs"] = float("nan")

    if provider_count_col:
        frame["provider_count"] = frame[provider_count_col]
    elif "provider_count" not in frame.columns:
        frame["provider_count"] = float("nan")

    return frame


def edge_filter_mask(
    df: pd.DataFrame,
    config: EdgeFilterConfig,
    *,
    spread_edge_col: str = "spread_edge",
    total_edge_col: str = "total_edge",
    win_prob_edge_col: str = "win_prob_edge",
    provider_count_col: str = "provider_count",
) -> pd.Series:
    """Return a boolean mask indicating rows that satisfy the edge thresholds."""

    if df.empty:
        return pd.Series([], dtype=bool)

    mask = pd.Series(True, index=df.index)

    if spread_edge_col in df and config.spread_edge_min > 0.0:
        mask &= df[spread_edge_col].abs() >= config.spread_edge_min

    if total_edge_col in df and config.total_edge_min > 0.0:
        mask &= df[total_edge_col].abs() >= config.total_edge_min

    if win_prob_edge_col in df and config.win_prob_edge_min > 0.0:
        mask &= df[win_prob_edge_col].abs() >= config.win_prob_edge_min

    if provider_count_col in df and config.min_provider_count > 0:
        mask &= df[provider_count_col].fillna(0).astype(int) >= config.min_provider_count

    return mask


def filter_edges(df: pd.DataFrame, config: EdgeFilterConfig, **kwargs) -> pd.DataFrame:
    """Convenience helper that returns rows meeting the configured thresholds."""

    mask = edge_filter_mask(df, config, **kwargs)
    return df.loc[mask].copy()


def allow_spread_bet(
    spread_edge: Optional[float],
    provider_count: Optional[int],
    config: EdgeFilterConfig,
) -> bool:
    """Determine if a spread bet passes the configured thresholds."""

    if spread_edge is None or pd.isna(spread_edge):
        return False
    if config.spread_edge_min > 0.0 and abs(spread_edge) < config.spread_edge_min:
        return False
    if config.min_provider_count > 0:
        count = 0 if provider_count is None or pd.isna(provider_count) else int(provider_count)
        if count < config.min_provider_count:
            return False
    return True


def allow_total_bet(
    total_edge: Optional[float],
    provider_count: Optional[int],
    config: EdgeFilterConfig,
) -> bool:
    """Determine if a total bet passes the configured thresholds."""

    if total_edge is None or pd.isna(total_edge):
        return False
    if config.total_edge_min > 0.0 and abs(total_edge) < config.total_edge_min:
        return False
    if config.min_provider_count > 0:
        count = 0 if provider_count is None or pd.isna(provider_count) else int(provider_count)
        if count < config.min_provider_count:
            return False
    return True
