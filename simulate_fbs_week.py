"""Simulate an entire FBS week with opponent-adjusted power updates."""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import oddslogic_loader
from cfb.io import oddslogic as oddslogic_io
from cfb.io import the_odds_api
from cfb.sim import fbs as sim_fbs

LOGO_PATH = Path("assets/betting_worldwide_logo.png")
FRESHNESS_THRESHOLD = timedelta(hours=12)
PROVIDER_WEIGHTS: dict[str, float] = {
    "circa": 1.0,
    "sports411": 1.0,
    "pinnacle": 1.0,
    "consensus": 0.8,
    "betonline": 0.7,
    "fanduel": 0.6,
    "buckeye": 0.6,
    "wph": 0.6,
}
DEFAULT_PROVIDER_WEIGHT = 0.4


def _ensure_oddslogic_lines(
    df: pd.DataFrame,
    *,
    providers: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Refresh market lines using live OddsLogic data for stale providers."""

    target_providers = [p.strip() for p in providers or [] if p and p.strip()]
    if not target_providers:
        target_providers = ["Fanduel", "Sports411"]

    def _parse_timestamp(value: object) -> datetime | None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except (ValueError, OSError, OverflowError):
                return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        return None

    now_utc = datetime.now(timezone.utc)
    freshness_threshold = FRESHNESS_THRESHOLD

    def _row_is_stale(row: pd.Series) -> bool:
        raw = row.get("market_provider_lines")
        if not isinstance(raw, str) or not raw.strip():
            return True
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return True
        targets = {p.lower().replace(" ", "") for p in target_providers}
        freshest: datetime | None = None
        found_target = False
        for name, info in payload.items():
            normalized = name.lower().replace(" ", "")
            if targets and normalized not in targets:
                continue
            found_target = True
            ts = _parse_timestamp(info.get("last_updated") or info.get("spread_updated") or info.get("total_updated"))
            if ts is None:
                return True
            if freshest is None or ts > freshest:
                freshest = ts
        if not targets:
            found_target = True if payload else False
            if payload:
                for info in payload.values():
                    ts = _parse_timestamp(info.get("last_updated") or info.get("spread_updated") or info.get("total_updated"))
                    if ts is None:
                        return True
                    if freshest is None or ts > freshest:
                        freshest = ts
        if not found_target:
            return True
        if freshest is None:
            return True
        return (now_utc - freshest) > freshness_threshold

    needs_refresh = df.apply(_row_is_stale, axis=1)
    if not needs_refresh.any():
        return df

    kickoff_dates = {
        start_dt.date()
        for start_dt in pd.to_datetime(df.loc[needs_refresh, "start_date"], errors="coerce", utc=True)
        if not pd.isna(start_dt)
    }
    if not kickoff_dates:
        return df

    live_lookup = oddslogic_io.fetch_live_market_lookup(
        kickoff_dates,
        classification="fbs",
        providers=None,
    )
    if not live_lookup:
        return df

    enriched = df.copy()
    for idx, row in enriched.loc[needs_refresh].iterrows():
        existing_spread = row.get("market_spread")
        existing_count = row.get("market_provider_count") or 0
        if pd.notna(existing_spread) and existing_count and existing_count >= len(target_providers):
            continue
        start_dt = pd.to_datetime(row.get("start_date"), errors="coerce", utc=True)
        if pd.isna(start_dt):
            continue
        kickoff_date = start_dt.date()
        home_key = oddslogic_loader.normalize_label(str(row.get("home_team") or ""))
        away_key = oddslogic_loader.normalize_label(str(row.get("away_team") or ""))

        entry = live_lookup.get((kickoff_date, home_key, away_key))
        invert = False
        if entry is None:
            entry = live_lookup.get((kickoff_date, away_key, home_key))
            if entry:
                invert = True
        if entry is None:
            continue

        existing_raw = row.get("market_provider_lines")
        if isinstance(existing_raw, str) and existing_raw.strip():
            try:
                provider_lines = json.loads(existing_raw)
            except json.JSONDecodeError:
                provider_lines = {}
        else:
            provider_lines = {}

        for name, payload in entry.get("providers", {}).items():
            spread_val = payload.get("spread_value")
            open_spread = payload.get("open_spread_value")
            if spread_val is not None:
                spread_val = float(spread_val)
                if invert:
                    spread_val = -spread_val
            if open_spread is not None:
                open_spread = float(open_spread)
                if invert:
                    open_spread = -open_spread
            total_val = payload.get("total_value")
            if total_val is not None:
                total_val = float(total_val)
            open_total = payload.get("open_total_value")
            if open_total is not None:
                open_total = float(open_total)
            provider_lines[name] = {
                "spread_home": spread_val,
                "spread_away": -spread_val if spread_val is not None else None,
                "total": total_val,
                "home_moneyline": payload.get("home_moneyline"),
                "away_moneyline": payload.get("away_moneyline"),
                "last_updated": payload.get("spread_updated") or payload.get("total_updated"),
                "source": "OddsLogic",
                "open_spread_home": open_spread,
                "open_total": open_total,
                "spread_updated": payload.get("spread_updated"),
                "total_updated": payload.get("total_updated"),
                "staleness_hours": None,
            }
            ts = _parse_timestamp(provider_lines[name]["last_updated"])
            if ts is not None:
                provider_lines[name]["staleness_hours"] = max(0.0, (now_utc - ts).total_seconds() / 3600.0)

        if not provider_lines:
            continue

        providers_sorted = sorted(provider_lines.keys())

        def _staleness(info: dict[str, object]) -> timedelta:
            ts = _parse_timestamp(info.get("last_updated"))
            if ts is None:
                return timedelta.max
            return now_utc - ts

        primary_name = None
        lookup_map = {p.lower().replace(" ", ""): p for p in providers_sorted}
        for target in target_providers:
            normalized = target.lower().replace(" ", "")
            provider_name = lookup_map.get(normalized)
            if not provider_name:
                continue
            info = provider_lines.get(provider_name) or {}
            if _staleness(info) <= freshness_threshold:
                primary_name = provider_name
                break
        if primary_name is None:
            best_name = None
            best_ts = None
            for name, info in provider_lines.items():
                ts = _parse_timestamp(info.get("last_updated"))
                if ts is None:
                    continue
                if best_ts is None or ts > best_ts:
                    best_ts = ts
                    best_name = name
            primary_name = best_name or (providers_sorted[0] if providers_sorted else None)

        enriched.at[idx, "market_provider_lines"] = json.dumps(provider_lines, sort_keys=True)
        enriched.at[idx, "market_providers"] = ", ".join(providers_sorted)
        enriched.at[idx, "market_provider_count"] = len(providers_sorted)
        enriched.at[idx, "market_primary_provider"] = primary_name

        primary_info = provider_lines.get(primary_name or "")
        if primary_info:
            spread_home = primary_info.get("spread_home")
            if spread_home is not None:
                if _staleness(primary_info) <= freshness_threshold:
                    market_spread = -float(spread_home)
                    enriched.at[idx, "market_spread"] = market_spread
                    model_spread = enriched.at[idx, "model_spread"]
                    if pd.notna(model_spread):
                        enriched.at[idx, "spread_vs_market"] = float(model_spread) - market_spread
            total_val = primary_info.get("total")
            if total_val is not None:
                if _staleness(primary_info) <= freshness_threshold:
                    enriched.at[idx, "market_total"] = float(total_val)
                    model_total = enriched.at[idx, "model_total"]
                    if pd.notna(model_total):
                        enriched.at[idx, "total_vs_market"] = float(model_total) - float(total_val)

    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate all FBS games for a given week.")
    parser.add_argument("week", type=int, help="Target week to simulate.")
    parser.add_argument("--year", type=int, default=2024, help="Season year (default 2024).")
    parser.add_argument("--api-key", type=str, help="Optional CFBD API key (defaults to env).")
    parser.add_argument("--neutral-default", action="store_true",
                        help="Treat games with missing neutral flag as neutral.")
    parser.add_argument(
        "--season-type",
        choices=["regular", "postseason"],
        default="regular",
        help="Season type to query (default regular).",
    )
    parser.add_argument(
        "--include-completed",
        action="store_true",
        help="Include games that CFBD already marks as completed (useful for retrospective sims).",
    )
    parser.add_argument("--output", type=Path, help="Optional CSV output path.")
    parser.add_argument("--html", type=Path, help="Optional HTML summary output path (shareable).")
    parser.add_argument(
        "--providers",
        type=str,
        help="Comma-separated sportsbook providers to include (default: all available).",
    )
    return parser.parse_args()


def _logo_data_uri() -> str | None:
    if not LOGO_PATH.exists():
        return None
    try:
        data = LOGO_PATH.read_bytes()
    except OSError:
        return None
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_html(df: pd.DataFrame, *, title: str, providers: Sequence[str] | None = None) -> str:
    df = df.copy()
    df = _ensure_oddslogic_lines(df, providers=providers)
    df["start_dt"] = pd.to_datetime(df["start_date"], errors="coerce", utc=True)
    df["kickoff_display"] = df["start_dt"].dt.tz_convert("US/Eastern").dt.strftime("%a %m/%d %I:%M %p ET")

    total_games = len(df)
    actionable_spreads: list[float] = []
    actionable_totals: list[float] = []
    avg_edge = np.nan
    providers_seen: set[str] = set()

    def _format_providers(raw: str | float) -> str:
        names = [p.strip() for p in str(raw or "").split(",") if p.strip()]
        for name in names:
            providers_seen.add(name)
        if not names:
            return "<span class='chip muted'>—</span>"
        label = "1 book" if len(names) == 1 else f"{len(names)} books"
        return f"<span class='chip'>{label}</span>"

    def _edge_class(edge: float | None) -> str:
        if edge is None or not np.isfinite(edge):
            return "edge light neutral"
        magnitude = abs(edge)
        band = "strong" if magnitude >= 1.5 else "medium" if magnitude >= 1.0 else "light"
        direction = "positive" if edge >= 0 else "negative"
        return f"edge {direction} {band}"

    def _kelly_fraction(model_spread: float | None, market_spread: float | None, sigma: float | None, edge: float | None) -> float | None:
        if not np.isfinite(model_spread) or not np.isfinite(market_spread) or not np.isfinite(sigma) or sigma <= 0 or edge is None:
            return None
        z = (model_spread - market_spread) / sigma
        cover_prob_home = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
        if edge >= 0:
            p = float(cover_prob_home)
        else:
            p = float(1.0 - cover_prob_home)
        b = 100.0 / 110.0
        kelly = (b * p - (1 - p)) / b
        return max(0.0, kelly)

    def _format_weather(row: pd.Series) -> str:
        bits: list[str] = []
        condition = row.get("weather_condition") or ""
        if isinstance(condition, str) and condition and condition.lower() not in {"clear", "fair"}:
            bits.append(condition.title())
        temp = row.get("weather_temp")
        if pd.notna(temp):
            bits.append(f"{temp:.0f}°F")
        wind = row.get("weather_wind")
        if pd.notna(wind):
            bits.append(f"{wind:.0f} mph wind")
        return " · ".join(bits) if bits else "—"

    def _flag_badges(row: pd.Series, edge: float | None) -> str:
        flags: list[str] = []
        if pd.isna(row.get("market_spread")) and pd.isna(row.get("market_total")):
            flags.append("NO-MKT")
        if abs(row.get("weather_total_adj") or 0.0) >= 1.0:
            flags.append("WX")
        if (row.get("market_provider_count") or 0) <= 1:
            flags.append("STALE")
        if row.get("neutral"):
            flags.append("NEU")
        if edge is not None and abs(edge) <= 0.5:
            flags.append("SMALL")
        return "".join(f"<span class='flag'>{code}</span>" for code in flags) or "<span class='flag muted'>—</span>"

    def _format_kelly(kelly: float | None) -> str:
        if kelly is None or not np.isfinite(kelly) or kelly <= 0:
            return "—"
        return f"{kelly * 0.5:.1%}"

    def _format_edge_pct(p: float | None) -> str:
        if p is None or not np.isfinite(p):
            return "—"
        return f"{p:+.1f}%"

    def _prob_to_moneyline(prob: float | None) -> float | None:
        if prob is None or not np.isfinite(prob):
            return None
        if prob <= 0.0 or prob >= 1.0:
            return None
        if prob >= 0.5:
            return -100.0 * prob / (1.0 - prob)
        return 100.0 * (1.0 - prob) / prob

    def _format_moneyline(value: float | None) -> str:
        if value is None or not np.isfinite(value):
            return "—"
        return f"{float(value):+.0f}"

    def _to_float(value: object) -> float | None:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None
        return val if np.isfinite(val) else None

    def _infer_home_margin(model_val: float | None, market_val: float | None) -> float | None:
        if market_val is None or not np.isfinite(market_val):
            return market_val
        if model_val is None or not np.isfinite(model_val):
            return market_val
        alt = -market_val
        model_sign = 0 if model_val == 0 else math.copysign(1.0, model_val)
        market_sign = 0 if market_val == 0 else math.copysign(1.0, market_val)
        if model_sign != 0 and model_sign != market_sign:
            return market_val
        if abs(model_val - alt) < abs(model_val - market_val):
            return alt
        return market_val

    records_by_game: dict[str, dict[str, object]] = {}
    for _, row in df.iterrows():
        providers_html = _format_providers(row.get("market_providers") or "")
        weather_html = _format_weather(row)
        kickoff = row.get("kickoff_display") or row.get("start_date") or ""
        kickoff_dt = row.get("start_dt")
        if isinstance(kickoff_dt, pd.Timestamp) and pd.notna(kickoff_dt):
            kickoff_sort = kickoff_dt.value
        else:
            kickoff_sort = float("inf")
        sigma = row.get("prob_sigma")
        game_label = f"{row.get('away_team')} @ {row.get('home_team')}"
        record = records_by_game.setdefault(
            game_label,
            {
                "kickoff": kickoff,
                "kickoff_sort": kickoff_sort,
                "game": game_label,
                "spread_row": None,
                "total_row": None,
                "model_home_ml": None,
            },
        )
        if record["kickoff"] in ("", None) and kickoff:
            record["kickoff"] = kickoff
        if record.get("kickoff_sort", float("inf")) > kickoff_sort:
            record["kickoff_sort"] = kickoff_sort

        provider_lines_raw = row.get("market_provider_lines")
        if isinstance(provider_lines_raw, str) and provider_lines_raw:
            try:
                provider_lines = json.loads(provider_lines_raw)
            except json.JSONDecodeError:
                provider_lines = {}
        elif isinstance(provider_lines_raw, dict):
            provider_lines = provider_lines_raw
        else:
            provider_lines = {}

        kickoff_dt = row.get("start_dt")
        now_utc = pd.Timestamp.now(tz="UTC")
        game_started = (
            isinstance(kickoff_dt, pd.Timestamp)
            and not pd.isna(kickoff_dt)
            and now_utc >= kickoff_dt
        )
        provider_lines_for_use = provider_lines if not game_started else {}

        def _extract_book(book_key: str) -> tuple[float | None, float | None, float | None, float | None]:
            key_target = "".join(ch.lower() for ch in book_key if ch.isalnum())
            spread_current = spread_open = total_current = total_open = None
            for name, info in provider_lines_for_use.items():
                normalized = "".join(ch.lower() for ch in (name or "") if ch.isalnum())
                if normalized != key_target:
                    continue
                spread_home = info.get("spread_home")
                if spread_home is not None and not pd.isna(spread_home):
                    spread_current = -float(spread_home)
                spread_open_home = info.get("open_spread_home")
                if spread_open_home is not None and not pd.isna(spread_open_home):
                    spread_open = -float(spread_open_home)
                total_val = info.get("total")
                if total_val is not None and not pd.isna(total_val):
                    total_current = float(total_val)
                open_total = info.get("open_total")
                if open_total is not None and not pd.isna(open_total):
                    total_open = float(open_total)
            return spread_current, spread_open, total_current, total_open

        fd_spread, fd_open_spread, fd_total, fd_open_total = _extract_book("fanduel")
        s411_spread, s411_open_spread, s411_total, s411_open_total = _extract_book("sports411")

        def _weighted_market_lines(data: dict[str, dict]) -> tuple[float | None, float | None]:
            if not data:
                return None, None
            spread_sum = spread_weight = 0.0
            total_sum = total_weight = 0.0
            for name, info in data.items():
                normalized = "".join(ch.lower() for ch in (name or "") if ch.isalnum())
                weight = PROVIDER_WEIGHTS.get(normalized, DEFAULT_PROVIDER_WEIGHT)
                staleness = info.get("staleness_hours")
                if staleness is not None and staleness > FRESHNESS_THRESHOLD.total_seconds() / 3600.0:
                    continue
                spread_home = info.get("spread_home")
                if spread_home is not None and not pd.isna(spread_home):
                    home_margin = -float(spread_home)
                    spread_sum += weight * home_margin
                    spread_weight += weight
                total_val = info.get("total")
                if total_val is not None and not pd.isna(total_val):
                    total_sum += weight * float(total_val)
                    total_weight += weight
            spread_value = (spread_sum / spread_weight) if spread_weight else None
            total_value = (total_sum / total_weight) if total_weight else None
            return spread_value, total_value

        weighted_spread_margin, weighted_total_value = _weighted_market_lines(provider_lines_for_use)
        record["weighted_spread_margin"] = weighted_spread_margin
        record["weighted_total_value"] = weighted_total_value

        model_spread_val = _to_float(row.get("model_spread"))
        market_spread_book = _to_float(row.get("market_spread"))
        home_prob = _to_float(row.get("model_home_win_prob"))
        model_home_ml = _prob_to_moneyline(home_prob)
        if model_home_ml is not None:
            record["model_home_ml"] = model_home_ml

        if model_spread_val is not None and market_spread_book is not None:
            market_home_margin = _infer_home_margin(model_spread_val, market_spread_book)
            raw_edge = model_spread_val - market_home_margin
            bet_home = raw_edge >= 0
            bet_side = row.get("home_team") if bet_home else row.get("away_team")
            bet_line = None
            if market_spread_book is not None and np.isfinite(market_spread_book):
                bet_line = -market_spread_book if bet_home else market_spread_book
            model_line = -model_spread_val if bet_home else model_spread_val
            edge_value = raw_edge
            weighted_spread_line = None
            if weighted_spread_margin is not None:
                weighted_spread_line = -weighted_spread_margin if bet_home else weighted_spread_margin

            cover_prob_shift = None
            kelly = None
            if np.isfinite(sigma) and sigma > 0:
                z = raw_edge / sigma
                home_cover_prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
                target_prob = home_cover_prob if bet_home else 1.0 - home_cover_prob
                cover_prob_shift = (target_prob - 0.5) * 100
                b = 100.0 / 110.0
                kelly_val = (b * target_prob - (1 - target_prob)) / b
                kelly = max(0.0, kelly_val)

            weighted_spread_line = None
            if weighted_spread_margin is not None:
                weighted_spread_line = -weighted_spread_margin if bet_home else weighted_spread_margin

            bet_label = f"{bet_side} {bet_line:+.1f}" if bet_line is not None else bet_side or "--"
            record["spread_row"] = {
                "row_type": "spread",
                "market": "Spread",
                "bet_label": bet_label,
                "game": game_label,
                "model_value": model_line,
                "market_value": bet_line,
                "weighted_spread": weighted_spread_line,
                "weighted_total": None,
                "model_home_ml": model_home_ml,
                "edge": edge_value,
                "edge_pct": cover_prob_shift,
                "kelly": kelly,
                "providers_html": providers_html,
                "weather_html": weather_html,
                "kickoff": kickoff,
                "kickoff_sort": record.get("kickoff_sort", float("inf")),
                "flags_html": _flag_badges(row, raw_edge),
                "edge_class": _edge_class(edge_value),
                "primary": row.get("market_primary_provider") or "—",
                "fanduel_open": fd_open_spread,
                "fanduel_current": fd_spread,
                "sports411_open": s411_open_spread,
                "sports411_current": s411_spread,
            }
        else:
            model_spread = model_spread_val
            spread_row = {
                    "row_type": "spread",
                    "market": "Spread (model)",
                    "bet_label": f"{row.get('home_team')} {model_spread:+.1f}" if pd.notna(model_spread) else f"{row.get('home_team')} model",
                    "game": game_label,
                    "model_value": model_spread,
                    "market_value": None,
                    "weighted_spread": weighted_spread_margin,
                    "weighted_total": None,
                    "model_home_ml": model_home_ml,
                    "edge": None,
                    "edge_pct": None,
                    "kelly": None,
                "providers_html": "<span class='chip muted'>—</span>",
                "weather_html": weather_html,
                "kickoff": kickoff,
                "kickoff_sort": record.get("kickoff_sort", float("inf")),
                "flags_html": _flag_badges(row, None),
                "edge_class": _edge_class(None),
                "primary": "—",
                "fanduel_open": fd_open_spread,
                "fanduel_current": fd_spread,
                    "sports411_open": s411_open_spread,
                    "sports411_current": s411_spread,
                }
            record["spread_row"] = spread_row

        model_total_val = _to_float(row.get("model_total"))
        market_total_book = _to_float(row.get("market_total"))

        if model_total_val is not None and market_total_book is not None:
            raw_edge_total = model_total_val - market_total_book
            bet_over = raw_edge_total >= 0
            total_sigma = 12.5
            cover_prob_shift = None
            kelly_total = None
            if total_sigma > 0:
                z = raw_edge_total / total_sigma
                over_prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
                target_prob = over_prob if bet_over else 1.0 - over_prob
                cover_prob_shift = (target_prob - 0.5) * 100
                b = 100.0 / 110.0
                kelly_val = (b * target_prob - (1 - target_prob)) / b
                kelly_total = max(0.0, kelly_val)
            bet_label = f"{'Over' if bet_over else 'Under'} {market_total_book:.1f}"
            record["total_row"] = {
                "row_type": "total",
                "market": "Total",
                "bet_label": bet_label,
                "game": game_label,
                "model_value": model_total_val,
                "market_value": market_total_book,
                "weighted_spread": None,
                "weighted_total": weighted_total_value,
                "model_home_ml": model_home_ml,
                "edge": raw_edge_total,
                "edge_pct": cover_prob_shift,
                "kelly": kelly_total,
                "providers_html": providers_html,
                "weather_html": weather_html,
                "kickoff": kickoff,
                "kickoff_sort": record.get("kickoff_sort", float("inf")),
                "flags_html": _flag_badges(row, raw_edge_total),
                "edge_class": _edge_class(raw_edge_total),
                "primary": row.get("market_primary_provider") or "—",
                "fanduel_open": fd_open_total,
                "fanduel_current": fd_total,
                "sports411_open": s411_open_total,
                "sports411_current": s411_total,
            }
        else:
            model_total = row.get("model_total")
            total_row = {
                    "row_type": "total",
                    "market": "Total (model)",
                    "bet_label": f"Total {model_total:.1f}" if pd.notna(model_total) else "Total",
                    "game": game_label,
                    "model_value": model_total,
                    "market_value": None,
                    "weighted_spread": None,
                    "weighted_total": weighted_total_value,
                    "model_home_ml": model_home_ml,
                    "edge": None,
                    "edge_pct": None,
                    "kelly": None,
                    "providers_html": "<span class='chip muted'>—</span>",
                    "weather_html": weather_html,
                    "kickoff": kickoff,
                    "kickoff_sort": record.get("kickoff_sort", float("inf")),
                    "flags_html": _flag_badges(row, None),
                    "edge_class": _edge_class(None),
                    "primary": "—",
                    "fanduel_open": fd_open_total,
                    "fanduel_current": fd_total,
                    "sports411_open": s411_open_total,
                    "sports411_current": s411_total,
                }
            record["total_row"] = total_row

    records: list[dict] = []
    paired_rows: list[tuple[dict, dict]] = []
    for record in sorted(records_by_game.values(), key=lambda r: r["kickoff"] or ""):
        spread_row = record.get("spread_row")
        total_row = record.get("total_row")
        fallback_ml = record.get("model_home_ml")
        if spread_row is None and total_row is None:
            continue
        if spread_row is None:
            spread_row = {
                "row_type": "spread",
                "market": "Spread (model)",
                "bet_label": "Model only",
                "game": record["game"],
                "model_value": None,
                "market_value": None,
                "weighted_spread": None,
                "weighted_total": None,
                "model_home_ml": fallback_ml,
                "edge": None,
                "edge_pct": None,
                "kelly": None,
                "providers_html": "<span class='chip muted'>—</span>",
                "weather_html": "—",
                "kickoff": record["kickoff"],
                "flags_html": "<span class='flag muted'>—</span>",
                "edge_class": _edge_class(None),
                "primary": "—",
            }
        if total_row is None:
            total_row = {
                "row_type": "total",
                "market": "Total (model)",
                "bet_label": "Model only",
                "game": record["game"],
                "model_value": None,
                "market_value": None,
                "weighted_spread": None,
                "weighted_total": None,
                "model_home_ml": fallback_ml,
                "edge": None,
                "edge_pct": None,
                "kelly": None,
                "providers_html": "<span class='chip muted'>—</span>",
                "weather_html": "—",
                "kickoff": record["kickoff"],
                "flags_html": "<span class='flag muted'>—</span>",
                "edge_class": _edge_class(None),
                "primary": "—",
            }
        records.append(spread_row)
        records.append(total_row)
        paired_rows.append((spread_row, total_row))

    def _format_value(value: float | None, precision: int = 1, signed: bool = True) -> str:
        if value is None or not np.isfinite(value):
            return "—"
        fmt = f"{{:{'+' if signed else ''}.{precision}f}}"
        return fmt.format(value)

    spread_edges = [
        rec["edge"]
        for rec in records
        if rec.get("row_type") == "spread" and rec.get("edge") is not None
    ]
    total_edges = [
        rec["edge"]
        for rec in records
        if rec.get("row_type") == "total" and rec.get("edge") is not None
    ]
    actionable_spreads = [abs(val) for val in spread_edges if val is not None and abs(val) >= 1.0]
    actionable_totals = [abs(val) for val in total_edges if val is not None and abs(val) >= 1.5]
    avg_edge = float(np.mean(actionable_spreads)) if actionable_spreads else np.nan

    table_rows = []
    for spread_row, total_row in paired_rows:
        for rec in (spread_row, total_row):
            row_type = rec.get("row_type")
            row_class = "row-total" if row_type == "total" else "row-spread"
            game_display = rec["game"] if row_type == "spread" else ""
            bet_display = rec["bet_label"]
            if row_type == "total":
                bet_display = f"<span class='total-label'>{bet_display}</span>"
            model_signed = row_type not in {"spread", "total"}
            table_rows.append(
                f"<tr class='{row_class}'>"
                f"<td class='cell-bet'>{bet_display}</td>"
                f"<td class='cell-game'>{game_display}</td>"
                f"<td class='cell-market'>{rec['market']}</td>"
                f"<td>{_format_value(rec['model_value'], signed=model_signed)}</td>"
                f"<td>{_format_value(rec['market_value'])}</td>"
                f"<td>{_format_value(rec.get('weighted_spread'))}</td>"
                f"<td>{_format_value(rec.get('weighted_total'))}</td>"
                f"<td>{_format_moneyline(rec.get('model_home_ml'))}</td>"
                f"<td>{_format_value(rec.get('fanduel_open'))}</td>"
                f"<td>{_format_value(rec.get('fanduel_current'))}</td>"
                f"<td>{_format_value(rec.get('sports411_open'))}</td>"
                f"<td>{_format_value(rec.get('sports411_current'))}</td>"
                f"<td><span class='{rec['edge_class']}'>{_format_value(rec['edge'])}</span></td>"
                f"<td>{_format_edge_pct(rec['edge_pct'])}</td>"
                f"<td>{_format_kelly(rec['kelly'])}</td>"
                f"<td>{rec['providers_html']}</td>"
                f"<td class='cell-primary'>{rec['primary']}</td>"
                f"<td>{rec['weather_html']}</td>"
                f"<td>{rec['kickoff']}</td>"
                f"<td>{rec['flags_html']}</td>"
                "</tr>"
            )

    top_spreads = sorted(
        (rec for rec in records if rec.get("row_type") == "spread" and rec["edge"] is not None),
        key=lambda r: abs(r["edge"]),
        reverse=True,
    )[:5]
    top_totals = sorted(
        (rec for rec in records if rec.get("row_type") == "total" and rec["edge"] is not None),
        key=lambda r: abs(r["edge"]),
        reverse=True,
    )[:5]

    def _top_list(items: list[dict]) -> str:
        if not items:
            return "<li>No qualifying edges.</li>"
        return "".join(
            f"<li><span class='edge-badge {_edge_class(item['edge'])}'>{_format_value(item['edge'])}</span>"
            f"<span class='top-game'>{item['bet_label']}</span>"
            f"<span class='top-note'>{item['game']}</span></li>"
            for item in items
        )

    kelly_values = [
        rec["kelly"]
        for rec in records
        if rec["kelly"] is not None and np.isfinite(rec["kelly"]) and rec["kelly"] > 0
    ]
    avg_edge_display = _format_value(avg_edge) if not np.isnan(avg_edge) else "—"

    top_overall = sorted(
        (rec for rec in records if rec["edge"] is not None),
        key=lambda r: abs(r["edge"]),
        reverse=True,
    )[:5]
    headline_edges = "".join(
        f"<div class='edge-callout'>"
        f"<span class='edge-badge {_edge_class(item['edge'])}'>{_format_value(item['edge'])}</span>"
        f"<span class='edge-bet'>{item['bet_label']}</span>"
        f"<span class='edge-ev'>{_format_edge_pct(item['edge_pct'])}</span>"
        f"<span class='edge-game'>{item['game']}</span>"
        "</div>"
        for item in top_overall
    ) or "<div class='edge-callout empty'>No notable edges.</div>"
    logo_data = _logo_data_uri()
    logo_html = (
        f"<img src='{logo_data}' alt='Betting Worldwide logo' class='brand-logo' />" if logo_data else ""
    )

    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>{title}</title>
        <style>
          :root {{
            --bg: #f5f7fb;
            --card: #ffffff;
            --card-border: #dfe3eb;
            --text: #1f2630;
            --muted: #687385;
            --edge-strong: #03835a;
            --edge-medium: #d48806;
            --edge-light: #7b8899;
            --edge-negative: #c23f31;
            font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
          }}
          body {{
            background: var(--bg);
            color: var(--text);
            margin: 0;
          }}
          header {{
            padding: 32px 40px 20px;
            background: var(--card);
            border-bottom: 1px solid var(--card-border);
            display: flex;
            flex-direction: column;
            gap: 16px;
          }}
          .brand {{
            display: flex;
            gap: 16px;
            align-items: center;
          }}
          .brand-logo {{
            width: 72px;
            height: 72px;
            object-fit: contain;
            border-radius: 50%;
            border: 2px solid rgba(0,0,0,0.08);
          }}
          .brand-copy {{
            display: flex;
            flex-direction: column;
          }}
          header h1 {{
            margin: 0;
            font-size: 28px;
          }}
          .brand-tagline {{
            margin: 6px 0 0;
            color: var(--muted);
          }}
          .edge-callouts {{
            display: grid;
            gap: 10px;
          }}
          .edge-callout {{
            display: grid;
            grid-template-columns: auto auto auto 1fr;
            gap: 12px;
            align-items: center;
            background: rgba(3, 131, 90, 0.05);
            border-radius: 10px;
            padding: 8px 12px;
          }}
          .edge-callout.empty {{
            background: rgba(104, 115, 133, 0.08);
            color: var(--muted);
            font-style: italic;
          }}
          .edge-callout .edge-bet {{
            font-weight: 600;
          }}
          .edge-callout .edge-ev {{
            font-weight: 600;
            color: var(--muted);
          }}
          .edge-callout .edge-game {{
            color: var(--muted);
            font-size: 13px;
          }}
          .layout {{
            display: grid;
            grid-template-columns: minmax(260px, 320px) 1fr;
            gap: 28px;
            padding: 24px 40px 40px;
          }}
          .sidebar {{
            position: sticky;
            top: 24px;
            align-self: start;
            display: flex;
            flex-direction: column;
            gap: 24px;
          }}
          .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
          }}
          .kpi {{
            background: var(--card);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 12px 16px;
          }}
          .kpi-label {{
            display: block;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--muted);
          }}
          .kpi-value {{
            font-size: 20px;
            font-weight: 600;
            margin-top: 4px;
          }}
          .top-card {{
            background: var(--card);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 14px 16px;
          }}
          .top-card h3 {{
            margin: 0 0 10px;
            font-size: 16px;
          }}
          .top-list {{
            list-style: none;
            margin: 0;
            padding: 0;
            display: grid;
            gap: 10px;
          }}
          .top-list li {{
            display: flex;
            flex-direction: column;
            gap: 2px;
          }}
          .top-game {{
            font-weight: 600;
            font-size: 13px;
          }}
          .top-note {{
            color: var(--muted);
            font-size: 12px;
          }}
          main {{
            background: var(--card);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 16px 20px 20px;
            overflow-x: auto;
          }}
          table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
          }}
          thead th {{
            text-align: left;
            padding: 10px 8px;
            font-size: 12px;
            text-transform: uppercase;
            color: var(--muted);
            letter-spacing: 0.04em;
            position: sticky;
            top: 0;
            background: var(--card);
            border-bottom: 1px solid var(--card-border);
            z-index: 1;
          }}
          tbody td {{
            padding: 10px 8px;
            border-bottom: 1px solid rgba(223, 227, 235, 0.6);
            vertical-align: middle;
          }}
          tbody tr.row-total {{
            background: rgba(3, 131, 90, 0.04);
          }}
          tbody tr.row-total td.cell-bet {{
            padding-left: 24px;
            font-style: italic;
            color: var(--muted);
          }}
          tbody tr.row-total td.cell-market {{
            color: var(--muted);
          }}
          tbody tr.row-total td.cell-game {{
            color: var(--muted);
          }}
          .total-label {{
            font-style: italic;
            color: var(--muted);
          }}
          tbody tr:hover {{
            background: rgba(3, 131, 90, 0.08);
          }}
          .chip {{
            display: inline-block;
            background: rgba(3, 131, 90, 0.12);
            color: #026a49;
            padding: 2px 8px;
            border-radius: 999px;
            margin-right: 4px;
            font-size: 12px;
            font-weight: 600;
          }}
          .chip.muted {{
            background: rgba(104, 115, 133, 0.15);
            color: var(--muted);
          }}
          .flag {{
            display: inline-block;
            background: rgba(194, 63, 49, 0.15);
            color: #b22d21;
            padding: 2px 6px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 600;
            margin-right: 4px;
          }}
          .flag.muted {{
            background: rgba(104, 115, 133, 0.1);
            color: var(--muted);
          }}
          .cell-bet {{
            font-weight: 600;
          }}
          .cell-game {{
            font-weight: 500;
          }}
          .cell-market {{
            font-weight: 500;
            color: var(--muted);
          }}
          .cell-primary {{
            color: #026a49;
            font-weight: 600;
          }}
          .edge {{
            font-weight: 600;
          }}
          .edge.positive {{
            color: var(--edge-strong);
          }}
          .edge.negative {{
            color: var(--edge-negative);
          }}
          .edge.medium {{
            color: var(--edge-medium);
          }}
          .edge.light {{
            color: var(--edge-light);
          }}
          .edge-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 48px;
            padding: 2px 8px;
            border-radius: 999px;
            font-weight: 600;
            margin-right: 8px;
            background: rgba(3, 131, 90, 0.1);
          }}
          .edge-badge.negative {{
            background: rgba(194, 63, 49, 0.1);
          }}
          footer {{
            padding: 16px 40px 28px;
            color: var(--muted);
            font-size: 12px;
          }}
          @media (max-width: 1080px) {{
            .layout {{
              grid-template-columns: 1fr;
            }}
            .sidebar {{
              position: static;
              grid-row: 2;
            }}
            .kpi-grid {{
              grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            }}
            table {{
              font-size: 12px;
            }}
          }}
        </style>
      </head>
      <body>
        <header>
          <div class="brand">
            {logo_html}
            <div class="brand-copy">
              <h1>{title}</h1>
              <p class="brand-tagline">Betting Worldwide model vs. market snapshot.</p>
            </div>
          </div>
          <div class="edge-callouts">
            {headline_edges}
          </div>
        </header>
        <div class="layout">
          <aside class="sidebar">
            <div class="kpi-grid">
              <div class="kpi"><span class="kpi-label">Games</span><span class="kpi-value">{total_games}</span></div>
              <div class="kpi"><span class="kpi-label">Actionable ATS</span><span class="kpi-value">{len(actionable_spreads)}</span></div>
              <div class="kpi"><span class="kpi-label">Actionable Totals</span><span class="kpi-value">{len(actionable_totals)}</span></div>
              <div class="kpi"><span class="kpi-label">Avg Edge (pts)</span><span class="kpi-value">{avg_edge_display}</span></div>
              <div class="kpi"><span class="kpi-label">Kelly f=0.5 (median)</span><span class="kpi-value">{_format_kelly(np.median(kelly_values)) if kelly_values else '—'}</span></div>
              <div class="kpi"><span class="kpi-label">Providers Seen</span><span class="kpi-value">{len(providers_seen)}</span></div>
            </div>
            <div class="top-card">
              <h3>Top Spread Edges</h3>
              <ul class="top-list">
                {_top_list(top_spreads)}
              </ul>
            </div>
            <div class="top-card">
              <h3>Top Total Edges</h3>
              <ul class="top-list">
                {_top_list(top_totals)}
              </ul>
            </div>
          </aside>
          <main>
            <table>
              <thead>
                <tr>
                  <th>Bet</th>
                  <th>Game</th>
                  <th>Market</th>
                  <th>Model</th>
                  <th>Market</th>
                  <th>Weighted Spread</th>
                  <th>Weighted Total</th>
                  <th>Home ML (model)</th>
                  <th>FanDuel OP</th>
                  <th>FanDuel</th>
                  <th>Sports411 OP</th>
                  <th>Sports411</th>
                  <th>Edge (pts)</th>
                  <th>Edge (%)</th>
                  <th>Kelly (0.5)</th>
                  <th>Providers</th>
                  <th>Primary</th>
                  <th>Weather</th>
                  <th>Kickoff</th>
                  <th>Flags</th>
                </tr>
              </thead>
              <tbody>
                {''.join(table_rows)}
              </tbody>
            </table>
          </main>
        </div>
        <footer>Edge ≥ 1.0 pts (spread) or 1.5 pts (total) highlighted in the KPI strip. Kelly assumes -110 pricing and halves the optimal fraction.</footer>
      </body>
    </html>
    """
    return html


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD API key required. Set CFBD_API_KEY or pass --api-key.")

    df = sim_fbs.simulate_week(
        args.year,
        args.week,
        api_key=api_key,
        season_type=args.season_type,
        include_completed=args.include_completed,
        neutral_default=args.neutral_default,
        providers=[p.strip() for p in args.providers.split(",") if p.strip()] if args.providers else None,
    )

    if df.empty:
        print("No upcoming games found for the specified week.")
        return
    provider_filter = [p.strip() for p in args.providers.split(",") if p.strip()] if args.providers else None
    df = _ensure_oddslogic_lines(df, providers=provider_filter)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved projections to {args.output}")
    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        html = _render_html(df, title=f"FBS Week {args.week} Simulations", providers=provider_filter)
        args.html.write_text(html, encoding="utf-8")
        print(f"Saved summary HTML to {args.html}")
    if not args.output and not args.html:
        pd.set_option("display.float_format", lambda v: f"{v:0.3f}")
        print(df.sort_values("home_win_prob", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
