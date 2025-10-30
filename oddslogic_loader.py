"""Utilities for loading OddsLogic archive data into modelling workflows."""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd

SHARP_PROVIDER_NAMES = {"Circa", "Pinnacle", "Superbook", "South Point", "Bookmaker", "BetCris"}
SHARP_PROVIDER_IDS = {9, 47, 45, 13, 31, 33}


def normalize_label(label: str) -> str:
    return "".join(ch for ch in label.upper() if ch.isalnum())


def _split_top_level(text: str) -> list[str]:
    """Split OddsLogic schedule payload into top-level `{ ... }` chunks."""
    chunks: list[str] = []
    level = 0
    buffer: list[str] = []
    for char in text:
        if char == "{":
            if level > 0:
                buffer.append(char)
            level += 1
        elif char == "}":
            level -= 1
            if level > 0:
                buffer.append(char)
            else:
                chunks.append("".join(buffer))
                buffer.clear()
        else:
            if level > 0:
                buffer.append(char)
    return chunks


def _classify_header(header: str) -> str:
    upper = header.upper()
    if "COLLEGE FOOTBALL" in upper and "FCS" in upper:
        return "fcs"
    if "COLLEGE FOOTBALL" in upper and "FCS" not in upper:
        return "fbs"
    return "other"


def _parse_schedule_raw(raw: str) -> Dict[int, Dict[str, Optional[str]]]:
    """Parse the OddsLogic all.txt payload into a game-number lookup."""
    if not raw:
        return {}
    blocks = _split_top_level(raw)
    if not blocks:
        return {}
    schedule: Dict[int, Dict[str, Optional[str]]] = {}
    for block in blocks[1:]:
        header_upper = block.upper()
        if "COLLEGE FOOTBALL" in header_upper and "FCS" in header_upper:
            classification = "fcs"
        elif "COLLEGE FOOTBALL" in header_upper:
            classification = "fbs"
        else:
            classification = "other"
        pieces = _split_top_level(block)
        if not pieces:
            continue
        header_fields = pieces[0].split(",")
        header_text = ",".join(header_fields[5:]).strip('"') if len(header_fields) >= 6 else ""
        for piece in pieces[1:]:
            if not piece:
                continue
            team_start = piece.rfind("{")
            team_part = piece[team_start + 1 : -1] if team_start != -1 else ""
            field_part = (
                piece[: team_start - 1]
                if team_start > 0 and piece[team_start - 1] == ","
                else piece[:team_start]
            )
            fields = [f.strip('"') for f in field_part.split(",") if f is not None]
            team_fields = [t.strip('"') for t in team_part.split(",")]
            if not fields:
                continue
            try:
                game_number = int(fields[0])
            except ValueError:
                continue
            start_date = fields[4] if len(fields) > 4 else None
            start_time_local = fields[7] if len(fields) > 7 else None
            start_datetime = fields[9] if len(fields) > 9 else None
            home_team = team_fields[0] if len(team_fields) >= 1 else None
            home_abbr = team_fields[1] if len(team_fields) >= 2 else None
            away_team = team_fields[5] if len(team_fields) >= 6 else None
            away_abbr = team_fields[6] if len(team_fields) >= 7 else None
            schedule[game_number] = {
                "classification": classification,
                "header": header_text or "",
                "start_date": start_date,
                "start_time_local": start_time_local,
                "start_datetime": start_datetime,
                "home_team": home_team,
                "home_abbr": home_abbr,
                "away_team": away_team,
                "away_abbr": away_abbr,
                "home_key": normalize_label(home_team or ""),
                "away_key": normalize_label(away_team or ""),
            }
    return schedule


def _load_schedule_map(raw_dir: Path) -> Dict[str, Dict[int, Dict[str, Optional[str]]]]:
    """Load per-date schedule maps from the oddslogic_archive/raw directory."""
    schedule_map: Dict[str, Dict[int, Dict[str, Optional[str]]]] = {}
    pattern = re.compile(r"schedule_(\d{4}-\d{2}-\d{2})\.txt$")
    for schedule_file in raw_dir.glob("schedule_*.txt"):
        match = pattern.match(schedule_file.name)
        if not match:
            continue
        date_str = match.group(1)
        raw = schedule_file.read_text(encoding="utf-8").strip()
        schedule = _parse_schedule_raw(raw)
        if schedule:
            schedule_map[date_str] = schedule
    return schedule_map


def _augment_with_schedule(
    df: pd.DataFrame, schedule_map: Dict[str, Dict[int, Dict[str, Optional[str]]]]
) -> pd.DataFrame:
    if not schedule_map:
        return df

    lookup = {
        (date_str, game_number): entry for date_str, mapping in schedule_map.items() for game_number, entry in mapping.items()
    }
    if not lookup:
        return df

    df = df.copy()
    df["_game_number"] = pd.to_numeric(df["game_number"], errors="coerce").astype("Int64")

    def _needs_fill(row: pd.Series) -> bool:
        if pd.isna(row["_game_number"]):
            return False
        has_home = bool(str(row.get("home_team") or "").strip())
        has_away = bool(str(row.get("away_team") or "").strip())
        start_dt = row.get("start_datetime")
        has_datetime = False
        if isinstance(start_dt, str):
            has_datetime = bool(start_dt.strip())
        elif start_dt is not None and not pd.isna(start_dt):
            has_datetime = True
        return not (has_home and has_away and has_datetime)

    def _apply_schedule(row: pd.Series) -> pd.Series:
        game_number = row["_game_number"]
        if pd.isna(game_number):
            return row
        date_key = str(row.get("date") or "")
        entry = lookup.get((date_key, int(game_number)))
        if not entry:
            return row
        if entry.get("classification"):
            row["classification"] = entry["classification"]
        if entry.get("header"):
            row["category_header"] = entry["header"]
        for field in ("home_team", "home_abbr", "away_team", "away_abbr", "start_date", "start_time_local"):
            if not str(row.get(field) or "").strip() and entry.get(field):
                row[field] = entry[field]
        if pd.isna(row.get("start_datetime")) or not str(row.get("start_datetime") or "").strip():
            if entry.get("start_datetime"):
                row["start_datetime"] = entry["start_datetime"]
        if not str(row.get("home_key") or "").strip() and entry.get("home_key"):
            row["home_key"] = entry["home_key"]
        if not str(row.get("away_key") or "").strip() and entry.get("away_key"):
            row["away_key"] = entry["away_key"]
        return row

    mask = df.apply(_needs_fill, axis=1)
    if mask.any():
        df.loc[mask] = df.loc[mask].apply(_apply_schedule, axis=1)

    df = df.drop(columns="_game_number")
    return df


def load_archive_dataframe(path: Path | str) -> pd.DataFrame:
    """Load all per-date CSV exports produced by oddslogic_scraper."""
    base = Path(path)
    csv_dir = base / "csv" if (base / "csv").is_dir() else base
    csv_files = sorted(csv_dir.glob("oddslogic_lines_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No archive CSVs found in {csv_dir}")
    frames = [pd.read_csv(csv_file) for csv_file in csv_files]
    df = pd.concat(frames, ignore_index=True)

    raw_dir = base / "raw"
    if raw_dir.is_dir():
        schedule_map = _load_schedule_map(raw_dir)
        if schedule_map:
            df = _augment_with_schedule(df, schedule_map)

    # Best effort conversions.
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(int)
    df["line_value"] = pd.to_numeric(df["line_value"], errors="coerce")
    df["line_price"] = pd.to_numeric(df["line_price"], errors="coerce")
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce")
    start_date = pd.to_datetime(df["start_date"], errors="coerce")
    df["kickoff_date"] = df["start_datetime"].dt.date
    df.loc[df["kickoff_date"].isna(), "kickoff_date"] = start_date.dt.date
    df["home_key"] = df["home_key"].fillna("").apply(normalize_label)
    df["away_key"] = df["away_key"].fillna("").apply(normalize_label)
    df["sportsbook_name"] = df["sportsbook_name"].fillna("")
    df["classification"] = df["classification"].fillna("unknown")
    df["row_type"] = df["row_type"].fillna("other")
    return df


def _normalize_classifications(classification: Optional[Sequence[str] | str]) -> Optional[set[str]]:
    if classification is None:
        return None
    if isinstance(classification, str):
        if not classification:
            return None
        return {classification.lower()}
    normalized = {cls.lower() for cls in classification if cls}
    return normalized or None


def _closing_by_row(
    df: pd.DataFrame,
    *,
    classification: Optional[Sequence[str] | str],
    row_type: str,
    providers: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    class_filter = _normalize_classifications(classification)
    mask = pd.Series(True, index=df.index)
    if class_filter is not None:
        mask &= df["classification"].str.lower().isin(class_filter)
    mask &= df["row_type"] == row_type
    if providers:
        provider_set = {p.casefold() for p in providers}
        mask &= df["sportsbook_name"].str.casefold().isin(provider_set)

    subset = (
        df.loc[mask]
        .dropna(subset=["kickoff_date", "home_key", "away_key", "line_value"])
        .sort_values(["kickoff_date", "game_number", "sportsbook_id", "timestamp"])
    )
    if subset.empty:
        return subset

    grouping = [
        "kickoff_date",
        "game_number",
        "home_key",
        "away_key",
        "sportsbook_id",
        "sportsbook_name",
    ]
    subset = subset.groupby(grouping, as_index=False).tail(1)
    return subset


def build_closing_lookup(
    df: pd.DataFrame,
    classification: Optional[Sequence[str] | str] = None,
    providers: Optional[Sequence[str]] = None,
) -> Dict[Tuple[dt.date, str, str], Dict[str, object]]:

    spreads = _closing_by_row(df, classification=classification, row_type="spread", providers=providers)
    totals = _closing_by_row(df, classification=classification, row_type="total", providers=providers)

    total_map = {
        (
            row.kickoff_date,
            row.home_key,
            row.away_key,
            row.sportsbook_id,
        ): row
        for row in totals.itertuples()
    }

    def _provider_rank(name: str, sportsbook_id: Optional[int]) -> int:
        if sportsbook_id in SHARP_PROVIDER_IDS:
            return 2
        if name in SHARP_PROVIDER_NAMES:
            return 2
        if name:
            return 1
        return 0

    closing_lookup: Dict[Tuple[dt.date, str, str], Dict[str, object]] = {}
    for row in spreads.itertuples():
        key = (row.kickoff_date, row.home_key, row.away_key, row.sportsbook_id)
        total_row = total_map.get(key)
        lookup_key = (row.kickoff_date, row.home_key, row.away_key)
        entry = closing_lookup.setdefault(
            lookup_key,
            {
                "providers": {},
                "_best_rank": -1,
                "_best_key": "",
            },
        )
        provider_name = row.sportsbook_name or ""
        provider_key = provider_name or f"id_{row.sportsbook_id}"
        provider_rank = _provider_rank(provider_name, row.sportsbook_id)
        spread_value = (
            float(row.line_value)
            if row.line_value is not None and not pd.isna(row.line_value)
            else None
        )
        spread_price = (
            float(row.line_price)
            if row.line_price is not None and not pd.isna(row.line_price)
            else None
        )
        provider_payload = {
            "spread_value": spread_value,
            "spread_price": spread_price,
            "sportsbook_id": int(row.sportsbook_id) if row.sportsbook_id is not None else None,
            "sportsbook_name": provider_name,
            "timestamp": int(row.timestamp) if hasattr(row, "timestamp") else None,
            "home_team": getattr(row, "home_team", ""),
            "away_team": getattr(row, "away_team", ""),
            "game_number": row.game_number,
        }
        if total_row is not None:
            total_value = (
                float(total_row.line_value)
                if total_row.line_value is not None and not pd.isna(total_row.line_value)
                else None
            )
            total_price = (
                float(total_row.line_price)
                if total_row.line_price is not None and not pd.isna(total_row.line_price)
                else None
            )
            provider_payload.update(
                {
                    "total_value": total_value,
                    "total_price": total_price,
                    "total_direction": getattr(total_row, "total_direction", None),
                }
            )
        entry["providers"][provider_key] = provider_payload

        best_rank = entry["_best_rank"]
        best_key = entry["_best_key"]
        should_update = False
        if provider_rank > best_rank:
            should_update = True
        elif provider_rank == best_rank:
            if provider_key < best_key or not best_key:
                should_update = True

        if should_update:
            entry["_best_rank"] = provider_rank
            entry["_best_key"] = provider_key
            entry.update(
                {
                    "spread_value": provider_payload.get("spread_value"),
                    "spread_price": provider_payload.get("spread_price"),
                    "sportsbook_id": provider_payload.get("sportsbook_id"),
                    "sportsbook_name": provider_payload.get("sportsbook_name"),
                    "home_team": provider_payload.get("home_team"),
                    "away_team": provider_payload.get("away_team"),
                    "game_number": provider_payload.get("game_number"),
                }
            )
            if total_row is not None:
                entry.update(
                    {
                        "total_value": provider_payload.get("total_value"),
                        "total_price": provider_payload.get("total_price"),
                        "total_direction": provider_payload.get("total_direction"),
                    }
                )

    for entry in closing_lookup.values():
        entry.pop("_best_rank", None)
        entry.pop("_best_key", None)

    return closing_lookup


def summarize_coverage(
    df: pd.DataFrame,
    classification: Optional[Sequence[str] | str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aggregate coverage stats for OddsLogic spread closings.

    Parameters
    ----------
    df : pd.DataFrame
        Flattened archive dataframe from :func:`load_archive_dataframe`.
    classification : Optional[Sequence[str] | str]
        Optional subset of classifications to include (e.g., ``"fbs"``).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of ``(coverage_by_classification, provider_coverage)``.
    """

    class_filter = _normalize_classifications(classification)
    subset = df.copy()
    subset = subset[subset["row_type"] == "spread"]
    if class_filter is not None:
        subset = subset[subset["classification"].str.lower().isin(class_filter)]

    subset = subset.dropna(subset=["kickoff_date", "home_key", "away_key"])
    subset = subset.assign(classification=subset["classification"].fillna("unknown"))
    if subset.empty:
        return (
            pd.DataFrame(columns=["classification", "games", "avg_providers", "pct_single_provider"]),
            pd.DataFrame(columns=["classification", "sportsbook_name", "games"]),
        )

    game_group = (
        subset.groupby(["classification", "kickoff_date", "home_key", "away_key"])
        .agg(provider_count=("sportsbook_id", "nunique"))
        .reset_index()
    )

    coverage = (
        game_group.groupby("classification")
        .agg(
            games=("provider_count", "size"),
            avg_providers=("provider_count", "mean"),
            pct_single_provider=("provider_count", lambda s: float((s == 1).sum()) / len(s)),
        )
        .reset_index()
    )

    provider_games = (
        subset.drop_duplicates(["classification", "kickoff_date", "home_key", "away_key", "sportsbook_name"])
        .groupby(["classification", "sportsbook_name"])
        .size()
        .reset_index(name="games")
        .sort_values(["classification", "games"], ascending=[True, False])
    )

    return coverage, provider_games
