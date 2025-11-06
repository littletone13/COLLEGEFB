#!/usr/bin/env python3
"""
Normalize wager history exports (e.g., betonline.csv) and append them to a master ledger.

This script parses the export, computes derived fields (stake, net, ROI, record flags),
and writes/updates tracking/wagers_master.csv along with tracking/wagers_summary.csv.

Usage:
    python scripts/update_wager_master.py \
        --input ~/Desktop/betonline.csv \
        [--output tracking/wagers_master.csv]
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_OUTPUT = Path("tracking") / "wagers_master.csv"
SUMMARY_OUTPUT = Path("tracking") / "wagers_summary.csv"
DEFAULT_CLOSING_PROVIDERS = ["BetOnline", "Circa", "Sports411", "FanDuel", "Pinnacle"]
MAX_CLOSING_DAY_DELTA = 14  # days window for matching closings to wagers

MONEY_PATTERN = re.compile(r"[$,]")
PRICE_PATTERN = re.compile(r"([+-]\d{3})")
HALF_MAP = {
    "½": ".5",
    "¼": ".25",
    "¾": ".75",
}


def normalize_provider_key(value: str | None) -> str:
    if value is None:
        return ""
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


PROVIDER_CANONICAL = {
    normalize_provider_key("BetOnline"): "BetOnline",
    normalize_provider_key("BetOnline.ag"): "BetOnline",
    normalize_provider_key("Circa"): "Circa",
    normalize_provider_key("Sports411"): "Sports411",
    normalize_provider_key("FanDuel"): "FanDuel",
    normalize_provider_key("Fanduel"): "FanDuel",
    normalize_provider_key("Pinnacle"): "Pinnacle",
    normalize_provider_key("Draft Kings"): "DraftKings",
    normalize_provider_key("DraftKings"): "DraftKings",
    normalize_provider_key("Sports 411"): "Sports411",
    normalize_provider_key("Bookmaker"): "Bookmaker",
}


SHARP_CONSENSUS_PROVIDERS = {
    normalize_provider_key("Circa"),
    normalize_provider_key("Sports411"),
    normalize_provider_key("Pinnacle"),
    normalize_provider_key("Sharp"),
    normalize_provider_key("LowVig"),
    normalize_provider_key("BetOnline"),
}

STATUS_PRIORITY = {
    "won": 3,
    "win": 3,
    "lost": 3,
    "loss": 3,
    "push": 2,
    "draw": 2,
    "void": 1,
    "cancelled": 1,
    "canceled": 1,
    "pending": 0,
}


def canonical_provider_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    key = normalize_provider_key(raw)
    if not key:
        return None
    if key in PROVIDER_CANONICAL:
        return PROVIDER_CANONICAL[key]
    for candidate_key, canon in PROVIDER_CANONICAL.items():
        if key.startswith(candidate_key) or candidate_key.startswith(key):
            return canon
    return raw.strip()


def normalize_money(value: str | float | int | None) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value = MONEY_PATTERN.sub("", str(value))
    try:
        return float(value)
    except ValueError:
        return None


def normalize_fraction(text: str) -> str:
    for orig, repl in HALF_MAP.items():
        text = text.replace(orig, repl)
    return text


def american_to_prob(price: float | int | None) -> float | None:
    if price is None or price == 0 or not np.isfinite(price):
        return None
    price = float(price)
    if price > 0:
        return 100.0 / (price + 100.0)
    return -price / (-price + 100.0)


def american_to_moneyline(prob: float | None) -> float | None:
    if prob is None or not np.isfinite(prob):
        return None
    if prob <= 0.0 or prob >= 1.0:
        return None
    if prob >= 0.5:
        return -100.0 * prob / (1.0 - prob)
    return 100.0 * (1.0 - prob) / prob


def _default_archive_path() -> Path | None:
    env_path = os.environ.get("ODDSLOGIC_ARCHIVE_DIR")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate
    fallback = Path("../data/oddslogic/oddslogic_ncaa_all")
    fallback = fallback.resolve()
    return fallback if fallback.exists() else None


def _candidate_game_numbers(rotation_raw: str | int | float | None) -> list[int]:
    if rotation_raw is None or (isinstance(rotation_raw, float) and np.isnan(rotation_raw)):
        return []
    try:
        rotation = int(float(str(rotation_raw).strip()))
    except (TypeError, ValueError):
        return []
    candidates = {rotation}
    if rotation % 2 == 0:
        candidates.add(rotation - 1)
    else:
        candidates.add(rotation + 1)
    candidates.add(rotation - 1)
    candidates.add(rotation + 1)
    return [cand for cand in candidates if cand > 0]


def _collect_game_numbers(df: pd.DataFrame) -> list[int]:
    if "rotation" not in df.columns:
        return []
    numbers: set[int] = set()
    for value in df["rotation"].dropna():
        for cand in _candidate_game_numbers(value):
            numbers.add(cand)
    return sorted(numbers)


def _load_oddslogic_subset(
    archive_dir: Path,
    game_numbers: Sequence[int],
    *,
    providers: Sequence[str] | None = None,
) -> pd.DataFrame:
    if not game_numbers:
        return pd.DataFrame()
    archive_dir = archive_dir.expanduser()
    csv_dir = archive_dir / "csv" if (archive_dir / "csv").is_dir() else archive_dir
    if not csv_dir.exists():
        raise FileNotFoundError(f"OddsLogic archive CSV directory not found: {csv_dir}")

    provider_keys = (
        {normalize_provider_key(p) for p in providers if p}
        if providers
        else None
    )

    usecols = [
        "date",
        "classification",
        "game_number",
        "timestamp",
        "sportsbook_name",
        "row_type",
        "line_value",
        "line_price",
    ]
    frames: list[pd.DataFrame] = []
    game_set = set(game_numbers)
    for csv_path in sorted(csv_dir.glob("oddslogic_lines_*.csv")):
        df = pd.read_csv(csv_path, usecols=usecols)
        df["game_number"] = pd.to_numeric(df["game_number"], errors="coerce")
        mask = df["game_number"].isin(game_set)
        if not mask.any():
            continue
        subset = df.loc[mask].copy()
        if provider_keys is not None:
            subset["_provider_key"] = subset["sportsbook_name"].apply(normalize_provider_key)
            subset = subset[subset["_provider_key"].isin(provider_keys)]
            if subset.empty:
                continue
        frames.append(subset)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["_provider_key"] = df.get("_provider_key", df["sportsbook_name"].apply(normalize_provider_key))
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(int)
    df["line_value"] = pd.to_numeric(df["line_value"], errors="coerce")
    df["line_price"] = pd.to_numeric(df["line_price"], errors="coerce")
    df["game_number"] = pd.to_numeric(df["game_number"], errors="coerce").astype("Int64")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("timestamp")
    group_cols = ["date", "game_number", "sportsbook_name", "_provider_key", "row_type", "classification"]
    df = df.groupby(group_cols, as_index=False).tail(1)
    return df


def _build_closing_map(df: pd.DataFrame) -> Dict[int, Dict[str, Dict[datetime.date, Dict[str, object]]]]:
    if df.empty:
        return {}
    df = df.sort_values("timestamp")
    closing_map: Dict[int, Dict[str, Dict[datetime.date, Dict[str, object]]]] = {}
    for _, row in df.iterrows():
        game_number = row.get("game_number")
        if game_number is None or pd.isna(game_number):
            continue
        game_number = int(game_number)
        provider_key = row.get("_provider_key", "")
        if not provider_key:
            continue
        entry = closing_map.setdefault(game_number, {})
        provider_bucket = entry.setdefault(provider_key, {})
        date_val = row.get("date")
        if isinstance(date_val, pd.Timestamp):
            date_key = date_val.date()
        else:
            try:
                date_key = pd.to_datetime(date_val).date()  # type: ignore[arg-type]
            except Exception:
                continue
        record = provider_bucket.get(date_key, {})
        record.setdefault(
            "sportsbook_name",
            canonical_provider_name(row.get("sportsbook_name", "")) or row.get("sportsbook_name", ""),
        )
        record.setdefault("classification", row.get("classification", None))
        record["date"] = date_key
        timestamp_val = row.get("timestamp", None)
        ts_int = int(timestamp_val) if timestamp_val is not None and not pd.isna(timestamp_val) else None
        # Keep the latest timestamp per market.
        if ts_int is not None:
            record["timestamp"] = max(ts_int, record.get("timestamp", 0) or 0)
        line_value = row.get("line_value", None)
        line_price = row.get("line_price", None)
        row_type = row.get("row_type", "")
        if row_type == "spread":
            record["spread_line"] = float(line_value) if line_value is not None and not pd.isna(line_value) else None
            record["spread_price"] = float(line_price) if line_price is not None and not pd.isna(line_price) else None
        elif row_type == "total":
            record["total_line"] = float(line_value) if line_value is not None and not pd.isna(line_value) else None
            record["total_price"] = float(line_price) if line_price is not None and not pd.isna(line_price) else None
        provider_bucket[date_key] = record
    return closing_map


def _select_closing_record(
    records: Dict[date, Dict[str, object]],
    wager_date: date | None,
) -> Dict[str, object] | None:
    if not records:
        return None
    if wager_date is None:
        latest_date = max(records.keys())
        return records.get(latest_date)
    future_candidates = []
    for rec_date, rec in records.items():
        delta_days = (rec_date - wager_date).days
        if delta_days < 0:
            continue
        future_candidates.append((delta_days, rec_date, rec))
    if future_candidates:
        delta_days, rec_date, rec = min(future_candidates, key=lambda item: (item[0], item[1]))
        if delta_days <= MAX_CLOSING_DAY_DELTA:
            return rec
    return None


def _compute_clv_line(
    *,
    market_type: str,
    selection: str | None,
    bet_line: float | None,
    closing_line: float | None,
) -> float | None:
    if bet_line is None or closing_line is None:
        return np.nan
    if not np.isfinite(bet_line) or not np.isfinite(closing_line):
        return np.nan
    market_type = (market_type or "").lower()
    if market_type == "total":
        sel = (selection or "").strip().lower()
        if sel.startswith("over"):
            return float(closing_line) - float(bet_line)
        if sel.startswith("under"):
            return float(bet_line) - float(closing_line)
        # Fallback – assume same orientation as spreads.
    return float(bet_line) - float(closing_line)


def _apply_closing_data(
    master: pd.DataFrame,
    closing_map: Dict[int, Dict[str, Dict[str, object]]],
    *,
    provider_priority: Sequence[str],
) -> pd.DataFrame:
    if not closing_map:
        return master

    priority_keys = [normalize_provider_key(p) for p in provider_priority if p]
    updated = master.copy()

    if "closing_provider" not in updated.columns:
        updated["closing_provider"] = np.nan
    if "closing_timestamp" not in updated.columns:
        updated["closing_timestamp"] = np.nan
    if "closing_line" not in updated.columns:
        updated["closing_line"] = np.nan
    if "closing_price" not in updated.columns:
        updated["closing_price"] = np.nan
    if "clv_line" not in updated.columns:
        updated["clv_line"] = np.nan
    if "clv_price" not in updated.columns:
        updated["clv_price"] = np.nan
    # Reset prior values to avoid carrying stale closings when data is unavailable this run.
    updated["closing_line"] = np.nan
    updated["closing_price"] = np.nan
    updated["clv_line"] = np.nan
    updated["clv_price"] = np.nan
    updated["closing_provider"] = updated["closing_provider"].astype(object)
    updated["closing_provider"] = None
    updated["closing_timestamp"] = np.nan

    for idx, row in updated.iterrows():
        rotation_raw = row.get("rotation")
        candidates = _candidate_game_numbers(rotation_raw)
        resolved_game = None
        for cand in candidates:
            if cand in closing_map:
                resolved_game = cand
                break
        if resolved_game is None:
            continue
        provider_map = closing_map[resolved_game]
        source_provider = canonical_provider_name(row.get("source")) or row.get("source")
        sequence = []
        if source_provider:
            sequence.append(normalize_provider_key(source_provider))
        for key in priority_keys:
            if key not in sequence:
                sequence.append(key)
        if not sequence:
            sequence = list(provider_map.keys())

        wager_date_raw = row.get("wager_date")
        wager_date_val: date | None
        if isinstance(wager_date_raw, pd.Timestamp):
            wager_date_val = wager_date_raw.date()
        else:
            try:
                wager_date_val = pd.to_datetime(wager_date_raw).date()  # type: ignore[arg-type]
            except Exception:
                wager_date_val = None

        chosen_payload = None
        chosen_key = ""
        consensus_payloads = []
        for prov_key, records in provider_map.items():
            payload = _select_closing_record(records, wager_date_val)
            if payload is None:
                continue
            name_norm = normalize_provider_key(payload.get("sportsbook_name"))
            if name_norm in SHARP_CONSENSUS_PROVIDERS:
                consensus_payloads.append(payload)

        for provider_key in sequence:
            if not provider_key:
                continue
            records = provider_map.get(provider_key)
            if records is None:
                # try matching canonically if stored differently
                for stored_key, stored_records in provider_map.items():
                    any_record = next(iter(stored_records.values()), None)
                    if any_record and provider_key == normalize_provider_key(any_record.get("sportsbook_name")):
                        records = stored_records
                        provider_key = stored_key
                        break
            if not records:
                continue
            payload = _select_closing_record(records, wager_date_val)
            if payload is None:
                continue
            market_type = (row.get("market_type") or "").lower()
            if market_type == "spread":
                closing_line = payload.get("spread_line")
                closing_price = payload.get("spread_price")
            elif market_type == "total":
                closing_line = payload.get("total_line")
                closing_price = payload.get("total_price")
            else:
                closing_line = None
                closing_price = None
            if closing_line is None and closing_price is None:
                continue
            chosen_payload = payload
            chosen_key = provider_key
            break

        if not chosen_payload:
            continue

        market_type = (row.get("market_type") or "").lower()
        def _consensus_values(payloads: list[Dict[str, object]], line_key: str, price_key: str) -> tuple[float | None, float | None, int | None]:
            if not payloads:
                return (None, None, None)
            line_values = [p.get(line_key) for p in payloads if p.get(line_key) is not None and not pd.isna(p.get(line_key))]
            price_values = [p.get(price_key) for p in payloads if p.get(price_key) is not None and not pd.isna(p.get(price_key))]
            timestamps = [int(p.get("timestamp", 0)) for p in payloads if p.get("timestamp") is not None]
            line_out = float(np.mean([float(v) for v in line_values])) if line_values else None
            price_out = float(np.mean([float(v) for v in price_values])) if price_values else None
            ts_out = max(timestamps) if timestamps else None
            return (line_out, price_out, ts_out)

        chosen_line = None
        chosen_price_val = None
        if market_type == "spread":
            chosen_line = chosen_payload.get("spread_line")
            chosen_price_val = chosen_payload.get("spread_price")
            consensus_line, consensus_price, consensus_ts = _consensus_values(consensus_payloads, "spread_line", "spread_price")
        elif market_type == "total":
            chosen_line = chosen_payload.get("total_line")
            chosen_price_val = chosen_payload.get("total_price")
            consensus_line, consensus_price, consensus_ts = _consensus_values(consensus_payloads, "total_line", "total_price")
        else:
            consensus_line = consensus_price = consensus_ts = None

        final_line = consensus_line if consensus_line is not None else chosen_line
        final_price = consensus_price if consensus_price is not None else chosen_price_val
        closing_label = "SharpConsensus" if consensus_line is not None else chosen_payload.get("sportsbook_name")
        closing_ts = consensus_ts if consensus_ts is not None else chosen_payload.get("timestamp")

        clv_line = _compute_clv_line(
            market_type=market_type,
            selection=row.get("selection"),
            bet_line=float(row["bet_line"]) if pd.notna(row.get("bet_line")) else None,
            closing_line=final_line,
        )
        bet_price = row.get("bet_price")
        closing_price = final_price
        if bet_price is None or pd.isna(bet_price) or closing_price is None or pd.isna(closing_price):
            clv_price = np.nan
        else:
            clv_price = float(bet_price) - float(closing_price)

        closing_line = final_line
        closing_price_val = final_price

        if closing_line is not None:
            updated.at[idx, "closing_line"] = closing_line
        if closing_price_val is not None:
            updated.at[idx, "closing_price"] = closing_price_val
        updated.at[idx, "clv_line"] = clv_line
        updated.at[idx, "clv_price"] = clv_price
        updated.at[idx, "closing_provider"] = closing_label
        updated.at[idx, "closing_timestamp"] = closing_ts
    return updated


def parse_description(base: str) -> tuple[str | None, str | None]:
    """
    Extract rotation (if present) and a team/matchup label from the description string.
    Example: "FOOTBALL - 309027 Brown +12½ -115"
    """
    if not isinstance(base, str):
        return None, None
    if " - " not in base:
        return None, base.strip()
    _, tail = base.split(" - ", 1)
    parts = tail.strip().split()
    if not parts:
        return None, None
    rotation = parts[0] if parts[0].isdigit() else None
    remainder = tail[len(rotation) :].strip() if rotation else tail
    remainder = remainder.strip()
    # Remove odds/lines at the end for cleaner matchup text
    remainder = normalize_fraction(remainder)
    remainder = re.sub(r"[+-]?\d+(?:\.\d+)?", "", remainder)
    remainder = PRICE_PATTERN.sub("", remainder)
    remainder = remainder.replace("forgame", "").replace("for", " ").strip()
    return rotation, re.sub(r"\s{2,}", " ", remainder).strip()


def parse_market(row: pd.Series) -> dict[str, object]:
    desc_short = str(row.get("description_short") or "")
    desc_short = normalize_fraction(desc_short.lower())
    base_desc = normalize_fraction(str(row.get("description") or ""))
    rotation, matchup = parse_description(base_desc)

    price_match = PRICE_PATTERN.search(desc_short) or PRICE_PATTERN.search(base_desc.lower())
    price = int(price_match.group(1)) if price_match else None

    bet_type = str(row.get("bet_type") or "").strip()
    market = bet_type.lower()
    selection = None
    line = None

    if "total" in market or "total" in desc_short:
        market = "total"
        m = re.search(r"(over|under)[^\d]*([0-9]+(?:\.\d+)?)", desc_short)
        if not m:
            m = re.search(r"(over|under)[^\d]*([0-9]+(?:\.\d+)?)", base_desc.lower())
        if m:
            selection = m.group(1).title()
            line = float(m.group(2))
    elif "money" in market or price_match and not re.search(r"[+-]\d+(?:\.\d+)?", desc_short):
        market = "moneyline"
        selection = matchup
    else:
        market = "spread"
        m = re.search(r"([+-]\d+(?:\.\d+)?)", desc_short)
        if not m:
            m = re.search(r"([+-]\d+(?:\.\d+)?)", base_desc.lower())
        if m:
            line = float(m.group(1))
        selection = matchup

    return {
        "rotation": rotation,
        "matchup": matchup,
        "selection": selection,
        "market_type": market,
        "bet_line": line,
        "bet_price": price,
        "bet_probability": american_to_prob(price),
    }


def compute_result_fields(source_row: pd.Series) -> dict[str, object]:
    status_candidates = [
        source_row.get("status_badge"),
        source_row.get("status"),
        source_row.get("status_badge_secondary"),
        source_row.get("result_context"),
    ]
    status = ""
    best_priority = -1
    for candidate in status_candidates:
        if candidate is None or (isinstance(candidate, float) and pd.isna(candidate)):
            continue
        candidate_text = str(candidate).strip()
        if not candidate_text:
            continue
        normalized = candidate_text.lower()
        priority = STATUS_PRIORITY.get(normalized)
        if priority is None:
            normalized_head = normalized.split()[0]
            priority = STATUS_PRIORITY.get(normalized_head, 0)
        if priority > best_priority:
            best_priority = priority
            status = normalized
    if not status:
        status = "pending"
    stake = normalize_money(source_row.get("stake"))
    to_win = normalize_money(source_row.get("to_win"))
    to_return = normalize_money(source_row.get("to_return"))

    if status in {"won", "win"}:
        net = (to_return if to_return is not None else (stake or 0) + (to_win or 0)) - (stake or 0)
        roi = net / stake if stake else np.nan
        rec = (1, 0, 0)
    elif status in {"lost", "loss"}:
        net = -(stake or 0)
        roi = net / stake if stake else np.nan
        rec = (0, 1, 0)
    elif status in {"push", "draw"}:
        net = 0.0
        roi = 0.0
        rec = (0, 0, 1)
    else:  # pending / cancelled / unknown
        net = np.nan
        roi = np.nan
        rec = (0, 0, 0)

    return {
        "status_normalized": status,
        "stake": stake,
        "to_win": to_win,
        "to_return": to_return,
        "net_profit": net,
        "roi": roi,
        "record_win": rec[0],
        "record_loss": rec[1],
        "record_push": rec[2],
    }


def normalize_wager_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    rename_map = {
        "bet-history__table__body__rows__columns--id": "wager_id",
        "bet-history__table__body__rows__columns--date": "wager_date",
        "bet-history__table__body__rows__columns--description": "description",
        "bet-history__row-description-text": "description_short",
        "bet-history__table__body__rows__columns--type": "bet_type",
        "bet-history__table__body__rows__columns--status": "status",
        "bet-history__table__body__rows__columns--amount--mobile": "stake",
        "bet-history__table__body__rows__columns--towin": "to_win",
        "bet-history__table__body__rows__columns--toreturn": "to_return",
        "bet-history__details-item 4": "placed_at",
        "bet-history__badge--pending": "status_badge",
        "bet-history__details-item 13": "details_extra",
        "bet-history__badge--won": "status_badge_secondary",
        "cell-row 7": "device_context",
        "cell-row 16": "result_context",
    }
    df = df.rename(columns=rename_map)

    for legacy_col in rename_map.keys():
        if legacy_col in df.columns:
            df = df.drop(columns=legacy_col)

    parsed_rows = []
    for _, row in df.iterrows():
        base = row.to_dict()
        base["stake"] = normalize_money(base.get("stake"))
        base["to_win"] = normalize_money(base.get("to_win"))
        base["to_return"] = normalize_money(base.get("to_return"))
        market_fields = parse_market(row)
        result_fields = compute_result_fields(base)
        base.update(market_fields)
        base.update(result_fields)
        base["source"] = source
        base["ingested_at"] = datetime.now(tz=timezone.utc).isoformat()
        base.setdefault("closing_line", np.nan)
        base.setdefault("closing_price", np.nan)
        base.setdefault("closing_provider", "")
        base.setdefault("closing_timestamp", np.nan)
        base["clv_line"] = (
            base["bet_line"] - base["closing_line"]
            if base.get("bet_line") is not None and pd.notna(base.get("closing_line"))
            else np.nan
        )
        base["clv_price"] = (
            base["bet_price"] - base["closing_price"]
            if base.get("bet_price") is not None and pd.notna(base.get("closing_price"))
            else np.nan
        )
        parsed_rows.append(base)

    normalized = pd.DataFrame(parsed_rows)
    if "wager_date" in normalized.columns:
        normalized["wager_date"] = pd.to_datetime(
            normalized["wager_date"], errors="coerce"
        )
    return normalized


def load_inputs(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            print(f"[warn] Input not found: {path}")
            continue
        df = pd.read_csv(path)
        frames.append(normalize_wager_frame(df, source=path.stem))
    if not frames:
        raise FileNotFoundError("No valid input files were provided.")
    return pd.concat(frames, ignore_index=True)


def update_master(input_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing = pd.read_csv(output_path, parse_dates=["wager_date"], keep_default_na=False)
        combined = pd.concat([existing, input_df], ignore_index=True)
        combined["status_normalized"] = combined["status_normalized"].fillna(""
            ).astype(str).str.strip().str.lower()
        combined["status_priority"] = combined["status_normalized"].map(STATUS_PRIORITY).fillna(0)
        combined["ingested_at"] = pd.to_datetime(combined["ingested_at"], errors="coerce")
        combined.sort_values(["status_priority", "ingested_at"], ascending=[False, True], inplace=True)
        combined = combined.drop_duplicates(subset=["wager_id", "description", "source"], keep="first")
        combined.drop(columns=["status_priority"], inplace=True)
        combined.sort_values("ingested_at", inplace=True)
    else:
        combined = input_df

    if "rotation" in combined.columns:
        combined["rotation"] = pd.to_numeric(combined["rotation"], errors="coerce")
    numeric_cols = [
        "bet_line",
        "bet_price",
        "bet_probability",
        "closing_line",
        "closing_price",
        "clv_line",
        "clv_price",
    ]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    legacy_cols = [col for col in ("cell-row 7", "cell-row 16") if col in combined.columns]
    if legacy_cols:
        combined = combined.drop(columns=legacy_cols)
    combined.to_csv(output_path, index=False)
    print(f"[info] Updated master ledger: {output_path} ({len(combined)} wagers)")
    return combined


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["net_profit"] = pd.to_numeric(df["net_profit"], errors="coerce")
    df["stake"] = pd.to_numeric(df["stake"], errors="coerce")
    if "to_win" in df.columns:
        df["to_win"] = pd.to_numeric(df["to_win"], errors="coerce")
    if "to_return" in df.columns:
        df["to_return"] = pd.to_numeric(df["to_return"], errors="coerce")
    if "clv_line" in df.columns:
        df["clv_line"] = pd.to_numeric(df["clv_line"], errors="coerce")
    if "clv_price" in df.columns:
        df["clv_price"] = pd.to_numeric(df["clv_price"], errors="coerce")

    summary_rows = []
    for label, group in [("overall", df)] + [(bt, df[df["market_type"] == bt]) for bt in sorted(df["market_type"].dropna().unique())]:
        completed = group[group["status_normalized"].isin({"won", "win", "lost", "loss", "push", "draw"})]
        pending = group[group["status_normalized"].isin({"pending"})]
        wins = completed["record_win"].sum()
        losses = completed["record_loss"].sum()
        pushes = completed["record_push"].sum()
        stakes = group["stake"].sum(skipna=True)
        net = completed["net_profit"].sum(skipna=True)
        roi = net / completed["stake"].sum(skipna=True) if completed["stake"].sum(skipna=True) else np.nan
        clv = completed["clv_line"].mean(skipna=True)
        clv_price = completed["clv_price"].mean(skipna=True)

        summary_rows.append(
            {
                "segment": label,
                "bets_total": len(group),
                "bets_completed": len(completed),
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "pending": len(pending),
                "stake_total": stakes,
                "net_profit_completed": net,
                "roi_completed": roi,
                "avg_clv_line": clv,
                "avg_clv_price": clv_price,
                "pending_to_win": pending["to_win"].sum(skipna=True),
            }
        )
    summary = pd.DataFrame(summary_rows)
    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    print(f"[info] Wrote summary metrics: {SUMMARY_OUTPUT}")
    return summary


def parse_args() -> argparse.Namespace:
    default_input = Path.home() / "Desktop" / "betonline.csv"
    default_archive = _default_archive_path()
    parser = argparse.ArgumentParser(description="Normalize wager export(s) into the master ledger.")
    parser.add_argument("--input", nargs="+", default=[default_input], type=Path, help="Input CSV file(s) exported from sportsbook portal(s).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to the master ledger CSV (default tracking/wagers_master.csv).")
    parser.add_argument(
        "--oddslogic-archive",
        type=Path,
        default=default_archive,
        help="Path to OddsLogic archive (default reads $ODDSLOGIC_ARCHIVE_DIR or ../data/oddslogic/oddslogic_ncaa_all).",
    )
    parser.add_argument(
        "--closing-providers",
        nargs="+",
        default=DEFAULT_CLOSING_PROVIDERS,
        help="Preferred providers to load from OddsLogic for closing numbers.",
    )
    parser.add_argument(
        "--closing-priority",
        nargs="+",
        default=None,
        help="Provider priority order when assigning closings (defaults to --closing-providers list).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_df = load_inputs(args.input)
    master = update_master(input_df, args.output)
    archive_path = args.oddslogic_archive
    provider_list_raw = args.closing_providers or []
    provider_list: list[str] = []
    seen_providers: set[str] = set()
    for provider in provider_list_raw:
        canonical = canonical_provider_name(provider) or provider
        key = normalize_provider_key(canonical)
        if not key or key in seen_providers:
            continue
        seen_providers.add(key)
        provider_list.append(canonical)

    priority_raw = args.closing_priority or provider_list
    priority_list: list[str] = []
    seen_priority: set[str] = set()
    for provider in priority_raw:
        canonical = canonical_provider_name(provider) or provider
        key = normalize_provider_key(canonical)
        if not key or key in seen_priority:
            continue
        seen_priority.add(key)
        priority_list.append(canonical)
    if not priority_list:
        priority_list = provider_list

    if archive_path:
        try:
            game_numbers = _collect_game_numbers(master)
            if game_numbers:
                closings_df = _load_oddslogic_subset(archive_path, game_numbers, providers=provider_list)
                closing_map = _build_closing_map(closings_df)
                if closing_map:
                    master = _apply_closing_data(master, closing_map, provider_priority=priority_list)
                    master.to_csv(args.output, index=False)
                    print(f"[info] Applied closing data for {len(closing_map)} games.")
                else:
                    print("[warn] No matching OddsLogic closing data found; CLV fields remain unchanged.")
            else:
                print("[warn] No rotation numbers present; skipping OddsLogic closing merge.")
        except FileNotFoundError as exc:
            print(f"[warn] {exc}")

    summarize(master)


if __name__ == "__main__":
    main()
