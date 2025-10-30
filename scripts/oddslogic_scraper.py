#!/usr/bin/env python3
"""
Automate retrieval of OddsLogic archived odds.

For each date in a supplied range the script:
  * downloads sportsbook metadata,
  * retrieves the archived schedule (all.txt) to resolve OddsLogic game
    numbers into team/kickoff metadata, and
  * downloads the archived lines feed (lines-all.txt) to capture every
    sportsbook line update.

The final CSV (one per date) contains normalized columns that can be
joined to modelling datasets (FBS/FCS spreads/totals). Raw payloads are
also cached for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

BASE_URL = "https://odds.oddslogic.com/OddsLogic"
PHP_BASE = f"{BASE_URL}/sources/php/get.php"
SPORTSBOOKS_ENDPOINT = f"{PHP_BASE}?data=sportsbooks.txt"
LINES_ENDPOINT = f"{PHP_BASE}?data=lines-all.txt&date={{date}}"
SCHEDULE_ENDPOINT = f"{PHP_BASE}?data=all.txt&date={{date}}"

USER_AGENT = "oddslogic-archive-fetcher/1.1"

FRACTION_MAP = {
    "½": 0.5,
    "¼": 0.25,
    "¾": 0.75,
}

ALLOWED_CLASSIFICATIONS = {"fbs", "fcs"}
ALLOWED_ROW_TYPES = {1, 2}


@dataclass
class Sportsbook:
    sportsbook_id: int
    active: bool
    name: str
    abbr: str
    restricted: bool
    url: Optional[str]
    score_in_line_column: bool


@dataclass
class ScheduleEntry:
    game_number: int
    event_id: Optional[int]
    header: str
    classification: str
    start_date: Optional[str]
    start_time_local: Optional[str]
    start_datetime: Optional[str]
    home_team: Optional[str]
    home_abbr: Optional[str]
    away_team: Optional[str]
    away_abbr: Optional[str]
    home_key: Optional[str] = field(init=False)
    away_key: Optional[str] = field(init=False)

    def __post_init__(self) -> None:
        self.home_key = normalize_label(self.home_team or "")
        self.away_key = normalize_label(self.away_team or "")


@dataclass
class LineRecord:
    line_id: str
    game_number: Optional[int]
    period: Optional[int]
    line_type: Optional[int]
    sportsbook_id: Optional[int]
    sportsbook_name: Optional[str]
    row: Optional[int]
    is_opener: bool
    last_line_flag: int
    timestamp: Optional[int]
    value_raw: str
    arrow: str
    takeback: Optional[int]
    classification: str
    header: str
    event_id: Optional[int]
    start_date: Optional[str]
    start_time_local: Optional[str]
    start_datetime: Optional[str]
    home_team: Optional[str]
    home_abbr: Optional[str]
    away_team: Optional[str]
    away_abbr: Optional[str]
    home_key: Optional[str]
    away_key: Optional[str]
    row_type: str
    line_value: Optional[float]
    line_price: Optional[int]
    total_direction: Optional[str]


def daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    step = dt.timedelta(days=1)
    current = start
    while current <= end:
        yield current
        current += step


def http_get(url: str, *, retries: int = 3, sleep_seconds: float = 1.5) -> str:
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            payload = response.text.strip()
            if payload == "No data found":
                return ""
            return payload
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
            if attempt == retries - 1:
                raise
            time.sleep(sleep_seconds)
    return ""


def parse_sportsbooks(raw: str) -> Dict[int, Sportsbook]:
    if not raw:
        raise RuntimeError("Empty sportsbooks payload")
    text = raw.strip()
    if text.startswith("{{") and text.endswith("}}"):
        text = text[2:-2]
    entries = text.split("}{")
    mapping: Dict[int, Sportsbook] = {}
    for entry in entries:
        if not entry:
            continue
        fields = entry.split(",")
        while len(fields) < 7:
            fields.append("")
        sportsbook_id = int(fields[0])
        active = fields[1] == "1"
        name = fields[2].strip()
        abbr = fields[3].strip()
        restricted = fields[4] == "1"
        url = fields[5].strip() or None
        score_in_line_column = fields[6] == "1"
        mapping[sportsbook_id] = Sportsbook(
            sportsbook_id=sportsbook_id,
            active=active,
            name=name,
            abbr=abbr,
            restricted=restricted,
            url=url,
            score_in_line_column=score_in_line_column,
        )
    return mapping


def chunk_lines(raw: str) -> Iterable[str]:
    """Yield individual `{...}` chunks from the compact lines-all payload."""
    if not raw:
        return []
    start = 0
    length = len(raw)
    while start < length:
        start = raw.find("{", start)
        if start == -1:
            break
        end = raw.find("}", start)
        if end == -1:
            break
        yield raw[start + 1 : end]
        start = end + 1


def split_fields(chunk: str) -> List[str]:
    fields: List[str] = []
    buffer: List[str] = []
    in_quotes = False
    i = 0
    while i < len(chunk):
        char = chunk[i]
        if char == '"' and not in_quotes:
            in_quotes = True
            buffer.append(char)
        elif char == '"' and in_quotes:
            buffer.append(char)
            if i + 1 < len(chunk) and chunk[i + 1] == '"':
                buffer.append('"')
                i += 1
            else:
                in_quotes = False
        elif char == "," and not in_quotes:
            fields.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(char)
        i += 1
    if buffer or chunk.endswith(","):
        fields.append("".join(buffer).strip())
    return fields


def parse_line_id(line_id: str) -> Dict[str, Optional[int]]:
    suffix = ""
    if line_id.endswith("o1"):
        suffix = "o1"
        line_id = line_id[:-2]
    try:
        t_index = line_id.index("t")
        p_index = line_id.index("p", t_index)
        l_index = line_id.index("l", p_index)
        b_index = line_id.index("b", l_index)
        r_index = line_id.index("r", b_index)
    except ValueError:
        return {
            "game_number": None,
            "period": None,
            "line_type": None,
            "book_id": None,
            "row": None,
            "is_opener": suffix == "o1",
        }
    try:
        game_number = int(line_id[t_index + 1 : p_index])
    except ValueError:
        game_number = None
    try:
        period = int(line_id[p_index + 1 : l_index])
    except ValueError:
        period = None
    try:
        line_type = int(line_id[l_index + 1 : b_index])
    except ValueError:
        line_type = None
    try:
        book_id = int(line_id[b_index + 1 : r_index])
    except ValueError:
        book_id = None
    try:
        row = int(line_id[r_index + 1 :])
    except ValueError:
        row = None
    return {
        "game_number": game_number,
        "period": period,
        "line_type": line_type,
        "book_id": book_id,
        "row": row,
        "is_opener": suffix == "o1",
    }


def normalize_label(label: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", label.upper())


def split_top_level(text: str) -> List[str]:
    chunks: List[str] = []
    level = 0
    buffer: List[str] = []
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
                buffer = []
        else:
            if level > 0:
                buffer.append(char)
    return chunks


def classify_header(header: str) -> str:
    upper = header.upper()
    if "COLLEGE FOOTBALL" in upper and "FCS" in upper:
        return "fcs"
    if "COLLEGE FOOTBALL" in upper and "FCS" not in upper:
        return "fbs"
    return "other"


def parse_schedule(raw: str) -> Dict[int, ScheduleEntry]:
    if not raw:
        return {}
    entries: Dict[int, ScheduleEntry] = {}
    blocks = split_top_level(raw)
    if not blocks:
        return entries

    for block in blocks[1:]:
        block_upper = block.upper()
        if "COLLEGE FOOTBALL" in block_upper and "FCS" in block_upper:
            classification = "fcs"
        elif "COLLEGE FOOTBALL" in block_upper:
            classification = "fbs"
        else:
            classification = "other"
        pieces = split_top_level(block)
        if not pieces:
            continue
        header_fields = pieces[0].split(",")
        header_text = ",".join(header_fields[5:]).strip('"') if len(header_fields) >= 6 else ""
        for piece in pieces[1:]:
            if not piece:
                continue
            team_start = piece.rfind("{")
            team_part = piece[team_start + 1 : -1] if team_start != -1 else ""
            fields_part = (
                piece[: team_start - 1]
                if team_start > 0 and piece[team_start - 1] == ","
                else piece[:team_start]
            )
            fields = [f.strip('"') for f in fields_part.split(",") if f is not None]
            team_fields = [t.strip('"') for t in team_part.split(",")]
            if not fields:
                continue
            try:
                game_number = int(fields[0])
            except ValueError:
                continue
            event_id = None
            if len(fields) > 2 and fields[2]:
                try:
                    event_id = int(fields[2])
                except ValueError:
                    event_id = None
            start_date = fields[4] if len(fields) > 4 else None
            start_time_local = fields[7] if len(fields) > 7 else None
            start_datetime = fields[9] if len(fields) > 9 else None
            home_team = team_fields[0] if len(team_fields) >= 1 else None
            home_abbr = team_fields[1] if len(team_fields) >= 2 else None
            away_team = team_fields[5] if len(team_fields) >= 6 else None
            away_abbr = team_fields[6] if len(team_fields) >= 7 else None

            entries[game_number] = ScheduleEntry(
                game_number=game_number,
                event_id=event_id,
                header=header_text,
                classification=classification,
                start_date=start_date,
                start_time_local=start_time_local,
                start_datetime=start_datetime,
                home_team=home_team,
                home_abbr=home_abbr,
                away_team=away_team,
                away_abbr=away_abbr,
            )
    return entries


def convert_fractional(value: str) -> str:
    if not value:
        return value

    def replace_mixed(match: re.Match[str]) -> str:
        whole = int(match.group(1))
        frac_char = match.group(2)
        frac_val = FRACTION_MAP.get(frac_char, 0.0)
        magnitude = abs(whole) + frac_val
        signed = magnitude if whole >= 0 else -magnitude
        return f"{signed}"

    def replace_bare(match: re.Match[str]) -> str:
        sign = match.group(1)
        frac_char = match.group(2)
        frac_val = FRACTION_MAP.get(frac_char, 0.0)
        if sign == "-":
            frac_val = -frac_val
        return f"{frac_val}"

    transformed = re.sub(r"([+-]?\d+)([¼½¾])", replace_mixed, value)
    transformed = re.sub(r"([+-]?)([¼½¾])", replace_bare, transformed)
    return transformed


def convert_price_suffix(price_str: str) -> Optional[int]:
    if not price_str:
        return None
    try:
        raw = int(price_str)
    except ValueError:
        return None
    if abs(raw) >= 100 or raw == 0:
        return raw if raw != 0 else 100
    if raw > 0:
        return 100 + raw
    return -100 + raw


def parse_line_value(raw_value: str) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    if not raw_value:
        return None, None, None
    value = html.unescape(raw_value).replace("−", "-").replace(" ", "")
    if "Final" in value or "/" in value or "%" in value:
        return None, None, None
    direction = None
    if value and value[0].lower() in {"o", "u"}:
        direction = value[0].lower()
        value = value[1:]
    value = convert_fractional(value)
    if value.endswith("ev"):
        value = value[:-2] + "+00"
    match = re.match(r"([+-]?\d+(?:\.\d+)?)([+-]\d+)?", value)
    if not match:
        return None, None, direction
    try:
        line_value = float(match.group(1))
    except ValueError:
        return None, None, direction
    price_suffix = match.group(2) or ""
    line_price = convert_price_suffix(price_suffix) if price_suffix else None
    return line_value, line_price, direction


def parse_lines_payload(
    raw: str, sportsbook_map: Dict[int, Sportsbook], schedule: Dict[int, ScheduleEntry]
) -> List[LineRecord]:
    records: List[LineRecord] = []
    for chunk in chunk_lines(raw):
        fields = split_fields(chunk)
        if not fields:
            continue
        line_id = fields[0]
        last_line_flag = int(fields[1]) if len(fields) > 1 and fields[1].strip() else 0
        try:
            timestamp = int(fields[2]) if len(fields) > 2 and fields[2] else None
        except ValueError:
            timestamp = None
        value_raw = html.unescape(fields[3]) if len(fields) > 3 else ""
        arrow = fields[4] if len(fields) > 4 else ""
        try:
            takeback = int(fields[5]) if len(fields) > 5 and fields[5].strip() else 0
        except ValueError:
            takeback = 0

        id_info = parse_line_id(line_id)
        book_id = id_info["book_id"]
        sportsbook = sportsbook_map.get(book_id) if book_id is not None else None
        schedule_entry = schedule.get(id_info["game_number"] or -1)
        classification = schedule_entry.classification if schedule_entry else "unknown"
        classification_lower = classification.lower()
        # Enforce rotation-number length heuristic (FBS: 3 digits, FCS: 6 digits).
        game_number = id_info["game_number"]
        if classification_lower == "fbs":
            if not (isinstance(game_number, int) and 100 <= game_number <= 999):
                continue
        elif classification_lower == "fcs":
            if not (isinstance(game_number, int) and 100000 <= game_number <= 999999):
                continue
        else:
            continue

        if classification_lower not in ALLOWED_CLASSIFICATIONS:
            continue
        row = id_info["row"]
        if row not in ALLOWED_ROW_TYPES:
            continue
        line_value, line_price, direction = parse_line_value(value_raw)
        if line_value is None:
            continue
        value_lower = value_raw.lower()
        abs_value = abs(line_value)
        if direction in {"o", "u"} or value_lower.startswith(("o", "u")):
            row_type = "total"
        elif value_lower.startswith(("pk", "pick")):
            row_type = "spread"
        elif value_lower.startswith(("+", "-")):
            row_type = "spread"
        elif abs_value >= 25:
            row_type = "total"
        else:
            row_type = "spread"

        records.append(
            LineRecord(
                line_id=line_id,
                game_number=id_info["game_number"],
                period=id_info["period"],
                line_type=id_info["line_type"],
                sportsbook_id=book_id,
                sportsbook_name=sportsbook.name if sportsbook else None,
                row=row,
                is_opener=id_info["is_opener"],
                last_line_flag=last_line_flag,
                timestamp=timestamp,
                value_raw=value_raw,
                arrow=arrow,
                takeback=takeback,
                classification=classification,
                header=schedule_entry.header if schedule_entry else "",
                event_id=schedule_entry.event_id if schedule_entry else None,
                start_date=schedule_entry.start_date if schedule_entry else None,
                start_time_local=schedule_entry.start_time_local if schedule_entry else None,
                start_datetime=schedule_entry.start_datetime if schedule_entry else None,
                home_team=schedule_entry.home_team if schedule_entry else None,
                home_abbr=schedule_entry.home_abbr if schedule_entry else None,
                away_team=schedule_entry.away_team if schedule_entry else None,
                away_abbr=schedule_entry.away_abbr if schedule_entry else None,
                home_key=schedule_entry.home_key if schedule_entry else None,
                away_key=schedule_entry.away_key if schedule_entry else None,
                row_type=row_type,
                line_value=line_value,
                line_price=line_price,
                total_direction=direction,
            )
        )
    return records


def write_csv(path: Path, records: List[LineRecord], date_str: str) -> None:
    fieldnames = [
        "date",
        "classification",
        "category_header",
        "event_id",
        "line_id",
        "game_number",
        "period",
        "line_type",
        "row",
        "is_opener",
        "last_line_flag",
        "timestamp",
        "sportsbook_id",
        "sportsbook_name",
        "start_date",
        "start_time_local",
        "start_datetime",
        "home_team",
        "home_abbr",
        "away_team",
        "away_abbr",
        "home_key",
        "away_key",
        "row_type",
        "value",
        "arrow",
        "takeback",
        "line_value",
        "line_price",
        "total_direction",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "date": date_str,
                    "classification": record.classification,
                    "category_header": record.header,
                    "event_id": record.event_id,
                    "line_id": record.line_id,
                    "game_number": record.game_number,
                    "period": record.period,
                    "line_type": record.line_type,
                    "row": record.row,
                    "is_opener": int(record.is_opener),
                    "last_line_flag": record.last_line_flag,
                    "timestamp": record.timestamp,
                    "sportsbook_id": record.sportsbook_id,
                    "sportsbook_name": record.sportsbook_name or "",
                    "start_date": record.start_date or "",
                    "start_time_local": record.start_time_local or "",
                    "start_datetime": record.start_datetime or "",
                    "home_team": record.home_team or "",
                    "home_abbr": record.home_abbr or "",
                    "away_team": record.away_team or "",
                    "away_abbr": record.away_abbr or "",
                    "home_key": record.home_key or "",
                    "away_key": record.away_key or "",
                    "row_type": record.row_type,
                    "value": record.value_raw,
                    "arrow": record.arrow,
                    "takeback": record.takeback if record.takeback is not None else "",
                    "line_value": record.line_value if record.line_value is not None else "",
                    "line_price": record.line_price if record.line_price is not None else "",
                    "total_direction": record.total_direction or "",
                }
            )


def dump_raw(path: Path, payload: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download OddsLogic archive feeds.")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("oddslogic_archive"),
        help="Destination directory (default: oddslogic_archive)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Delay in seconds between HTTP requests (default: 1.0)",
    )
    parser.add_argument(
        "--refresh-sportsbooks",
        action="store_true",
        help="Force re-download of sportsbook metadata even if a cached mapping exists.",
    )
    args = parser.parse_args()

    try:
        start_date = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = dt.datetime.strptime(args.end, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"Invalid date format: {exc}") from exc
    if end_date < start_date:
        raise SystemExit("End date must be on or after start date.")

    output_dir: Path = args.out
    raw_dir = output_dir / "raw"
    csv_dir = output_dir / "csv"
    mapping_path = output_dir / "sportsbooks.json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    sportsbook_map: Dict[int, Sportsbook]
    if mapping_path.exists() and not args.refresh_sportsbooks:
        try:
            with mapping_path.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            sportsbook_map = {
                int(sid): Sportsbook(**data) for sid, data in cached.items()
            }
            print("Loaded sportsbook metadata from cache.", flush=True)
        except (json.JSONDecodeError, TypeError, ValueError, OSError):
            print("Cached sportsbooks.json unreadable; refetching…", flush=True)
            args.refresh_sportsbooks = True
    if not mapping_path.exists() or args.refresh_sportsbooks:
        print("Fetching sportsbook metadata…", flush=True)
        raw_sportsbooks = http_get(SPORTSBOOKS_ENDPOINT)
        if not raw_sportsbooks:
            raise RuntimeError("Empty sportsbooks payload")
        sportsbook_map = parse_sportsbooks(raw_sportsbooks)
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump({sid: sb.__dict__ for sid, sb in sportsbook_map.items()}, f, indent=2)

    for current_date in daterange(start_date, end_date):
        date_str = current_date.isoformat()
        print(f"Processing {date_str}…", flush=True)

        schedule_raw = http_get(SCHEDULE_ENDPOINT.format(date=date_str))
        schedule = parse_schedule(schedule_raw)
        dump_raw(raw_dir / f"schedule_{date_str}.txt", schedule_raw)

        lines_raw = http_get(LINES_ENDPOINT.format(date=date_str))
        if not lines_raw:
            print(f"  ! No lines feed for {date_str}")
            time.sleep(args.sleep)
            continue

        records = parse_lines_payload(lines_raw, sportsbook_map, schedule)
        if not records:
            print(f"  ! Parsed zero line records for {date_str}", file=sys.stderr)
            time.sleep(args.sleep)
            continue

        dump_raw(raw_dir / f"lines-all_{date_str}.txt", lines_raw)
        write_csv(csv_dir / f"oddslogic_lines_{date_str}.csv", records, date_str)
        print(f"  ✓ wrote {len(records):,} records")
        time.sleep(args.sleep)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
