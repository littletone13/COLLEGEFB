#!/usr/bin/env python3
"""Cron-friendly OddsLogic archive ingestion with checksum tracking."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.oddslogic_scraper import run_scrape


def parse_args() -> argparse.Namespace:
    default_out = os.environ.get("ODDSLOGIC_ARCHIVE_DIR", str(Path("oddslogic_archive_current").resolve()))
    parser = argparse.ArgumentParser(description="Incrementally ingest OddsLogic archives.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(default_out).expanduser(),
        help="Destination directory (default: $ODDSLOGIC_ARCHIVE_DIR or oddslogic_archive_current).",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD). When omitted, defaults to today minus --days.",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD). When omitted, defaults to today.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help="Number of trailing days to ingest when start/end not provided (default: 2).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Delay between HTTP requests passed through to the scraper (default: 1.0).",
    )
    parser.add_argument(
        "--refresh-sportsbooks",
        action="store_true",
        help="Force sportsbook metadata refresh even if cache exists.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional explicit path for checksum manifest (default: <out>/checksums.json).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose scraper output.",
    )
    return parser.parse_args()


def parse_date(date_str: str) -> dt.date:
    try:
        return dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"Invalid date '{date_str}': {exc}") from exc


def determine_window(args: argparse.Namespace) -> tuple[dt.date, dt.date]:
    today = dt.date.today()
    if args.start:
        start = parse_date(args.start)
    else:
        start = today - dt.timedelta(days=max(args.days, 1) - 1)
    if args.end:
        end = parse_date(args.end)
    else:
        end = today
    if end < start:
        raise SystemExit("End date must be on or after start date.")
    return start, end


def compute_sha256(path: Path, chunk_size: int = 65536) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> Dict[str, Dict[str, object]]:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data  # type: ignore[return-value]
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def update_manifest(
    manifest: Dict[str, Dict[str, object]],
    files: Iterable[Path],
    *,
    out_dir: Path,
) -> tuple[list[str], list[str]]:
    updated: list[str] = []
    unchanged: list[str] = []
    for file_path in files:
        if not file_path or not file_path.exists():
            continue
        rel_path = str(file_path.relative_to(out_dir))
        mtime = dt.datetime.fromtimestamp(file_path.stat().st_mtime, tz=dt.timezone.utc)
        info = {
            "sha256": compute_sha256(file_path),
            "size": file_path.stat().st_size,
            "modified": mtime.isoformat(timespec="seconds"),
        }
        if manifest.get(rel_path) == info:
            unchanged.append(rel_path)
            continue
        manifest[rel_path] = info
        updated.append(rel_path)
    return updated, unchanged


def gather_files(summary: list[dict]) -> set[Path]:
    files: set[Path] = set()
    for entry in summary:
        for key in ("csv_path", "lines_path", "schedule_path"):
            path = entry.get(key)
            if isinstance(path, Path):
                files.add(path)
    return files


def main() -> int:
    args = parse_args()
    start, end = determine_window(args)

    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest or (out_dir / "checksums.json")

    summary = run_scrape(
        start,
        end,
        out_dir,
        sleep=args.sleep,
        refresh_sportsbooks=args.refresh_sportsbooks,
        verbose=not args.quiet,
    )

    manifest = load_manifest(manifest_path)
    files = gather_files(summary)
    updated, _ = update_manifest(manifest, files, out_dir=out_dir)

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    total_records = sum(entry.get("records", 0) or 0 for entry in summary)
    ingested_days = sum(1 for entry in summary if entry.get("records", 0))

    print(
        f"Ingested {ingested_days} day(s) covering {total_records:,} line records."
        f" Updated {len(updated)} file checksum(s)."
    )

    if updated:
        for rel in updated:
            print(f"  • {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
