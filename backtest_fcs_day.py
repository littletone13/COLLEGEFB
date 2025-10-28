"""Backtest a saved FCS projection slate against real results for a single date.

This script is a lightweight wrapper around the ad-hoc analysis we ran in the
CLI earlier. It uses the ESPN FCS scoreboard (groups=81) as the source of
official finals because the NCAA scoreboard endpoint has become unreliable for
historical dates. Team labels are mapped into the PFF naming set via the same
alias tables that power ``simulate_fcs_week.py`` to ensure consistent joins.
"""
from __future__ import annotations

import argparse
import difflib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional
import re

import pandas as pd
import requests

import ncaa_stats
from simulate_fcs_week import DISPLAY_NAME_OVERRIDES, TEAM_NAME_ALIASES, _normalize_label


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest a single-day FCS slate.")
    parser.add_argument(
        "--date",
        dest="target_date",
        type=str,
        help="Date to backtest in YYYY-MM-DD (defaults to yesterday).",
    )
    parser.add_argument(
        "--projections",
        type=Path,
        help="Path to the projections CSV (defaults to latest matchup_projections_*_ncaa.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV output path for the merged per-game rows (defaults to fcs_backtest_<date>.csv).",
    )
    parser.add_argument(
        "--closing-lines",
        type=Path,
        help="Optional CSV of sportsbook closing lines to compare against model projections.",
    )
    return parser.parse_args()


def _default_date(value: Optional[str]) -> date:
    if value:
        return datetime.strptime(value, "%Y-%m-%d").date()
    return date.today() - timedelta(days=1)


def _resolve_projection_path(path: Optional[Path]) -> Path:
    if path:
        return path
    candidates = sorted(Path.cwd().glob("matchup_projections_*_ncaa.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No matchup_projections_*_ncaa.csv files found; provide --projections.")
    return candidates[0]


PFF_NAMES = list(ncaa_stats.SLUG_TO_PFF.values())
PFF_SET = set(PFF_NAMES)


def _map_team(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    raw = label.upper().strip()
    if raw in PFF_SET:
        return raw
    normalized = _normalize_label(raw)
    alias = TEAM_NAME_ALIASES.get(normalized)
    if alias:
        return alias
    match = difflib.get_close_matches(raw, PFF_NAMES, n=1, cutoff=0.5)
    return match[0] if match else None


def _resolve_entry(entry: dict | None) -> Optional[str]:
    if not entry:
        return None
    team = entry.get("team") or {}
    labels: Iterable[Optional[str]] = (
        team.get("location"),
        team.get("abbreviation"),
        team.get("displayName"),
        team.get("shortDisplayName"),
        team.get("name"),
    )
    for label in labels:
        mapped = _map_team(label)
        if mapped:
            return mapped
    return None


def _fetch_results(day: date) -> pd.DataFrame:
    url = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
    params = {"groups": 81, "dates": day.strftime("%Y%m%d")}
    data = requests.get(url, params=params, timeout=30).json()
    records: list[dict] = []
    for event in data.get("events", []):
        status = (event.get("status") or {}).get("type") or {}
        if not status.get("completed"):
            continue
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        teams = comp.get("competitors") or []
        home = next((team for team in teams if team.get("homeAway") == "home"), None)
        away = next((team for team in teams if team.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        home_code = _resolve_entry(home)
        away_code = _resolve_entry(away)
        if not home_code or not away_code:
            continue
        try:
            home_score = float(home.get("score"))
            away_score = float(away.get("score"))
        except (TypeError, ValueError):
            continue
        records.append(
            {
                "home_team": DISPLAY_NAME_OVERRIDES.get(home_code, home_code),
                "away_team": DISPLAY_NAME_OVERRIDES.get(away_code, away_code),
                "actual_home_points": home_score,
                "actual_away_points": away_score,
                "start_date_actual": comp.get("date"),
            }
        )
    return pd.DataFrame(records)


def _standardize_half(token: str) -> str:
    replacements = {
        "V2": "½",
        "Y2": "½",
        "y2": "½",
        "Yz": "½",
        "yz": "½",
        "V": "½",
    }
    for src, dst in replacements.items():
        token = token.replace(src, dst)
    return token


def _clean_team_label(label: object) -> str:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return ""
    text = str(label).strip()
    if not text:
        return text
    tokens = []
    stop_words = {"Final", "End", "OT", "OT2", "4th", "Q"}
    for tok in text.split():
        normalized = tok.strip()
        if normalized in stop_words:
            break
        if re.search(r"\d", normalized) or ":" in normalized:
            break
        tokens.append(normalized)
    return " ".join(tokens) if tokens else text


def _parse_line_value(token: object) -> float | None:
    if token is None or (isinstance(token, float) and pd.isna(token)):
        return None
    if isinstance(token, (int, float)):
        return float(token)
    token = str(token).strip()
    if not token:
        return None
    token = _standardize_half(token)
    token = token.replace("½", ".5")
    token = token.replace("~", "-")
    token = token.replace("“", "").replace("”", "")
    match = re.search(r"([+-]?\d+(?:\.5)?)", token)
    if not match:
        return None
    value = float(match.group(1))
    while abs(value) >= 100:
        value /= 10.0
    return value


def _load_closing_lines(path: Path, season_date: date) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df

    df = df.copy()
    missing_home_by_rot = {
        308929: "North Carolina Central",
        308931: "Fordham",
        308933: "Maine",
        308935: "New Hampshire",
        308939: "Valparaiso",
        308941: "North Carolina A&T",
        308943: "Tennessee Tech",
    }
    def fill_home(row: pd.Series) -> object:
        rot = row.get("away_rot")
        if pd.isna(row.get("home_team")) and pd.notna(rot):
            rot_int = int(rot)
            if rot_int in missing_home_by_rot:
                return missing_home_by_rot[rot_int]
        return row.get("home_team")

    df["home_team"] = df.apply(fill_home, axis=1)
    df["away_team_clean"] = df["away_team"].apply(_clean_team_label)
    df["home_team_clean"] = df["home_team"].apply(_clean_team_label)
    df["away_team_norm"] = df["away_team_clean"].apply(_map_team)
    df["home_team_norm"] = df["home_team_clean"].apply(_map_team)
    df = df.dropna(subset=["away_team_norm", "home_team_norm"])

    spread_cols = [
        "spread_sports411",
        "spread_circa",
        "spread_pinnacle",
        "spread_betonline",
    ]
    total_cols = [
        "total_sports411",
        "total_circa",
        "total_pinnacle",
        "total_betonline",
    ]

    def looks_like_total(token: object) -> bool:
        if not isinstance(token, str):
            return False
        lower = token.lower()
        return ("o" in lower or "u" in lower or bool(re.search(r"\d0-\d", lower)))

    for idx, row in df.iterrows():
        if any(looks_like_total(row.get(col)) for col in spread_cols) and not any(
            looks_like_total(row.get(col)) for col in total_cols
        ):
            # Swap: the OCR placed totals where spreads should be (commonly on Ivy matchups).
            df.loc[idx, spread_cols], df.loc[idx, total_cols] = (
                df.loc[idx, total_cols].values,
                df.loc[idx, spread_cols].values,
            )

    for col in spread_cols + total_cols:
        df[f"{col}_value"] = df[col].apply(_parse_line_value)

    def pick(row: pd.Series, cols: list[str]) -> float | None:
        for col in cols:
            value = row.get(col)
            if pd.notna(value):
                return float(value)
        return None

    df["closing_spread"] = df.apply(
        pick,
        axis=1,
        cols=[
            "spread_circa_value",
            "spread_sports411_value",
            "spread_pinnacle_value",
            "spread_betonline_value",
        ],
    )
    df["closing_total"] = df.apply(
        pick,
        axis=1,
        cols=[
            "total_circa_value",
            "total_sports411_value",
            "total_pinnacle_value",
            "total_betonline_value",
        ],
    )

    df["closing_date"] = season_date

    return df[
        [
            "closing_date",
            "away_team_norm",
            "home_team_norm",
            "closing_spread",
            "closing_total",
            "spread_circa",
            "total_circa",
        ]
    ]


PAIR_CORRECTIONS = {
    ("W CAROLINA", "DELWARE ST"): ("NC CENT", "DELWARE ST"),
    ("W CAROLINA", "CAMPBELL"): ("NC A&T", "CAMPBELL"),
    ("TENN TECH", "SCAR STATE"): ("TENN TECH", "SE MO ST"),
    ("MERCER", "MAINE"): ("MERCER", "VMI"),
    ("ALAB A&M", "ALABAMA ST"): ("ALABAMA ST", "ALAB A&M"),
    ("GRAMBLING", "JACKSON ST"): ("JACKSON ST", "GRAMBLING"),
}


def _apply_pair_correction(home: str, away: str) -> tuple[str, str]:
    return PAIR_CORRECTIONS.get((home, away), (home, away))


def _prepare_projections(path: Path, day: date) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    parsed = pd.to_datetime(df["start_date"], errors="coerce")
    df = df.assign(start_dt=parsed, start_date_pred=df["start_date"])
    mask = parsed.dt.date.isin({day, day + timedelta(days=1)})
    filtered = df.loc[mask].copy() if mask.any() else df.copy()
    rename_map = {
        "spread": "pred_spread",
        "total": "pred_total",
        "home_points": "pred_home_points",
        "away_points": "pred_away_points",
        "home_win_prob": "pred_home_win_prob",
        "home_ml": "pred_home_ml",
        "away_ml": "pred_away_ml",
    }
    return filtered.rename(columns=rename_map)


def _compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df["actual_margin"] = df["actual_home_points"] - df["actual_away_points"]
    df["actual_total"] = df["actual_home_points"] + df["actual_away_points"]
    df["spread_error"] = df["pred_spread"] - df["actual_margin"]
    df["total_error"] = df["pred_total"] - df["actual_total"]
    df["home_flag"] = df["actual_margin"].apply(lambda x: 1.0 if x > 0 else (0.0 if x < 0 else 0.5))
    df["brier"] = (df["pred_home_win_prob"] - df["home_flag"]) ** 2
    return df


def main() -> None:
    args = _parse_args()
    target_date = _default_date(args.target_date)
    projections_path = _resolve_projection_path(args.projections)
    preds = _prepare_projections(projections_path, target_date)
    results = _fetch_results(target_date)
    merged = preds.merge(results, on=["home_team", "away_team"], how="inner")
    if merged.empty:
        raise SystemExit("No overlap between projections and completed games.")

    merged = _compute_metrics(merged)

    if args.closing_lines:
        closings = _load_closing_lines(args.closing_lines, target_date)
        if closings.empty:
            print("Closing lines file loaded but produced no normalized rows; skipping market comparison.")
        else:
            corrections = merged.apply(
                lambda row: _apply_pair_correction(row["home_team"], row["away_team"]), axis=1
            )
            merged[["merge_home_team", "merge_away_team"]] = list(corrections)
            merged["merge_home_team_norm"] = merged["merge_home_team"].apply(_map_team)
            merged["merge_away_team_norm"] = merged["merge_away_team"].apply(_map_team)
            merged = merged.merge(
                closings,
                left_on=["merge_away_team_norm", "merge_home_team_norm"],
                right_on=["away_team_norm", "home_team_norm"],
                how="left",
            )
            if merged["closing_spread"].notna().any():
                merged["spread_edge_vs_close"] = merged["pred_spread"] - merged["closing_spread"]
            if merged["closing_total"].notna().any():
                merged["total_edge_vs_close"] = merged["pred_total"] - merged["closing_total"]

    out_path = args.output or Path(f"fcs_backtest_{target_date.isoformat()}.csv")
    out_columns = [
        "home_team",
        "away_team",
        "start_date_pred",
        "start_date_actual",
        "pred_spread",
        "pred_total",
        "pred_home_points",
        "pred_away_points",
        "pred_home_win_prob",
        "pred_home_ml",
        "pred_away_ml",
        "actual_home_points",
        "actual_away_points",
        "actual_margin",
        "actual_total",
        "spread_error",
        "total_error",
        "brier",
    ]
    extra_columns = [
        col
        for col in [
            "closing_spread",
            "closing_total",
            "spread_edge_vs_close",
            "total_edge_vs_close",
            "spread_circa",
            "total_circa",
        ]
        if col in merged.columns
    ]
    drop_cols = [
        col
        for col in (
            "away_team_norm",
            "home_team_norm",
            "merge_home_team",
            "merge_away_team",
            "merge_home_team_norm",
            "merge_away_team_norm",
        )
        if col in merged.columns
    ]
    merged.drop(columns=drop_cols, inplace=True, errors="ignore")
    merged[out_columns + extra_columns].to_csv(out_path, index=False)

    summary = {
        "games": int(merged.shape[0]),
        "spread_mae": float(merged["spread_error"].abs().mean()),
        "total_mae": float(merged["total_error"].abs().mean()),
        "brier": float(merged["brier"].mean()),
    }

    missing_results = results.shape[0] - merged.shape[0]
    missing_preds = preds.shape[0] - merged.shape[0]

    print(f"Projection file: {projections_path.name}")
    print(f"Results date: {target_date}")
    print(f"Matched games: {summary['games']}")
    print(f"Results without projections: {missing_results}")
    print(f"Unplayed/Unmatched projections: {missing_preds}")
    print(f"Spread MAE: {summary['spread_mae']:.2f}")
    print(f"Total MAE: {summary['total_mae']:.2f}")
    print(f"Brier: {summary['brier']:.3f}")
    print(f"Saved per-game detail to {out_path}")

    if args.closing_lines and "closing_spread" in merged.columns:
        covered = merged["closing_spread"].notna().sum()
        total = merged.shape[0]
        print(f"Closing lines matched for {covered}/{total} games")

    largest = merged.assign(abs_spread_error=merged["spread_error"].abs())
    cols = ["home_team", "away_team", "pred_spread", "actual_margin", "spread_error"]
    print("Top 5 spread misses:")
    print(largest.nlargest(5, "abs_spread_error")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
