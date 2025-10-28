"""Simulate upcoming FCS games using the PFF-derived ratings."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict

import pandas as pd

import difflib
import json
import re
import requests

import fcs
import ncaa_stats


def _normalize_label(label: str) -> str:
    return re.sub(r"[^A-Z0-9 ]", "", label.upper()).strip()


RAW_TEAM_NAME_ALIASES = {
    "LONG ISLAND": "LIUSHAR",
    "LONG ISLAND UNIVERSITY": "LIUSHAR",
    "LONG ISLAND UNIVERSITY SHARKS": "LIUSHAR",
    "LIU": "LIUSHAR",
    "LIU SHARKS": "LIUSHAR",
    "NEW HAVEN": "NEWHVN",
    "NEW HAVEN CHARGERS": "NEWHVN",
    "UNIVERSITY OF NEW HAVEN": "NEWHVN",
    "RHODE ISLAND": "RHODE ISLD",
    "RHODE ISLAND RAMS": "RHODE ISLD",
    "NORTH DAKOTA": "N DAKOTA",
    "NORTH DAKOTA STATE": "N DAK ST",
    "NORTH DAKOTA ST": "N DAK ST",
    "NDSU": "N DAK ST",
    "SOUTH DAKOTA": "S DAKOTA",
    "SOUTH DAKOTA STATE": "S DAK ST",
    "SOUTH DAKOTA ST": "S DAK ST",
    "SDSU": "S DAK ST",
    "EAST TEXAS A&M": "TXAMCO",
    "EAST TEXAS A AND M": "TXAMCO",
    "E TEXAS A&M": "TXAMCO",
    "NORTH CAROLINA CENTRAL": "NC CENT",
    "NORTH CAROLINA A&T": "NC A&T",
    "NORTH CAROLINA A AND T": "NC A&T",
    "NORTH CAROLINA A AND T STATE": "NC A&T",
    "SOUTHEASTERN LOUISIANA": "SE LA",
    "HOUSTON CHRISTIAN": "HOUCHR",
    "HOUSTON BAPTIST": "HOUCHR",
    "VALPARAISO": "VALPO",
    "MOREHEAD STATE": "MOREHEAD",
    "WILLIAM & MARY": "WM & MARY",
    "WILLIAM AND MARY": "WM & MARY",
    "TENNESSEE TECH": "TENN TECH",
    "SOUTH CAROLINA STATE": "SCAR STATE",
    "SE MISSOURI STATE": "SE MO ST",
    "SE MISSOURI": "SE MO ST",
    "SAINT FRANCIS": "ST FRANCIS",
    "ST FRANCIS (PA)": "ST FRANCIS",
    "ROBERT MORRIS": "ROB MORRIS",
    "PRAIRIE VIEW A&M": "PRVIEW A&M",
    "PRAIRIE VIEW A AND M": "PRVIEW A&M",
    "BETHUNE COOKMAN": "BETH COOK",
    "ALABAMA A&M": "ALAB A&M",
    "ALABAMA STATE": "ALABAMA ST",
    "PORTLAND STATE": "PORTLAND",
    "NICHOLLS STATE": "NICHOLLS",
    "MCNEESE STATE": "MCNEESE",
    "GRAMBLING STATE": "GRAMBLING",
    "JACKSON STATE": "JACKSON ST",
    "STEPHEN F. AUSTIN": "STF AUSTIN",
    "STEPHEN F AUSTIN": "STF AUSTIN",
    "UTAH TECH": "UTAHTC",
    "UT RIO GRANDE VALLEY": "TXGV",
    "INCARNATE WORD": "INCAR WORD",
}


TEAM_NAME_ALIASES = {_normalize_label(key): value for key, value in RAW_TEAM_NAME_ALIASES.items()}

DISPLAY_NAME_OVERRIDES = {
    "TXAMCO": "East Texas A&M",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FCS projections for a given date window.")
    parser.add_argument("start_date", type=str, help="Anchor date (YYYY-MM-DD) for the slate, e.g., 2025-10-24.")
    parser.add_argument("--days", type=int, default=3, help="Number of days to include starting from start_date (default 3).")
    parser.add_argument("--output", type=Path, help="Optional CSV output path.")
    parser.add_argument("--html", type=Path, help="Optional HTML summary output path.")
    return parser.parse_args()


def to_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _format_bullets(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "<p>None.</p>"
    items = []
    for _, row in rows.iterrows():
        items.append(
            f"<li><strong>{row['home_team']}</strong> vs <strong>{row['away_team']}</strong>"
            f" | Spread {row['spread']:+.1f} | Total {row['total']:.1f} | Home Win {row['home_win_prob']*100:0.1f}%</li>"
        )
    return "<ul>" + "".join(items) + "</ul>"


def _render_html(df: pd.DataFrame, title: str) -> str:
    table = df[[
        "start_date",
        "home_team",
        "away_team",
        "spread",
        "total",
        "home_points",
        "away_points",
        "home_win_prob",
        "home_ml",
        "away_ml",
    ]].copy()
    table.columns = [
        "Kickoff",
        "Home",
        "Away",
        "Spread (H-A)",
        "Total",
        "Proj Home",
        "Proj Away",
        "Home Win%",
        "Home ML",
        "Away ML",
    ]
    table["Home Win%"] = table["Home Win%"] * 100
    styled = (
        table.style
        .background_gradient(subset=["Spread (H-A)"], cmap="RdYlGn_r")
        .background_gradient(subset=["Home Win%"], cmap="RdYlGn")
        .format({
            "Spread (H-A)": "{:+.1f}",
            "Total": "{:.1f}",
            "Proj Home": "{:.1f}",
            "Proj Away": "{:.1f}",
            "Home Win%": "{:.1f}",
            "Home ML": "{:+.0f}",
            "Away ML": "{:+.0f}",
        })
        .hide(axis="index")
    )

    close = df[df["spread"].abs() <= 3].sort_values("home_win_prob", ascending=False)
    shootouts = df.sort_values("total", ascending=False).head(5)
    slogs = df.sort_values("total").head(5)
    upsets = df[df["home_win_prob"] <= 0.4].sort_values("home_win_prob").head(5)

    sections = {
        "Close calls (≤ 3 pts)": _format_bullets(close),
        "Projected shootouts": _format_bullets(shootouts),
        "Defensive slogs": _format_bullets(slogs),
        "Upset radar": _format_bullets(upsets),
    }

    section_html = "".join(f"<section><h2>{title}</h2>{body}</section>" for title, body in sections.items())

    return f"""
    <html>
      <head>
        <meta charset='utf-8' />
        <title>{title}</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 2rem; background: #fafafa; color: #222; }}
          h1 {{ margin-bottom: 0.5rem; }}
          section {{ margin-bottom: 1.5rem; }}
          ul {{ margin: 0.4rem 0 0.4rem 1.2rem; }}
        </style>
      </head>
      <body>
        <h1>{title}</h1>
        <p>Generated using PFF-derived opponent/unit ratings plus NCAA scoreboard schedule.</p>
        {section_html}
        <section><h2>Board</h2>{styled.to_html()}</section>
      </body>
    </html>
    """


def main() -> None:
    args = parse_args()
    anchor = to_date(args.start_date)
    end_date = anchor + timedelta(days=args.days)

    def infer_season_year(day: date) -> int:
        return day.year if day.month >= 7 else day.year - 1

    season_year = infer_season_year(anchor)

    ratings = fcs.load_team_ratings(season_year=season_year)
    book = fcs.RatingBook(ratings, fcs.RatingConstants())

    pff_names = list(ncaa_stats.SLUG_TO_PFF.values())
    pff_set = set(pff_names)

    def map_team(name: str | None) -> str | None:
        if not name:
            return None
        raw = name.upper()
        if raw in pff_set:
            return raw
        normalized = _normalize_label(raw)
        alias = TEAM_NAME_ALIASES.get(normalized)
        if alias:
            return alias
        matches = difflib.get_close_matches(raw, pff_names, n=1, cutoff=0.5)
        return matches[0] if matches else None

    def resolve_entry(entry: dict) -> str | None:
        team = entry.get("team") or {}
        labels = [
            team.get("location"),
            team.get("abbreviation"),
            team.get("displayName"),
            team.get("shortDisplayName"),
            team.get("name"),
        ]
        for label in labels:
            candidate = map_team(label)
            if candidate:
                return candidate
        return None

    def fetch_espn_schedule(start: date, end: date) -> pd.DataFrame:
        records: list[dict] = []
        current = start
        while current <= end:
            datestr = current.strftime("%Y%m%d")
            url = (
                "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
                f"?groups=81&dates={datestr}"
            )
            try:
                data = requests.get(url, timeout=30).json()
            except Exception:
                current += timedelta(days=1)
                continue
            for event in data.get("events", []):
                comp = event.get("competitions", [{}])[0]
                competitors = comp.get("competitors", [])
                home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if not home or not away:
                    continue
                home_team = resolve_entry(home)
                away_team = resolve_entry(away)
                start_iso = comp.get("date")
                records.append(
                    {
                        "start_date": start_iso,
                        "home_team": home_team,
                        "away_team": away_team,
                    }
                )
            current += timedelta(days=1)
        df = pd.DataFrame(records)
        if df.empty:
            return df
        df["start_dt"] = pd.to_datetime(df["start_date"], errors="coerce")
        df = df[(df["start_dt"].dt.date >= start) & (df["start_dt"].dt.date <= end)]
        return df.dropna(subset=["home_team", "away_team"])

    def fetch_ncaa_schedule(start: date, end: date, season_year: int) -> pd.DataFrame:
        try:
            scoreboard = ncaa_stats.fetch_scoreboard_games(season_year)
        except Exception:
            return pd.DataFrame(columns=["start_date", "home_team", "away_team"])
        if scoreboard.empty:
            return scoreboard
        scoreboard["date"] = pd.to_datetime(scoreboard["date"], errors="coerce").dt.date
        window = scoreboard[(scoreboard["date"] >= start) & (scoreboard["date"] <= end)].copy()
        if window.empty:
            return pd.DataFrame(columns=["start_date", "home_team", "away_team"])
        window["home_team"] = window["home_slug"].map(ncaa_stats.SLUG_TO_PFF.get)
        window["away_team"] = window["away_slug"].map(ncaa_stats.SLUG_TO_PFF.get)
        window["start_date"] = window["start_date"].fillna(window["date"].apply(lambda d: datetime.combine(d, datetime.min.time()).isoformat()))
        return window.dropna(subset=["home_team", "away_team"])[["start_date", "home_team", "away_team"]]

    espn_slate = fetch_espn_schedule(anchor, end_date)
    ncaa_slate = fetch_ncaa_schedule(anchor, end_date, season_year)

    slate = pd.concat([espn_slate, ncaa_slate], ignore_index=True)
    if not slate.empty:
        slate = slate.drop_duplicates(subset=["start_date", "home_team", "away_team"])
    if slate.empty:
        print("No upcoming FCS games in the requested window.")
        return

    projections: Dict[str, list] = {
        "start_date": [],
        "home_team": [],
        "away_team": [],
        "spread": [],
        "total": [],
        "home_points": [],
        "away_points": [],
        "home_win_prob": [],
        "home_ml": [],
        "away_ml": [],
    }

    for _, row in slate.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        try:
            result = book.predict(home_team, away_team, neutral_site=False)
        except KeyError:
            continue
        base_home = result.get("home_team") or result.get("team_one")
        base_away = result.get("away_team") or result.get("team_two")
        display_home = DISPLAY_NAME_OVERRIDES.get(base_home, base_home)
        display_away = DISPLAY_NAME_OVERRIDES.get(base_away, base_away)
        projections["start_date"].append(row.get("start_date"))
        projections["home_team"].append(display_home)
        projections["away_team"].append(display_away)
        projections["spread"].append(result.get("spread_home_minus_away") or result.get("spread_team_one_minus_team_two"))
        projections["total"].append(result["total_points"])
        projections["home_points"].append(result.get("home_points") or result.get("team_one_points"))
        projections["away_points"].append(result.get("away_points") or result.get("team_two_points"))
        projections["home_win_prob"].append(result.get("home_win_prob") or result.get("team_one_win_prob"))
        projections["home_ml"].append(result.get("home_moneyline") or result.get("team_one_moneyline"))
        projections["away_ml"].append(result.get("away_moneyline") or result.get("team_two_moneyline"))

    df = pd.DataFrame(projections)
    if df.empty:
        print("No upcoming FCS games in the requested window.")
        return

    df = df.sort_values("start_date")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved projections to {args.output}")

    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        html = _render_html(df, title=f"FCS Slate {anchor} to {end_date}")
        args.html.write_text(html, encoding="utf-8")
        print(f"Saved HTML summary to {args.html}")

    if not args.output and not args.html:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
