"""Simulate upcoming FCS games using the PFF-derived ratings."""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict

import pandas as pd

from cfb.sim import fcs as sim_fcs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FCS projections for a given date window.")
    parser.add_argument("start_date", type=str, help="Anchor date (YYYY-MM-DD) for the slate, e.g., 2025-10-24.")
    parser.add_argument("--days", type=int, default=3, help="Number of days to include starting from start_date (default 3).")
    parser.add_argument("--week", type=int, help="CFBD week number to align with market lines.")
    parser.add_argument("--api-key", type=str, help="Optional CFBD API key (defaults to env).")
    parser.add_argument(
        "--providers",
        type=str,
        help="Comma-separated sportsbook providers to include when blending market priors (default: all).",
    )
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
        market_bits = []
        if pd.notna(row.get("market_spread")):
            market_bits.append(f"Mkt Spread {row['market_spread']:+.1f}")
        if pd.notna(row.get("spread_vs_market")):
            market_bits.append(f"Δ {row['spread_vs_market']:+.1f}")
        if pd.notna(row.get("market_total")):
            market_bits.append(f"Mkt Total {row['market_total']:.1f}")
        if pd.notna(row.get("total_vs_market")):
            market_bits.append(f"Δ {row['total_vs_market']:+.1f}")
        market_text = f" | {' / '.join(market_bits)}" if market_bits else ""
        items.append(
            f"<li><strong>{row['home_team']}</strong> vs <strong>{row['away_team']}</strong>"
            f" | Spread {row['spread']:+.1f} | Total {row['total']:.1f} | Home Win {row['home_win_prob']*100:0.1f}%{market_text}</li>"
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
        "market_spread",
        "market_total",
        "spread_vs_market",
        "total_vs_market",
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
        "Market Spread",
        "Market Total",
        "Spread Δ",
        "Total Δ",
    ]
    table["Home Win%"] = table["Home Win%"] * 100
    for col in ["Market Spread", "Market Total", "Spread Δ", "Total Δ"]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
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
            "Market Spread": "{:+.1f}",
            "Market Total": "{:.1f}",
            "Spread Δ": "{:+.1f}",
            "Total Δ": "{:+.1f}",
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

    api_key = args.api_key or os.environ.get("CFBD_API_KEY")
    provider_filter = [p.strip() for p in args.providers.split(",") if p.strip()] if args.providers else None

    df = sim_fcs.simulate_window(
        anchor,
        days=args.days,
        week=args.week,
        api_key=api_key,
        providers=provider_filter,
    )
    if df.empty:
        print("No upcoming FCS games in the requested window.")
        return

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
