#!/usr/bin/env python3
"""
Render an HTML dashboard summarizing wagers tracked in tracking/wagers_master.csv.

Usage:
    python scripts/render_wager_dashboard.py \
        --master tracking/wagers_master.csv \
        --summary tracking/wagers_summary.csv \
        --html tracking/wagers_overview.html
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MASTER = Path("tracking") / "wagers_master.csv"
DEFAULT_SUMMARY = Path("tracking") / "wagers_summary.csv"
DEFAULT_OUTPUT = Path("tracking") / "wagers_overview.html"

SEGMENT_LABELS = {
    "overall": "All Bets",
    "moneyline": "Moneylines",
    "spread": "Spreads",
    "total": "Totals",
}


def _format_money(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"${value:,.2f}"


def _format_number(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:+.1f}"


def _format_pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:+.1f}%"


def _format_factor(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.3f}"


def build_kpi_cards(summary: pd.DataFrame) -> str:
    overall = summary[summary["segment"] == "overall"].iloc[0]
    cards = [
        ("Total Bets", int(overall["bets_total"])),
        ("Completed", int(overall["bets_completed"])),
        ("Wins / Losses / Pushes",
         f"{int(overall['wins'])} / {int(overall['losses'])} / {int(overall['pushes'])}"),
        ("Stake (All)", _format_money(overall["stake_total"])),
        ("Net Profit", _format_money(overall["net_profit_completed"])),
        ("ROI (Completed)", _format_pct(overall["roi_completed"])),
        ("Pending Bets", int(overall["pending"])),
        ("Pending To Win", _format_money(overall["pending_to_win"])),
        ("Avg CLV (pts)", f"{overall['avg_clv_line']:+.2f}" if np.isfinite(overall["avg_clv_line"]) else "—"),
    ]
    return "".join(
        f"<div class='kpi-card'><span class='kpi-label'>{label}</span><span class='kpi-value'>{value}</span></div>"
        for label, value in cards
    )


def build_segment_table(summary: pd.DataFrame) -> str:
    rows = []
    for _, row in summary.iterrows():
        label = SEGMENT_LABELS.get(row["segment"], row["segment"].title())
        avg_clv_display = _format_number(row.get("avg_clv_line"))
        avg_clv_price_display = _format_number(row.get("avg_clv_price"))
        rows.append(
            f"<tr>"
            f"<td>{label}</td>"
            f"<td>{int(row['bets_total'])}</td>"
            f"<td>{int(row['bets_completed'])}</td>"
            f"<td>{int(row['wins'])}</td>"
            f"<td>{int(row['losses'])}</td>"
            f"<td>{int(row['pushes'])}</td>"
            f"<td>{int(row['pending'])}</td>"
            f"<td>{_format_money(row['stake_total'])}</td>"
            f"<td>{_format_money(row['net_profit_completed'])}</td>"
            f"<td>{_format_pct(row['roi_completed'])}</td>"
            f"<td>{avg_clv_display}</td>"
            f"<td>{avg_clv_price_display}</td>"
            f"</tr>"
        )
    return (
        "<table class='summary-table'>"
        "<thead><tr>"
        "<th>Segment</th><th>Total</th><th>Completed</th><th>Wins</th><th>Losses</th>"
        "<th>Pushes</th><th>Pending</th><th>Stake</th><th>Net</th><th>ROI</th>"
        "<th>Avg CLV pts</th><th>Avg CLV price</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def build_recent_table(master: pd.DataFrame, limit: int = 50) -> str:
    subset = master.sort_values("wager_date", ascending=False).head(limit)
    rows = []
    for _, row in subset.iterrows():
        status_val = row.get("status_normalized")
        if not isinstance(status_val, str) or not status_val.strip():
            status_val = row.get("status")
        status = str(status_val or "").title()
        w_date = row.get("wager_date")
        if isinstance(w_date, pd.Timestamp):
            date_display = w_date.strftime("%Y-%m-%d")
        else:
            date_display = str(w_date or "")
        line_display = _format_number(row.get("bet_line"))
        price_display = _format_number(row.get("bet_price"))
        closing_line_display = _format_number(row.get("closing_line"))
        closing_price_display = _format_number(row.get("closing_price"))
        clv_display = _format_number(row.get("clv_line"))
        clv_price_display = _format_number(row.get("clv_price"))
        closing_val = row.get("closing_provider")
        if isinstance(closing_val, str) and closing_val.strip():
            closing_book = closing_val
        elif pd.isna(closing_val):
            closing_book = "—"
        else:
            closing_book = str(closing_val)
        rows.append(
            "<tr>"
            f"<td>{date_display}</td>"
            f"<td>{row.get('source','')}</td>"
            f"<td>{row.get('matchup','')}</td>"
            f"<td>{row.get('selection','')}</td>"
            f"<td>{row.get('market_type','')}</td>"
            f"<td>{line_display}</td>"
            f"<td>{price_display}</td>"
            f"<td>{closing_line_display}</td>"
            f"<td>{closing_price_display}</td>"
            f"<td>{clv_display}</td>"
            f"<td>{clv_price_display}</td>"
            f"<td>{closing_book}</td>"
            f"<td>{_format_money(row.get('stake'))}</td>"
            f"<td>{_format_money(row.get('to_win'))}</td>"
            f"<td>{_format_money(row.get('net_profit'))}</td>"
            f"<td>{_format_pct(row.get('roi'))}</td>"
            f"<td>{status}</td>"
            "</tr>"
        )
    return (
        "<table class='detail-table'>"
        "<thead><tr>"
        "<th>Date</th><th>Book</th><th>Matchup</th><th>Selection</th><th>Market</th>"
        "<th>Line</th><th>Price</th><th>Closing Line</th><th>Closing Price</th>"
        "<th>CLV (pts)</th><th>CLV (price)</th><th>Closing Book</th>"
        "<th>Stake</th><th>To Win</th><th>Net</th><th>ROI</th><th>Status</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def build_html(master: pd.DataFrame, summary: pd.DataFrame, title: str) -> str:
    kpis_html = build_kpi_cards(summary)
    summary_html = build_segment_table(summary)
    recent_html = build_recent_table(master)
    generated_at = datetime.now(timezone.utc).isoformat()
    return f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>{title}</title>
    <style>
      body {{
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background: #f5f7fb;
        color: #1f2630;
        margin: 0;
      }}
      header {{
        padding: 24px 32px;
        background: #ffffff;
        border-bottom: 1px solid #dfe3eb;
        display: flex;
        flex-direction: column;
        gap: 8px;
      }}
      header h1 {{
        margin: 0;
        font-size: 28px;
      }}
      header .meta {{
        color: #687385;
        font-size: 13px;
      }}
      main {{
        padding: 24px 32px 64px;
        display: flex;
        flex-direction: column;
        gap: 32px;
      }}
      .kpi-grid {{
        display: grid;
        gap: 16px;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      }}
      .kpi-card {{
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #dfe3eb;
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 6px;
      }}
      .kpi-label {{
        font-size: 12px;
        text-transform: uppercase;
        color: #687385;
        letter-spacing: 0.08em;
      }}
      .kpi-value {{
        font-size: 20px;
        font-weight: 600;
      }}
      .summary-table, .detail-table {{
        width: 100%;
        border-collapse: collapse;
        background: #ffffff;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #dfe3eb;
      }}
      .summary-table th, .summary-table td,
      .detail-table th, .detail-table td {{
        padding: 10px 12px;
        text-align: left;
        border-bottom: 1px solid rgba(223,227,235,0.6);
        font-size: 13px;
      }}
      .summary-table th, .detail-table th {{
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #687385;
        background: #f9fbff;
        font-size: 12px;
      }}
      .detail-table tbody tr:nth-child(odd) {{
        background: rgba(3, 131, 90, 0.04);
      }}
      footer {{
        padding: 12px 32px 32px;
        color: #687385;
        font-size: 12px;
      }}
    </style>
  </head>
  <body>
    <header>
      <h1>{title}</h1>
      <div class="meta">Updated: {generated_at}</div>
    </header>
    <main>
      <section>
        <div class="kpi-grid">
          {kpis_html}
        </div>
      </section>
      <section>
        <h2>Segment Summary</h2>
        {summary_html}
      </section>
      <section>
        <h2>Recent Wagers</h2>
        {recent_html}
      </section>
    </main>
    <footer>Tracked wagers sourced from sportsbook exports; CLV fields populate when closing lines are provided.</footer>
  </body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an HTML dashboard for tracked wagers.")
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER, help="Path to wagers_master.csv")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Path to wagers_summary.csv")
    parser.add_argument("--html", type=Path, default=DEFAULT_OUTPUT, help="Output HTML path")
    parser.add_argument("--title", type=str, default="Wager Tracking Overview")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    master = pd.read_csv(args.master, parse_dates=["wager_date"])
    summary = pd.read_csv(args.summary)
    html = build_html(master, summary, args.title)
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(html, encoding="utf-8")
    print(f"[info] Wrote dashboard → {args.html}")


if __name__ == "__main__":
    from datetime import datetime
    main()
