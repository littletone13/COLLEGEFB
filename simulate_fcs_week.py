"""Simulate upcoming FCS games using the PFF-derived ratings."""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from cfb.sim import fcs as sim_fcs

FCS_TEMPLATE_PATH = Path("template_fcs_report.html")
LOGO_PATH = Path("assets/betting_worldwide_logo.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FCS projections for a given date window.")
    parser.add_argument(
        "start_date",
        type=str,
        nargs="?",
        default=date.today().isoformat(),
        help="Anchor date (YYYY-MM-DD) for the slate (defaults to today).",
    )
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
    parser.add_argument("--template", type=Path, help="Optional HTML template override (defaults to template_fcs_report.html).")
    return parser.parse_args()


def to_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _safe_float(value: object) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return float(val) if np.isfinite(val) else None


def _kelly_fraction(
    model_spread: Optional[float],
    market_spread: Optional[float],
    sigma: Optional[float],
    direction: Optional[str],
) -> Optional[float]:
    if model_spread is None or market_spread is None or sigma is None or sigma <= 0:
        return None
    if direction not in {"home", "away"}:
        return None
    z = (model_spread - market_spread) / sigma
    cover_prob_home = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
    probability = cover_prob_home if direction == "home" else 1.0 - cover_prob_home
    if probability <= 0.0 or probability >= 1.0:
        return None
    b = 100.0 / 110.0
    kelly = (b * probability - (1 - probability)) / b
    return max(0.0, kelly) if kelly > 0 else 0.0


def _edge_bucket(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    magnitude = abs(value)
    if magnitude >= 3.0:
        return "strong"
    if magnitude >= 1.5:
        return "medium"
    if magnitude >= 0.75:
        return "light"
    return None


def _prob_to_moneyline(prob: Optional[float]) -> Optional[float]:
    if prob is None:
        return None
    prob = _safe_float(prob)
    if prob is None:
        return None
    if prob <= 0.0 or prob >= 1.0:
        return None
    if prob >= 0.5:
        return -100.0 * prob / (1.0 - prob)
    return 100.0 * (1.0 - prob) / prob


def _logo_data_uri() -> Optional[str]:
    if not LOGO_PATH.exists():
        return None
    try:
        data = LOGO_PATH.read_bytes()
    except OSError:
        return None
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_html(df: pd.DataFrame, title: str, template_path: Optional[Path] = None) -> str:
    template = Path(template_path or FCS_TEMPLATE_PATH)
    if not template.exists():
        raise FileNotFoundError(f"FCS HTML template not found: {template}")

    df = df.copy()
    df["start_dt"] = pd.to_datetime(df["start_date"], errors="coerce", utc=True)
    df["kickoff_display"] = df["start_dt"].dt.tz_convert("US/Eastern").dt.strftime("%a %m/%d %I:%M %p ET")

    rows: list[dict[str, object]] = []
    providers_seen: set[str] = set()

    for row in df.itertuples(index=False):
        kickoff_dt = getattr(row, "start_dt", None)
        kickoff_sort = int(kickoff_dt.value) if isinstance(kickoff_dt, pd.Timestamp) and pd.notna(kickoff_dt) else None

        providers_raw = getattr(row, "market_providers", "") or ""
        providers = [p.strip() for p in str(providers_raw).split(",") if p.strip()]
        providers_seen.update(providers)

        model_spread = _safe_float(getattr(row, "model_spread", None))
        market_spread = _safe_float(getattr(row, "market_spread", None))
        spread_edge = _safe_float(getattr(row, "spread_edge", None))
        spread_direction = getattr(row, "spread_edge_direction", None)
        spread_direction = spread_direction.lower() if isinstance(spread_direction, str) else None
        sigma = _safe_float(getattr(row, "prob_sigma", None))
        kelly_fraction = _kelly_fraction(model_spread, market_spread, sigma, spread_direction)

        model_total = _safe_float(getattr(row, "model_total", None))
        market_total = _safe_float(getattr(row, "market_total", None))
        total_edge = _safe_float(getattr(row, "total_edge", None))

        notes: list[str] = []
        provider_count = int(getattr(row, "market_provider_count", 0) or 0)
        if provider_count == 0:
            notes.append("No market lines")
        elif provider_count == 1:
            notes.append("Single book")
        if spread_edge is not None and abs(spread_edge) < 1.0:
            notes.append("Spread edge < 1")
        if total_edge is not None and abs(total_edge) < 1.0:
            notes.append("Total edge < 1")

        rows.append(
            {
                "kickoff": getattr(row, "kickoff_display", ""),
                "kickoffSort": kickoff_sort,
                "game": f"{getattr(row, 'away_team')} @ {getattr(row, 'home_team')}",
                "homeTeam": getattr(row, "home_team"),
                "awayTeam": getattr(row, "away_team"),
                "modelSpread": model_spread,
                "marketSpread": market_spread,
                "spreadEdge": spread_edge,
                "spreadBucket": _edge_bucket(spread_edge),
                "spreadDirection": spread_direction,
                "modelTotal": model_total,
                "marketTotal": market_total,
                "totalEdge": total_edge,
                "totalBucket": _edge_bucket(total_edge),
                "homeWinProb": _safe_float(getattr(row, "model_home_win_prob", None)),
                "modelHomePoints": _safe_float(getattr(row, "model_home_points", None)),
                "modelAwayPoints": _safe_float(getattr(row, "model_away_points", None)),
                "modelHomeML": _safe_float(getattr(row, "model_home_ml", None)),
                "modelAwayML": _safe_float(getattr(row, "model_away_ml", None)),
                "marketPrimary": getattr(row, "market_primary_provider", None) or "—",
                "providers": providers,
                "providerCount": provider_count,
                "notes": notes,
                "kelly": kelly_fraction * 0.25 if kelly_fraction is not None else None,
                "winProbEdge": _safe_float(getattr(row, "win_prob_edge", None)),
                "homeWinML": _prob_to_moneyline(_safe_float(getattr(row, "model_home_win_prob", None))),
            }
        )

    meta = {
        "title": title,
        "generated": datetime.now(timezone.utc).isoformat(),
        "gameCount": len(rows),
        "providerCount": len(providers_seen),
        "providers": sorted(providers_seen),
        "logo": _logo_data_uri(),
    }

    def _fmt_number(value: Optional[float], *, signed: bool = False, precision: int = 1) -> str:
        if value is None or not math.isfinite(value):
            return "—"
        fmt = f"{{:{'+' if signed else ''}.{precision}f}}"
        return fmt.format(value)

    def _fmt_percent(value: Optional[float]) -> str:
        if value is None or not math.isfinite(value):
            return "—"
        return f"{value * 100:.1f}%"

    def _fmt_moneyline(value: Optional[float]) -> str:
        if value is None or not math.isfinite(value):
            return "—"
        return f"{value:+.0f}"

    def _edge_badge(value: Optional[float], direction: Optional[str], bucket: Optional[str], label: str) -> str:
        if value is None or not math.isfinite(value):
            return '<span class="chip muted">—</span>'
        classes = ["edge-badge"]
        classes.append(direction or "total")
        if bucket:
            classes.append(bucket)
        prefix = "Home" if direction == "home" else "Away" if direction == "away" else label
        return (
            f"<span class='{' '.join(classes)}'><span>{prefix}</span><span>{_fmt_number(value, signed=True)}</span></span>"
        )

    def _providers_html(providers: list[str]) -> str:
        if not providers:
            return '<span class="chip muted">—</span>'
        return "".join(f"<span class='chip'>{name}</span>" for name in providers)

    def _notes_html(notes: list[str]) -> str:
        if not notes:
            return '<span class="note muted">—</span>'
        return "".join(f"<span class='note'>{note}</span>" for note in notes)

    def _scoreline(row_dict: dict[str, object]) -> str:
        home = _fmt_number(row_dict.get("modelHomePoints"), signed=False)
        away = _fmt_number(row_dict.get("modelAwayPoints"), signed=False)
        return f"{row_dict.get('homeTeam')} {home} • {row_dict.get('awayTeam')} {away}"

    rows_html = []
    for row_dict in rows:
        spread_badge = _edge_badge(row_dict.get("spreadEdge"), row_dict.get("spreadDirection"), row_dict.get("spreadBucket"), "Spread")
        total_badge = _edge_badge(row_dict.get("totalEdge"), "total", row_dict.get("totalBucket"), "Total")
        notes_html = _notes_html(row_dict.get("notes", []) or [])
        providers_html = _providers_html(row_dict.get("providers", []) or [])
        rows_html.append(
            """
            <tr>
              <td>{kickoff}</td>
              <td>
                <div class="matchup">{game}</div>
                <div class="scoreline">{scoreline}</div>
              </td>
              <td>{model_spread}</td>
              <td>{market_spread}</td>
              <td>{spread_badge}</td>
              <td>{model_total}</td>
              <td>{market_total}</td>
              <td>{total_badge}</td>
              <td class="win-prob">{home_win_percent}<span>Home win</span></td>
              <td>{home_ml}</td>
              <td>{providers}</td>
              <td><div class="notes">{notes}</div></td>
              <td>{kelly}</td>
            </tr>
            """.format(
                kickoff=row_dict.get("kickoff") or "TBD",
                game=row_dict.get("game") or "",
                scoreline=_scoreline(row_dict),
                model_spread=_fmt_number(row_dict.get("modelSpread"), signed=True),
                market_spread=_fmt_number(row_dict.get("marketSpread"), signed=True),
                spread_badge=spread_badge,
                model_total=_fmt_number(row_dict.get("modelTotal")),
                market_total=_fmt_number(row_dict.get("marketTotal")),
                total_badge=total_badge,
                home_win_percent=_fmt_percent(row_dict.get("homeWinProb")),
                home_ml=_fmt_moneyline(row_dict.get("homeWinML")),
                providers=providers_html,
                notes=notes_html,
                kelly=_fmt_percent(row_dict.get("kelly")),
            )
        )

    rows_fallback = "\n".join(rows_html) if rows_html else "<tr><td colspan='13'>No games in window.</td></tr>"

    data_json = json.dumps(rows, separators=(",", ":"))
    meta_json = json.dumps(meta, separators=(",", ":"))

    html = template.read_text(encoding="utf-8")
    html = html.replace("__FCS_DATA__", data_json)
    html = html.replace("__FCS_META__", meta_json)
    html = html.replace("__FCS_ROWS__", rows_fallback)
    return html


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
        html = _render_html(df, title=f"FCS Slate {anchor} to {end_date}", template_path=args.template)
        args.html.write_text(html, encoding="utf-8")
        print(f"Saved HTML summary to {args.html}")

    if not args.output and not args.html:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
