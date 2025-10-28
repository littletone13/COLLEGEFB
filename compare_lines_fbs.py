"""Compare model projections to current market lines for an FBS week."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

import fbs

CFBD_API = "https://api.collegefootballdata.com"


def _fetch_cfbd(path: str, *, api_key: str, params: Optional[Dict[str, str]] = None) -> list[dict]:
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(CFBD_API + path, headers=headers, params=params or {}, timeout=60)
    if resp.status_code == 401:
        raise RuntimeError("CFBD API rejected the key (401).")
    resp.raise_for_status()
    return resp.json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find edges relative to sportsbook lines for FBS games.")
    parser.add_argument("week", type=int, help="Target week.")
    parser.add_argument("--year", type=int, default=2024, help="Season year (default 2024).")
    parser.add_argument("--provider", type=str, default="DraftKings", help="Sportsbook provider (default DraftKings).")
    parser.add_argument("--api-key", type=str, help="CFBD API key if not set via CFBD_API_KEY.")
    parser.add_argument(
        "--season-type",
        choices=["regular", "postseason"],
        default="regular",
        help="Season type to request from CFBD (default regular).",
    )
    parser.add_argument("--output", type=Path, help="Optional CSV output path.")
    parser.add_argument("--html", type=Path, help="Optional HTML report path.")
    return parser.parse_args()


def fetch_lines(year: int, week: int, provider: str, *, api_key: str, season_type: str = "regular") -> pd.DataFrame:
    records = []
    for entry in _fetch_cfbd(
        "/lines",
        api_key=api_key,
        params={
            "year": year,
            "week": week,
            "seasonType": season_type,
            "provider": provider.lower(),
        },
    ):
        for line in entry.get("lines", []):
            records.append(
                {
                    "game_id": entry.get("id"),
                    "provider": line.get("provider"),
                    "spread": line.get("spread"),
                    "over_under": line.get("overUnder"),
                    "home_moneyline": line.get("homeMoneyline"),
                    "away_moneyline": line.get("awayMoneyline"),
                }
            )
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.drop_duplicates(subset="game_id", keep="last")
    return df.set_index("game_id")


def moneyline_prob(ml: Optional[float]) -> Optional[float]:
    if ml is None or math.isnan(ml):
        return None
    ml = float(ml)
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return -ml / (-ml + 100.0)


def build_edges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["line_spread_home"] = df["line_spread"].apply(lambda x: None if pd.isna(x) else -float(x))
    df["spread_edge"] = df.apply(
        lambda row: row["pred_spread"] - row["line_spread_home"] if row["line_spread_home"] is not None else float("nan"),
        axis=1,
    )
    df["total_edge"] = df["pred_total"] - df["line_total"]

    df["home_implied"] = df["home_moneyline"].apply(moneyline_prob)
    df["away_implied"] = df["away_moneyline"].apply(moneyline_prob)
    df["home_prob_edge"] = df["pred_home_win"] - df["home_implied"]
    df["away_prob_edge"] = (1 - df["pred_home_win"]) - df["away_implied"]

    df["edge_side"] = df["spread_edge"].apply(lambda x: "Home" if x >= 0 else "Away")
    df["edge_spread_abs"] = df["spread_edge"].abs()
    df["total_edge_abs"] = df["total_edge"].abs()
    return df


def _fmt(value: Optional[float], fmt: str, *, default: str = "N/A") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return format(value, fmt)


def render_html(edges: pd.DataFrame, *, title: str) -> str:
    table = edges[[
        "start_date",
        "home_team",
        "away_team",
        "line_spread_home",
        "pred_spread",
        "spread_edge",
        "line_total",
        "pred_total",
        "total_edge",
        "pred_home_win",
        "home_moneyline",
        "home_prob_edge",
        "weather_condition",
        "weather_temp",
        "weather_wind",
        "weather_total_adj",
    ]].copy()
    table.columns = [
        "Kickoff (UTC)",
        "Home",
        "Away",
        "Line Spread (H-A)",
        "Model Spread",
        "Spread Edge",
        "Line Total",
        "Model Total",
        "Total Edge",
        "Home Win%",
        "Home ML",
        "Prob Edge",
        "Weather",
        "Temp (F)",
        "Wind (mph)",
        "Total Adj",
    ]
    table["Home Win%"] = table["Home Win%"] * 100
    table["Weather"] = table["Weather"].apply(
        lambda x: x.title() if isinstance(x, str) and x else "--"
    )
    styled = (
        table.style
        .background_gradient(subset=["Spread Edge"], cmap="RdYlGn")
        .background_gradient(subset=["Total Edge"], cmap="RdYlGn")
        .background_gradient(subset=["Prob Edge"], cmap="RdYlGn")
        .format({
            "Line Spread (H-A)": "{:+.1f}",
            "Model Spread": "{:+.1f}",
            "Spread Edge": "{:+.1f}",
            "Line Total": "{:.1f}",
            "Model Total": "{:.1f}",
            "Total Edge": "{:+.1f}",
            "Home Win%": "{:.1f}",
            "Home ML": "{:+.0f}",
            "Prob Edge": "{:+.3f}",
            "Temp (F)": "{:.0f}",
            "Wind (mph)": "{:.0f}",
            "Total Adj": "{:+.1f}",
        })
        .hide(axis="index")
    )

    top_spread = edges.sort_values("edge_spread_abs", ascending=False).head(5)
    top_total = edges.sort_values("total_edge_abs", ascending=False).head(5)
    top_prob = edges.assign(prob_edge_abs=edges["home_prob_edge"].abs()).sort_values("prob_edge_abs", ascending=False).head(5)

    def _weather_bits(row: pd.Series) -> str:
        bits = []
        cond = row.get('weather_condition')
        if isinstance(cond, str) and cond:
            bits.append(cond.title())
        temp = row.get('weather_temp')
        if pd.notna(temp):
            bits.append(f"{temp:.0f}F")
        wind = row.get('weather_wind')
        if pd.notna(wind):
            bits.append(f"wind {wind:.0f} mph")
        if pd.notna(row.get('weather_total_adj')) and row.get('weather_total_adj'):
            bits.append(f"adj {row['weather_total_adj']:+.1f}")
        return " | Weather " + " / ".join(bits) if bits else ""

    def bullet(rows: pd.DataFrame) -> str:
        items = []
        for _, row in rows.iterrows():
            line_spread = _fmt(row.get('line_spread_home'), '+.1f')
            items.append(
                f"<li><strong>{row['home_team']}</strong> vs <strong>{row['away_team']}</strong>"
                f" | Model Spread {row['pred_spread']:+.1f}"
                f" | Line {line_spread}"
                f" | Edge {row['spread_edge']:+.1f}{_weather_bits(row)}</li>"
            )
        return "<ul>" + "".join(items) + "</ul>" if items else "<p>None.</p>"

    def total_bullets(rows: pd.DataFrame) -> str:
        items = []
        for _, row in rows.iterrows():
            items.append(
                f"<li><strong>{row['home_team']}</strong> vs <strong>{row['away_team']}</strong>"
                f" | Line {_fmt(row['line_total'], '.1f')}"
                f" | Model {row['pred_total']:.1f}"
                f" | Edge {row['total_edge']:+.1f}{_weather_bits(row)}</li>"
            )
        return "<ul>" + "".join(items) + "</ul>" if items else "<p>None.</p>"

    def prob_bullets(rows: pd.DataFrame) -> str:
        items = []
        for _, row in rows.iterrows():
            items.append(
                f"<li><strong>{row['home_team']}</strong> vs <strong>{row['away_team']}</strong>"
                f" | Home Win {row['pred_home_win']*100:0.1f}%"
                f" | Implied {_fmt(None if row['home_implied'] is None else row['home_implied']*100, '0.1f')}%"
                f" | Edge {row['home_prob_edge']:+.3f}{_weather_bits(row)}</li>"
            )
        return "<ul>" + "".join(items) + "</ul>" if items else "<p>None.</p>"

    html = f"""
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
        <section>
          <h2>Biggest Spread Edges</h2>
          {bullet(top_spread)}
        </section>
        <section>
          <h2>Total Edges</h2>
          {total_bullets(top_total)}
        </section>
        <section>
          <h2>Moneyline / Win Probability Edges</h2>
          {prob_bullets(top_prob)}
        </section>
        <section>
          <h2>Board</h2>
          {styled.to_html()}
        </section>
      </body>
    </html>
    """
    return html


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD API key required.")

    games = fbs.fetch_games(args.year, api_key, season_type=args.season_type)
    cal_games = [g for g in games if g.get('completed') and g.get('week', 0) < args.week]
    ratings, book = fbs.build_rating_book(
        args.year,
        api_key=api_key,
        adjust_week=args.week,
        calibration_games=cal_games,
    )
    weather_lookup = fbs.fetch_game_weather(
        args.year,
        api_key,
        week=args.week,
        season_type=args.season_type,
    )
    lines = fetch_lines(
        args.year,
        args.week,
        args.provider,
        api_key=api_key,
        season_type=args.season_type,
    )

    projections: Dict[str, list] = {
        "game_id": [],
        "start_date": [],
        "home_team": [],
        "away_team": [],
        "pred_spread": [],
        "pred_total": [],
        "pred_home_win": [],
        "weather_condition": [],
        "weather_temp": [],
        "weather_wind": [],
        "weather_total_adj": [],
    }

    for game in games:
        if game.get("week") != args.week:
            continue
        if game.get("homeClassification") != "fbs" or game.get("awayClassification") != "fbs":
            continue
        if game.get("completed"):
            continue
        try:
            pred = book.predict(game["homeTeam"], game["awayTeam"], neutral_site=game.get("neutralSite", False))
        except KeyError:
            continue
        weather = weather_lookup.get(game['id']) or game.get('weather') or {}
        pred = fbs.apply_weather_adjustment(pred, weather)
        projections["game_id"].append(game["id"])
        projections["start_date"].append(game.get("startDate"))
        projections["home_team"].append(pred["home_team"])
        projections["away_team"].append(pred["away_team"])
        projections["pred_spread"].append(pred["spread_home_minus_away"])
        projections["pred_total"].append(pred["total_points"])
        projections["pred_home_win"].append(pred["home_win_prob"])
        projections["weather_condition"].append(
            pred.get("weather_condition")
            or weather.get("weatherCondition")
            or weather.get("condition")
        )
        projections["weather_temp"].append(weather.get("temperature"))
        projections["weather_wind"].append(weather.get("windSpeed"))
        projections["weather_total_adj"].append(pred.get("weather_total_adj"))

    pred_df = pd.DataFrame(projections).set_index("game_id")
    if pred_df.empty:
        print("No upcoming FBS games for the specified week.")
        return

    combined = pred_df.join(lines, how="inner")
    if combined.empty:
        print("No matching lines for the selected provider/week.")
        return
    combined = combined.rename(columns={"spread": "line_spread", "over_under": "line_total"})

    combined = build_edges(combined)

    combined = combined.reset_index().rename(columns={
        "index": "game_id",
    })

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.output, index=False)
        print(f"Saved comparison to {args.output}")
    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        html = render_html(combined, title=f"FBS Week {args.week} Market Comparison ({args.provider})")
        args.html.write_text(html, encoding="utf-8")
        print(f"Saved HTML report to {args.html}")
    if not args.output and not args.html:
        cols = ["game_id", "home_team", "away_team", "line_spread_home", "pred_spread", "spread_edge", "line_total", "pred_total", "total_edge", "home_moneyline", "home_prob_edge"]
        print(combined[cols].sort_values("edge_spread_abs", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
