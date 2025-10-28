"""Simulate an entire FBS week with opponent-adjusted power updates."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import pandas as pd

import fbs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate all FBS games for a given week.")
    parser.add_argument("week", type=int, help="Target week to simulate.")
    parser.add_argument("--year", type=int, default=2024, help="Season year (default 2024).")
    parser.add_argument("--api-key", type=str, help="Optional CFBD API key (defaults to env).")
    parser.add_argument("--neutral-default", action="store_true",
                        help="Treat games with missing neutral flag as neutral.")
    parser.add_argument(
        "--season-type",
        choices=["regular", "postseason"],
        default="regular",
        help="Season type to query (default regular).",
    )
    parser.add_argument(
        "--include-completed",
        action="store_true",
        help="Include games that CFBD already marks as completed (useful for retrospective sims).",
    )
    parser.add_argument("--output", type=Path, help="Optional CSV output path.")
    parser.add_argument("--html", type=Path, help="Optional HTML summary output path (shareable).")
    return parser.parse_args()


def _format_bullet_rows(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>None.</p>"
    items = []
    for _, row in df.iterrows():
        weather_bits = []
        condition = row.get('weather_condition')
        if isinstance(condition, str) and condition:
            weather_bits.append(condition.title())
        temp = row.get('weather_temp')
        if pd.notna(temp):
            weather_bits.append(f"{temp:.0f}F")
        wind = row.get('weather_wind')
        if pd.notna(wind):
            weather_bits.append(f"wind {wind:.0f} mph")
        market_bits = []
        market_spread = row.get("market_spread")
        market_total = row.get("market_total")
        spread_vs_market = row.get("spread_vs_market")
        total_vs_market = row.get("total_vs_market")
        if pd.notna(market_spread):
            market_bits.append(f"Mkt Spread {market_spread:+.1f}")
        if pd.notna(spread_vs_market):
            market_bits.append(f"Δ {spread_vs_market:+.1f}")
        if pd.notna(market_total):
            market_bits.append(f"Mkt Total {market_total:.1f}")
        if pd.notna(total_vs_market):
            market_bits.append(f"Δ {total_vs_market:+.1f}")
        market_text = f" | {' / '.join(market_bits)}" if market_bits else ""
        weather_text = f" | Weather {' / '.join(weather_bits)}" if weather_bits else ""
        items.append(
            f"<li><strong>{row['home_team']}</strong> vs <strong>{row['away_team']}</strong>"
            f" | Spread {row['spread_home_minus_away']:+.1f}"
            f" | Total {row['total_points']:.1f}"
            f" | Home Win {row['home_win_prob']*100:0.1f}%{market_text}{weather_text}</li>"
        )
    return "<ul>" + "".join(items) + "</ul>"


def _render_html(df: pd.DataFrame, *, title: str) -> str:
    table = df[[
        "start_date",
        "home_team",
        "away_team",
        "spread_home_minus_away",
        "total_points",
        "home_points",
        "away_points",
        "home_win_prob",
        "home_moneyline",
        "away_moneyline",
        "market_spread",
        "market_total",
        "spread_vs_market",
        "total_vs_market",
        "weather_condition",
        "weather_temp",
        "weather_wind",
        "weather_total_adj",
    ]].copy()
    table.columns = [
        "Kickoff (UTC)",
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
            "Temp (F)": "{:.0f}",
            "Wind (mph)": "{:.0f}",
            "Total Adj": "{:+.1f}",
        })
        .hide(axis="index")
    )
    table_html = styled.to_html()

    close_games = df[df["spread_home_minus_away"].abs() <= 3].sort_values("home_win_prob", ascending=False)
    shootouts = df.sort_values("total_points", ascending=False).head(5)
    slogs = df.sort_values("total_points").head(5)
    upsets = df[df["home_win_prob"] <= 0.40].sort_values("home_win_prob").head(5)

    sections = {
        "Close calls (≤ 3 pts)": _format_bullet_rows(close_games),
        "Projected shootouts": _format_bullet_rows(shootouts),
        "Defensive slogs": _format_bullet_rows(slogs),
        "Upset radar": _format_bullet_rows(upsets),
    }

    section_html = "".join(
        f"<section><h2>{heading}</h2>{body}</section>" for heading, body in sections.items()
    )

    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>{title}</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 2rem; background: #fafafa; color: #222; }}
          h1 {{ margin-bottom: 0.5rem; }}
          section {{ margin-bottom: 1.5rem; }}
          table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
          th, td {{ padding: 0.4rem 0.6rem; text-align: center; }}
          ul {{ margin: 0.4rem 0 0.4rem 1.2rem; }}
        </style>
      </head>
      <body>
        <h1>{title}</h1>
        <p>Projection run generated via CFBD advanced metrics with opponent-adjusted power updates.</p>
        {section_html}
        <section>
          <h2>Full Board</h2>
          {table_html}
        </section>
      </body>
    </html>
    """
    return html


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD API key required. Set CFBD_API_KEY or pass --api-key.")

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

    market_lookup = fbs.fetch_market_lines(
        args.year,
        api_key,
        week=args.week,
        season_type=args.season_type,
        classification="fbs",
    )

    projections: Dict[str, list] = {
        "game_id": [],
        "week": [],
        "start_date": [],
        "home_team": [],
        "away_team": [],
        "neutral": [],
        "spread_home_minus_away": [],
        "total_points": [],
        "home_points": [],
        "away_points": [],
        "home_win_prob": [],
        "home_moneyline": [],
        "away_moneyline": [],
        "market_spread": [],
        "market_total": [],
        "market_provider_count": [],
        "market_providers": [],
        "spread_vs_market": [],
        "total_vs_market": [],
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
        if game.get("completed") and not args.include_completed:
            continue
        neutral = game.get("neutralSite")
        if neutral is None:
            neutral = args.neutral_default
        try:
            result = book.predict(game["homeTeam"], game["awayTeam"], neutral_site=neutral)
        except KeyError:
            continue
        market = market_lookup.get(game.get("id"))
        result = fbs.apply_market_prior(result, market, prob_sigma=book.prob_sigma)

        weather = weather_lookup.get(game.get("id")) or game.get("weather") or {}
        result = fbs.apply_weather_adjustment(result, weather)
        projections["game_id"].append(game["id"])
        projections["week"].append(game.get("week"))
        projections["start_date"].append(game.get("startDate"))
        projections["home_team"].append(result["home_team"])
        projections["away_team"].append(result["away_team"])
        projections["neutral"].append(neutral)
        projections["spread_home_minus_away"].append(result["spread_home_minus_away"])
        projections["total_points"].append(result["total_points"])
        projections["home_points"].append(result["home_points"])
        projections["away_points"].append(result["away_points"])
        projections["home_win_prob"].append(result["home_win_prob"])
        projections["home_moneyline"].append(result["home_moneyline"])
        projections["away_moneyline"].append(result["away_moneyline"])
        projections["market_spread"].append(result.get("market_spread"))
        projections["market_total"].append(result.get("market_total"))
        providers = result.get("market_providers") or []
        projections["market_provider_count"].append(len(providers))
        projections["market_providers"].append(", ".join(providers))
        projections["spread_vs_market"].append(result.get("spread_vs_market"))
        projections["total_vs_market"].append(result.get("total_vs_market"))
        projections["weather_condition"].append(
            result.get("weather_condition")
            or weather.get("weatherCondition")
            or weather.get("condition")
        )
        projections["weather_temp"].append(weather.get("temperature"))
        projections["weather_wind"].append(weather.get("windSpeed"))
        projections["weather_total_adj"].append(result.get("weather_total_adj"))

    if not projections["game_id"]:
        print("No upcoming games found for the specified week.")
        return

    df = pd.DataFrame(projections)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved projections to {args.output}")
    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        html = _render_html(df, title=f"FBS Week {args.week} Simulations")
        args.html.write_text(html, encoding="utf-8")
        print(f"Saved summary HTML to {args.html}")
    if not args.output and not args.html:
        pd.set_option("display.float_format", lambda v: f"{v:0.3f}")
        print(df.sort_values("home_win_prob", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
