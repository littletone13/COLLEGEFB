import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

import market_anchor
import oddslogic_loader


def test_build_closing_lookup_prefers_sharp_provider():
    kickoff = dt.date(2024, 8, 24)
    data = [
        {
            "kickoff_date": kickoff,
            "date": "2024-08-24",
            "game_number": 305,
            "home_key": "TEAMA",
            "away_key": "TEAMB",
            "home_team": "Team A",
            "away_team": "Team B",
            "classification": "fbs",
            "row_type": "spread",
            "line_value": -6.5,
            "line_price": -110,
            "timestamp": 1724520000,
            "sportsbook_id": 9,
            "sportsbook_name": "Circa",
        },
        {
            "kickoff_date": kickoff,
            "date": "2024-08-24",
            "game_number": 305,
            "home_key": "TEAMA",
            "away_key": "TEAMB",
            "home_team": "Team A",
            "away_team": "Team B",
            "classification": "fbs",
            "row_type": "total",
            "line_value": 54.5,
            "line_price": -110,
            "timestamp": 1724520000,
            "sportsbook_id": 9,
            "sportsbook_name": "Circa",
        },
        {
            "kickoff_date": kickoff,
            "date": "2024-08-24",
            "game_number": 305,
            "home_key": "TEAMA",
            "away_key": "TEAMB",
            "home_team": "Team A",
            "away_team": "Team B",
            "classification": "fbs",
            "row_type": "spread",
            "line_value": -4.0,
            "line_price": -105,
            "timestamp": 1724520100,
            "sportsbook_id": 99,
            "sportsbook_name": "RecBook",
        },
        {
            "kickoff_date": kickoff,
            "date": "2024-08-24",
            "game_number": 305,
            "home_key": "TEAMA",
            "away_key": "TEAMB",
            "home_team": "Team A",
            "away_team": "Team B",
            "classification": "fbs",
            "row_type": "total",
            "line_value": 55.0,
            "line_price": -108,
            "timestamp": 1724520100,
            "sportsbook_id": 99,
            "sportsbook_name": "RecBook",
        },
    ]
    df = pd.DataFrame(data)

    lookup = oddslogic_loader.build_closing_lookup(df, "fbs")

    key = (kickoff, "TEAMA", "TEAMB")
    assert key in lookup
    entry = lookup[key]

    assert entry["spread_value"] == pytest.approx(-6.5)
    assert entry["total_value"] == pytest.approx(54.5)

    providers = entry["providers"]
    assert "Circa" in providers
    assert "RecBook" in providers
    assert providers["Circa"]["spread_value"] == pytest.approx(-6.5)
    assert providers["RecBook"]["spread_value"] == pytest.approx(-4.0)


def test_market_anchor_weights_sharp_books(monkeypatch):
    kickoff = dt.date(2024, 8, 24)
    home = "Team A"
    away = "Team B"
    home_key = oddslogic_loader.normalize_label(home)
    away_key = oddslogic_loader.normalize_label(away)

    closings = {
        (kickoff, home_key, away_key): {
            "providers": {
                "Circa": {
                    "spread_value": -6.0,
                    "sportsbook_id": 9,
                    "sportsbook_name": "Circa",
                },
                "RecBook": {
                    "spread_value": -4.0,
                    "sportsbook_id": 99,
                    "sportsbook_name": "RecBook",
                },
            }
        }
    }

    monkeypatch.setattr(market_anchor, "_load_closing_lookup", lambda config: closings)

    games = [
        {
            "id": 1,
            "startDate": "2024-08-24T16:00:00Z",
            "homeTeam": home,
            "awayTeam": away,
            "model_spread": -3.0,
        }
    ]

    config = market_anchor.MarketAnchorConfig(archive_path=Path("/tmp/dummy"))
    adjustments = market_anchor.derive_power_adjustments(games, config=config)

    assert adjustments[home] == pytest.approx(-1.3, rel=1e-3)
    assert adjustments[away] == pytest.approx(1.3, rel=1e-3)
