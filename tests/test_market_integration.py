import datetime as dt
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("yaml")

import market_anchor
import oddslogic_loader
from cfb.io import oddslogic as oddslogic_io
from cfb.config import load_config


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


def test_coverage_summary_handles_multi_classification():
    kickoff = dt.date(2024, 8, 24)
    df = pd.DataFrame(
        [
            {
                "kickoff_date": kickoff,
                "date": "2024-08-24",
                "classification": "fbs",
                "home_key": "TEAMA",
                "away_key": "TEAMB",
                "row_type": "spread",
                "sportsbook_id": 9,
                "sportsbook_name": "Circa",
            },
            {
                "kickoff_date": kickoff,
                "date": "2024-08-24",
                "classification": "fbs",
                "home_key": "TEAMA",
                "away_key": "TEAMB",
                "row_type": "spread",
                "sportsbook_id": 47,
                "sportsbook_name": "Pinnacle",
            },
            {
                "kickoff_date": kickoff,
                "date": "2024-08-24",
                "classification": "fcs",
                "home_key": "TEAMX",
                "away_key": "TEAMY",
                "row_type": "spread",
                "sportsbook_id": 13,
                "sportsbook_name": "South Point",
            },
            {
                "kickoff_date": kickoff,
                "date": "2024-08-24",
                "classification": "fcs",
                "home_key": "TEAMX",
                "away_key": "TEAMY",
                "row_type": "total",
                "sportsbook_id": 13,
                "sportsbook_name": "South Point",
            },
        ]
    )

    coverage, providers = oddslogic_loader.summarize_coverage(df)

    assert set(coverage["classification"]) == {"fbs", "fcs"}
    fbs_row = coverage.loc[coverage["classification"] == "fbs"].iloc[0]
    fcs_row = coverage.loc[coverage["classification"] == "fcs"].iloc[0]
    assert fbs_row["games"] == 1
    assert fbs_row["avg_providers"] == pytest.approx(2.0)
    assert fbs_row["pct_single_provider"] == 0.0
    assert fcs_row["games"] == 1
    assert fcs_row["avg_providers"] == pytest.approx(1.0)
    assert fcs_row["pct_single_provider"] == 1.0

    assert set(providers["sportsbook_name"]) == {"Circa", "Pinnacle", "South Point"}


def test_summarize_live_lines_builds_provider_lookup():
    kickoff = dt.date(2025, 10, 30)
    rows = [
        {
            "classification": "fbs",
            "row_type": "spread",
            "kickoff_date": kickoff,
            "home_key": "TEAMHOME",
            "away_key": "TEAMAWAY",
            "provider_label": "Circa",
            "sportsbook_name": "Circa",
            "sportsbook_id": 9,
            "line_value": -3.5,
            "line_price": -110,
            "timestamp": 200,
            "is_opener": False,
            "start_datetime": "2025-10-30 23:30:00",
            "home_team": "Team Home",
            "away_team": "Team Away",
        },
        {
            "classification": "fbs",
            "row_type": "spread",
            "kickoff_date": kickoff,
            "home_key": "TEAMHOME",
            "away_key": "TEAMAWAY",
            "provider_label": "Circa",
            "sportsbook_name": "Circa",
            "sportsbook_id": 9,
            "line_value": -4.0,
            "line_price": -115,
            "timestamp": 100,
            "is_opener": True,
            "start_datetime": "2025-10-30 23:30:00",
            "home_team": "Team Home",
            "away_team": "Team Away",
        },
        {
            "classification": "fbs",
            "row_type": "total",
            "kickoff_date": kickoff,
            "home_key": "TEAMHOME",
            "away_key": "TEAMAWAY",
            "provider_label": "Circa",
            "sportsbook_name": "Circa",
            "sportsbook_id": 9,
            "line_value": 52.5,
            "line_price": -108,
            "timestamp": 210,
            "is_opener": False,
            "start_datetime": "2025-10-30 23:30:00",
            "home_team": "Team Home",
            "away_team": "Team Away",
        },
        {
            "classification": "fbs",
            "row_type": "total",
            "kickoff_date": kickoff,
            "home_key": "TEAMHOME",
            "away_key": "TEAMAWAY",
            "provider_label": "Circa",
            "sportsbook_name": "Circa",
            "sportsbook_id": 9,
            "line_value": 53.5,
            "line_price": -110,
            "timestamp": 90,
            "is_opener": True,
            "start_datetime": "2025-10-30 23:30:00",
            "home_team": "Team Home",
            "away_team": "Team Away",
        },
    ]
    df = pd.DataFrame(rows)
    lookup = oddslogic_io.summarize_live_lines(df, "fbs")
    key = (kickoff, "TEAMHOME", "TEAMAWAY")
    assert key in lookup
    entry = lookup[key]
    assert entry["spread"] == pytest.approx(-3.5)
    assert entry["open_spread"] == pytest.approx(-4.0)
    provider = entry["providers"]["Circa"]
    assert provider["spread_value"] == pytest.approx(-3.5)
    assert provider["open_spread_value"] == pytest.approx(-4.0)
    assert provider["total_value"] == pytest.approx(52.5)


def test_load_config_defaults(tmp_path):
    config = load_config()
    assert pytest.approx(config["fbs"]["market"]["spread_weight"]) == 0.4
    assert "weather" in config["fbs"]
